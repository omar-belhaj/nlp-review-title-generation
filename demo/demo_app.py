# Live demo: type a review and follow it through the animated model page, with values computed live by the final model.
# Works in a fresh Colab or Kaggle session. Upload nlp_demo_files.zip next to this cell. No dataset or training needed.
import base64, glob, html, json, re, subprocess, sys, zipfile, zlib
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
import gradio as gr
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FILES = Path("demo_files")
if not (FILES / "imp_combined.pt").exists():
    if Path("nlp_demo_files.zip").is_file():
        zipfile.ZipFile("nlp_demo_files.zip").extractall(FILES)
    else:  # Kaggle unpacks uploaded zip files into folders
        found = glob.glob("**/imp_combined.pt", recursive=True) + glob.glob("/kaggle/input/**/imp_combined.pt", recursive=True)
        if not found:
            raise FileNotFoundError("imp_combined.pt not found. Upload nlp_demo_files.zip or a demo_files folder.")
        FILES = Path(found[0]).parent
CFG = json.loads((FILES / "demo_config.json").read_text())
vocabularies = json.loads((FILES / "vocabularies.json").read_text())
SRC_ITOS, TGT_ITOS = vocabularies["source"], vocabularies["target"]
SRC, TGT = {t: i for i, t in enumerate(SRC_ITOS)}, {t: i for i, t in enumerate(TGT_ITOS)}
PAD, UNK, SOS, EOS = 0, 1, 2, 3
V = len(TGT_ITOS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower().replace("’", "'").replace("`", "'")
    text = re.sub(r"([.,!?])\1+", r"\1", text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?]", text)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(SRC_ITOS), CFG["emb_dim"], padding_idx=PAD)
        self.rnn = nn.LSTM(CFG["emb_dim"], CFG["hidden_dim"] // 2, CFG["n_layers"], batch_first=True, bidirectional=True)
    def forward(self, src, lengths):
        packed = pack_padded_sequence(self.embedding(src), lengths, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        merge = lambda s: torch.cat([s[0::2], s[1::2]], dim=-1)
        return outputs, (merge(h), merge(c))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        h = CFG["hidden_dim"]
        self.W_s, self.W_h, self.v = nn.Linear(h, h, bias=False), nn.Linear(h, h, bias=False), nn.Linear(h, 1, bias=False)
    def forward(self, query, memory):
        scores = self.v(torch.tanh(self.W_s(query).unsqueeze(1) + self.W_h(memory))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), memory).squeeze(1), weights

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        h, e = CFG["hidden_dim"], CFG["emb_dim"]
        self.embedding = nn.Embedding(V, e, padding_idx=PAD)
        self.attention = Attention()
        self.rnn = nn.LSTM(e + h, h, CFG["n_layers"], batch_first=True)
        self.out = nn.Linear(h * 2 + e, V)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        h = CFG["hidden_dim"]
        self.encoder, self.decoder = Encoder(), Decoder()
        self.bridge_h, self.bridge_c = nn.Linear(h, h), nn.Linear(h, h)
        self.p_gen = nn.Linear(h * 2 + CFG["emb_dim"], 1)

model = Model().to(DEVICE)
model.load_state_dict(torch.load(FILES / "imp_combined.pt", map_location=DEVICE)["state"])
model.eval()

@torch.no_grad()
def trace_review(text):
    tokens = tokenize(text)[:CFG["max_src_len"]]
    if not tokens:
        return None
    oovs, ext = [], []
    for t in tokens:
        if t in ".,!?":
            ext.append(PAD)  # punctuation cannot be copied into a title
        elif t in TGT:
            ext.append(TGT[t])
        else:
            if t not in oovs:
                oovs.append(t)
            ext.append(V + oovs.index(t))
    word_of = lambda i: oovs[i - V] if i >= V else TGT_ITOS[i]
    src = torch.tensor([[SRC.get(t, UNK) for t in tokens]], device=DEVICE)
    ext = torch.tensor([ext], device=DEVICE)
    memory, (h, c) = model.encoder(src, torch.tensor([len(tokens)]))
    half = memory.size(-1) // 2
    encoder_view = {"forward": memory[0, :, :12].cpu().numpy().round(2).tolist(),
                    "backward": memory[0, :, half:half + 12].cpu().numpy().round(2).tolist()}
    state = (torch.tanh(model.bridge_h(h)), torch.tanh(model.bridge_c(c)))
    previous, steps, title = torch.tensor([SOS], device=DEVICE), [], []
    for t in range(CFG["max_tgt_tokens"] + 1):
        input_word = "<SOS>" if t == 0 else word_of(previous.item())
        embedded = model.decoder.embedding(previous.masked_fill(previous >= V, UNK))
        context, weights = model.decoder.attention(state[0][-1], memory)
        output, state = model.decoder.rnn(torch.cat([embedded, context], dim=-1).unsqueeze(1), state)
        features = torch.cat([output.squeeze(1), context, embedded], dim=-1)
        p = torch.sigmoid(model.p_gen(features))
        generated = torch.cat([torch.softmax(model.decoder.out(features), dim=-1) * p, torch.zeros(1, len(oovs), device=DEVICE)], dim=-1)
        copied = torch.zeros_like(generated).scatter_add(1, ext, (1 - p) * weights)
        total = generated + copied
        total[:, [PAD, UNK, SOS]] = -1.0
        if t == 0:
            total[:, EOS] = -1.0
        values, indices = total[0].topk(5)
        chosen = indices[0].item()
        steps.append({"input": input_word, "input_as_unk": bool(previous.item() >= V),
                      "attention": weights[0].cpu().numpy().round(3).tolist(), "p_gen": round(p.item(), 3),
                      "top": [{"word": word_of(i), "prob": round(v, 4), "generated": round(generated[0, i].item(), 4),
                               "copied": round(copied[0, i].item(), 4), "in_title_vocab": i < V}
                              for i, v in zip(indices.tolist(), values.tolist())],
                      "chosen": word_of(chosen)})
        previous = indices[:1]
        if chosen == EOS:
            break
        title.append(word_of(chosen))
    return {"tokens": tokens, "ids": src[0].tolist(), "is_unk": [t not in SRC for t in tokens], "encoder": encoder_view,
            "steps": steps, "title": " ".join(title), "label": "Your review", "text": text, "source": "live",
            "stars": None, "reference": None, "others": {}}

# The animated page, stored compressed so this file works alone. Its readable source is demo/index.html.
PAGE = zlib.decompress(base64.b64decode("""
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
""")).decode()

def show(text):
    trace = trace_review(text)
    if trace is None:
        return "Type a review first.", ""
    data = json.dumps({"model": "imp_combined", "live": True, "reviews": [trace]}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data)
    return trace["title"], f'<iframe title="Decoding steps" srcdoc="{html.escape(page, quote=True)}" style="width:100%;height:930px;border:0;border-radius:6px"></iframe>'

with gr.Blocks(title="Review Title Machine") as demo:
    gr.Markdown("# Review Title Machine\nType a food review, generate its title, then press **Play** or **Next** to follow each step of the model.")
    review = gr.Textbox(label="Review", lines=3, value="This tea is not good at all. It tastes like nothing and I would not recommend it.")
    run = gr.Button("Generate title", variant="primary")
    title = gr.Textbox(label="Generated title")
    detail = gr.HTML()
    run.click(show, review, [title, detail])
    gr.Examples(["Love this honey, so good!", "The flavor is great, but the price is far too high for such a small box."], review)

if __name__ == "__main__" and "NLP_DEMO_TEST" not in __import__("os").environ:
    demo.launch(share=True)
