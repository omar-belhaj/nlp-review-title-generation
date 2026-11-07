# Rebuilds the manual annotation sheet after a session reset, with no training.
# Needs: the Amazon dataset (as in the notebook) and imp_combined.pt, vocabularies.json, demo_config.json
# in a demo_files folder, or nlp_demo_files.zip next to this cell. Takes a few minutes.
import glob, html, json, os, re, subprocess, sys, zipfile
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rouge-score", "kagglehub"], check=True)
import pandas as pd
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from sklearn.model_selection import GroupShuffleSplit
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
def generate(text):
    tokens = tokenize(text)[:CFG["max_src_len"]]
    if not tokens:
        return "", [], []
    oovs, ext = [], []
    for t in tokens:
        if t in ".,!?":
            ext.append(PAD)
        elif t in TGT:
            ext.append(TGT[t])
        else:
            if t not in oovs:
                oovs.append(t)
            ext.append(V + oovs.index(t))
    src = torch.tensor([[SRC.get(t, UNK) for t in tokens]], device=DEVICE)
    ext = torch.tensor([ext], device=DEVICE)
    memory, (h, c) = model.encoder(src, torch.tensor([len(tokens)]))
    state = (torch.tanh(model.bridge_h(h)), torch.tanh(model.bridge_c(c)))
    previous, title, steps = torch.tensor([SOS], device=DEVICE), [], []
    for t in range(CFG["max_tgt_tokens"] + 1):
        embedded = model.decoder.embedding(previous.masked_fill(previous >= V, UNK))
        context, weights = model.decoder.attention(state[0][-1], memory)
        output, state = model.decoder.rnn(torch.cat([embedded, context], dim=-1).unsqueeze(1), state)
        features = torch.cat([output.squeeze(1), context, embedded], dim=-1)
        p = torch.sigmoid(model.p_gen(features))
        probs = torch.cat([torch.softmax(model.decoder.out(features), dim=-1) * p, torch.zeros(1, len(oovs), device=DEVICE)], dim=-1)
        probs = probs.scatter_add(1, ext, (1 - p) * weights)
        probs[:, [PAD, UNK, SOS]] = -1
        if t == 0:
            probs[:, EOS] = -1
        best = probs.argmax(-1)
        word = oovs[best.item() - V] if best.item() >= V else TGT_ITOS[best.item()]
        steps.append({"word": word, "prob": probs[0, best].item(), "p_gen": p.item(), "attention": weights[0].tolist()})
        if best.item() == EOS:
            break
        title.append(word)
        previous = best
    return " ".join(title), tokens, steps

# Same cleaning, filters, sample and split as the notebook, so these are the same test reviews
def find_dataset():
    candidates = glob.glob("/kaggle/input/**/Reviews.csv", recursive=True) + glob.glob("Reviews.csv")
    if candidates:
        return candidates[0]
    import kagglehub
    return str(Path(kagglehub.dataset_download("snap/amazon-fine-food-reviews")) / "Reviews.csv")

def normalize(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower().replace("’", "'").replace("`", "'")
    return re.sub(r"([.,!?])\1+", r"\1", text)

def tokenize_target(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize(text))

raw = pd.read_csv(os.environ.get("NLP_DATA_PATH") or find_dataset(), usecols=["ProductId", "Score", "Summary", "Text"])
data = raw.dropna(subset=["Text", "Summary"]).copy()
data["norm_text"] = data["Text"].map(normalize)
data = data.drop_duplicates("norm_text").drop(columns="norm_text")
data["src_len"] = data["Text"].map(lambda t: len(tokenize(t)))
data["tgt_len"] = data["Summary"].map(lambda t: len(tokenize_target(t)))
data = data[data["src_len"].between(CFG["min_src_tokens"], CFG["max_src_tokens_filter"])
            & data["tgt_len"].between(CFG["min_tgt_tokens"], CFG["max_tgt_tokens"])]
data = data.sample(min(CFG["n_samples"], len(data)), random_state=CFG["seed"]).reset_index(drop=True)
_, holdout_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG["seed"]).split(data, groups=data["ProductId"]))
holdout = data.iloc[holdout_idx]
_, test_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=CFG["seed"]).split(holdout, groups=holdout["ProductId"]))
test_df = holdout.iloc[test_rel].reset_index(drop=True)
print("Test reviews:", len(test_df))

sheet = test_df.sample(min(50, len(test_df)), random_state=CFG["seed"])[["Score", "Text", "Summary"]].copy()
sheet["reference"] = sheet["Summary"].map(lambda t: " ".join(tokenize_target(t)))
sheet["generated"] = [generate(text)[0] for text in sheet["Text"]]
scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
sheet["rouge1"] = [scorer.score(r, g)["rouge1"].fmeasure for r, g in zip(sheet["reference"], sheet["generated"])]
sheet["acceptable"] = ""
sheet = sheet.drop(columns="Summary")
sheet.to_csv("manual_annotation_sheet.csv", index=False)
print(f"Saved manual_annotation_sheet.csv: {len(sheet)} reviews, mean ROUGE-1 {sheet.rouge1.mean():.3f}")
print(sheet[["reference", "generated"]].head(5).to_string())
try:
    from google.colab import files
    files.download("manual_annotation_sheet.csv")
except ImportError:
    from IPython.display import FileLink, display
    display(FileLink("manual_annotation_sheet.csv"))
