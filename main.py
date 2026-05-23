import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# =========================
# CONFIG
# =========================
TRAIN_PATH = "data/TCR-Processed-Raw.csv"
TEST_PATH = "data/test_set.csv"
OUTPUT_PATH = "submission.csv"

N_SPLITS = 5
RANDOM_STATE = 42

CLASSES = ["viral", "bacterial", "cancer", "autoimmune"]

# =========================
# LABEL CLEANING
# =========================
def map_label(text):
    if pd.isna(text):
        return None
    t = str(text).lower()

    viral_keys = ["influenza", "cmv", "cytomegalovirus", "hiv", "ebv", "epstein", "htlv", "covid", "sars"]
    bacterial_keys = ["tuberculosis", "m. tuberculosis", "staphyl", "strept", "e. coli", "bacterial"]
    cancer_keys = ["melanoma", "tumor", "taa", "neoantigen", "cancer", "carcinoma"]
    autoimmune_keys = ["diabetes", "type 1", "autoimmune", "parkinson"]

    if any(k in t for k in viral_keys):
        return "viral"
    if any(k in t for k in bacterial_keys):
        return "bacterial"
    if any(k in t for k in cancer_keys):
        return "cancer"
    if any(k in t for k in autoimmune_keys):
        return "autoimmune"

    return None


# =========================
# SEQUENCE CLEANING
# =========================
def clean_seq(seq):
    if pd.isna(seq):
        return ""
    return re.sub(r"[^A-Z]", "", str(seq).upper())


# =========================
# FEATURE ENGINEERING
# =========================
def build_sequence(row):
    seq = clean_seq(row.get("CDR3.beta.aa", ""))
    v = str(row.get("TRBV", "")).replace("nan", "").upper()
    j = str(row.get("TRBJ", "")).replace("nan", "").upper()

    # include gene context as tokens
    return f"{seq} V{v} J{j}".strip()


# =========================
# LOAD DATA
# =========================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["label"] = train["Pathology"].apply(map_label)
train = train.dropna(subset=["label"]).reset_index(drop=True)

# encode labels
label2id = {c: i for i, c in enumerate(CLASSES)}
y = train["label"].map(label2id).values

# build text features
train_text = train.apply(build_sequence, axis=1)
test_text = test.apply(build_sequence, axis=1)

# =========================
# VECTORIZE (k-mer TF-IDF)
# =========================
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    min_df=2,
    max_features=100000
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# =========================
# MODEL
# =========================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    multi_class="multinomial",
    n_jobs=-1
)

# =========================
# CROSS VALIDATION (sanity check)
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros((len(train), len(CLASSES)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"\nFold {fold + 1}")

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)

    oof[val_idx] = preds

    val_pred = preds.argmax(axis=1)
    f1 = f1_score(y_val, val_pred, average="macro")
    print("Macro F1:", f1)

# overall CV score
oof_pred = oof.argmax(axis=1)
print("\nCV Macro F1:", f1_score(y, oof_pred, average="macro"))

# =========================
# HOLDOUT CHECK (NEW ADDITION)
# =========================
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_tr, y_tr)
pred = model.predict(X_val)

print("Holdout F1:", f1_score(y_val, pred, average="macro"))

# =========================
# FINAL TRAIN
# =========================
model.fit(X_train, y)

# =========================
# PREDICT TEST
# =========================
test_preds = model.predict_proba(X_test)

# =========================
# SUBMISSION
# =========================
submission = pd.DataFrame({
    "ID": test["ID"],
    "prediction": [CLASSES[i] for i in test_preds.argmax(axis=1)]
})

submission.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)