# Released under the MIT License. See LICENSE.
"""Train the TwiBot-22 Wei text baseline and score 200 paraphrased posts.

The public TwiBot-20 sample has no labels and the full release needs approval,
so this script does not read that sample as a training set. It uses post text
and bot/human labels from the MIT-licensed Zenodo deposit 10.5281/zenodo.14795885.
"""

import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline" / "wei"))

from bilstm_attention import (  # noqa: E402
    BiLSTMModel,
    batch_size,
    device,
    embedding_size,
    epochs,
    hidden_size,
    lr,
    num_classes,
    num_directions,
    num_layers,
    sentence_max_len,
    train,
    vocab_size,
)
from data_processor import DataProcessor  # noqa: E402
import torch.nn as nn  # noqa: E402

SEED = 100
N_EVAL = 200
PARAPHRASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_DIR = ROOT / "data" / "zenodo" / "valeriolagatta-botfarm-RLCSIO-8cda103" / "dataset"
GLOVE_PATH = ROOT / "data" / "glove" / "glove.twitter.27B.25d.txt"
SPLIT_PATH = ROOT / "outputs" / "heldout_split.json"
PARAPHRASE_PATH = ROOT / "outputs" / "paraphrases.jsonl"
RESULTS_PATH = ROOT / "results.json"
DATA_URI = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.I)


def readable_text(text):
    cleaned = DATA_URI.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_symbols(text):
    # Some posts end in emoji or embedded image markup. The saved original keeps
    # those characters. This fallback is only used after an empty decode.
    cleaned = re.sub(r"[^\w\s.,!?;:'\"%#@&()/\-$]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def load_labeled_posts():
    users_path = DATASET_DIR / "Users.csv"
    posts_path = DATASET_DIR / "Posts.csv"
    if not users_path.is_file() or not posts_path.is_file():
        raise SystemExit(
            "DartPost Users.csv and Posts.csv are not in data/zenodo. "
            "Download 10.5281/zenodo.14795885 and unzip it there. "
            "This script does not scrape Twitter and does not use the unlabeled TwiBot-20 sample."
        )
    users = {row["id"]: row for row in csv.DictReader(users_path.open(encoding="utf-8"))}
    posts = []
    for row in csv.DictReader(posts_path.open(encoding="utf-8")):
        user = users.get(row["AuthorId"])
        if user is None or user.get("isObserver") == "True":
            continue
        if user.get("isPuppet") == "True":
            label = "bot"
        elif user.get("isPuppet") == "False":
            label = "human"
        else:
            continue
        text = row.get("text") or ""
        if not text.strip():
            continue
        posts.append({"post_id": str(row["id"]), "text": text, "label": label})
    return posts


def write_split(posts):
    rng = random.Random(SEED)
    bots = [row for row in posts if row["label"] == "bot"]
    humans = [row for row in posts if row["label"] == "human"]
    rng.shuffle(bots)
    rng.shuffle(humans)
    n_bot = int(round(N_EVAL * len(bots) / len(posts)))
    n_human = N_EVAL - n_bot
    if n_bot > len(bots) or n_human > len(humans):
        raise SystemExit("Not enough labeled posts to hold out 200 stratified texts.")
    heldout = bots[:n_bot] + humans[:n_human]
    train_rows = bots[n_bot:] + humans[n_human:]
    records = []
    for row in train_rows:
        records.append({**row, "split": "train"})
    for row in heldout:
        records.append({**row, "split": "test"})
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(records), encoding="utf-8")
    return {
        "n_posts": len(posts),
        "n_train": len(train_rows),
        "n_eval": len(heldout),
        "n_eval_bot": n_bot,
        "n_eval_human": n_human,
        "n_train_bot": len(bots) - n_bot,
        "n_train_human": len(humans) - n_human,
    }


def paraphrase_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def paraphrase_heldout(processor):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "No paraphrase model library is installed. Install transformers or provide a local model. "
            "Scores were not invented."
        ) from exc
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL)
    except Exception as exc:
        raise SystemExit(
            "Could not load a real paraphrase model (%s). %s. "
            "Stopping without scores."
            % (PARAPHRASE_MODEL, exc)
        ) from exc
    para_device = paraphrase_device()
    model.to(para_device)
    model.eval()
    print("paraphrase device", para_device, flush=True)

    done = {}
    if PARAPHRASE_PATH.is_file():
        with PARAPHRASE_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("paraphrase_model") == PARAPHRASE_MODEL and row.get("paraphrase"):
                    done[str(row["post_id"])] = row

    PARAPHRASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = []
    with PARAPHRASE_PATH.open("a", encoding="utf-8") as handle:
        for post_id, text, label_vec in zip(
            processor.test_ids, processor.test_texts, processor.test_labels_onehot
        ):
            label = "bot" if int(np.argmax(label_vec)) == 1 else "human"
            if post_id in done:
                ordered.append(done[post_id]["paraphrase"])
                continue
            source = readable_text(text)
            if not source:
                raise SystemExit("Post %s has no readable text to paraphrase." % post_id)
            paraphrase = generate_paraphrase(tokenizer, model, source, para_device)
            if not paraphrase:
                raise SystemExit(
                    "The language model returned an empty paraphrase for post %s. "
                    "Stopping without scores." % post_id
                )
            row = {
                "post_id": post_id,
                "label": label,
                "original": text,
                "paraphrase": paraphrase,
                "paraphrase_model": PARAPHRASE_MODEL,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            ordered.append(paraphrase)
            print("paraphrased", len(ordered), "/", len(processor.test_ids), flush=True)
    return ordered


def generate_paraphrase(tokenizer, model, source, para_device):
    candidates = [source]
    stripped = strip_symbols(source)
    if stripped and normalize(stripped) != normalize(source):
        candidates.append(stripped)
    last = ""
    for text in candidates:
        decoded = decode_paraphrase(tokenizer, model, text, para_device, different_words=False)
        last = decoded or last
        if decoded and normalize(decoded) != normalize(text):
            return decoded
        decoded = decode_paraphrase(tokenizer, model, text, para_device, different_words=True)
        last = decoded or last
        if decoded and normalize(decoded) != normalize(text):
            return decoded
    return last.strip()


def decode_paraphrase(tokenizer, model, source, para_device, different_words):
    instruction = (
        "Paraphrase the user's social media post. Keep the same meaning and stance. "
        "Do not add facts. Reply with only the paraphrase."
    )
    if different_words:
        instruction += " Use different words from the original."
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": source},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    encoded = {key: value.to(para_device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=160,
            do_sample=False,
        )
    new_tokens = output[0][encoded["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = re.sub(r"^(paraphrase|rewrite)\s*:\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def classification_scores(model, features, labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size].to(device)
            pred = torch.argmax(model(batch), dim=1).cpu().numpy()
            predictions.append(pred)
    pred = np.concatenate(predictions)
    y = np.asarray(labels)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support_bot": int(np.sum(y == 1)),
        "support_human": int(np.sum(y == 0)),
    }


def main():
    if not GLOVE_PATH.is_file():
        raise SystemExit(
            "GloVe Twitter 25d is missing at %s. The Wei baseline freezes those embeddings. "
            "Download glove.twitter.27B.25d.txt from https://nlp.stanford.edu/data/glove.twitter.27B.zip."
            % GLOVE_PATH
        )
    posts = load_labeled_posts()
    split_info = write_split(posts)
    print("split", split_info, flush=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    processor = DataProcessor(str(SPLIT_PATH), str(GLOVE_PATH))
    train_datasets, test_datasets = processor.get_datasets(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_len=sentence_max_len,
    )
    processor.test_labels_onehot = test_datasets.tensors[1].numpy()
    if len(processor.test_texts) != N_EVAL:
        raise SystemExit("Expected %s held-out texts, found %s." % (N_EVAL, len(processor.test_texts)))

    paraphrases = paraphrase_heldout(processor)
    unchanged = sum(
        1 for src, dst in zip(processor.test_texts, paraphrases)
        if normalize(readable_text(src)) == normalize(dst)
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
    model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    model = train(model, train_loader, test_loader, optimizer, loss_func, epochs)

    y = test_datasets.tensors[1].argmax(dim=1).numpy()
    original = classification_scores(model, test_datasets.tensors[0], y)
    paraphrased_x = processor.encode_texts(paraphrases)
    paraphrased = classification_scores(model, paraphrased_x, y)
    results = {
        "question": (
            "After the Wei et al. text baseline is trained, how far do precision, recall, "
            "and F1 move on 200 held-out texts when those texts are paraphrased and the labels stay the same?"
        ),
        "twibot20": {
            "used": False,
            "reason": (
                "The public files are an unlabeled sample (TwiBot-20_sample.json). "
                "The original README says the full dataset is not posted and requires permission "
                "from shangbin@cs.washington.edu. The official TwiBot-20 split was not used."
            ),
        },
        "dataset": {
            "source": "Zenodo 10.5281/zenodo.14795885",
            "license": "MIT",
            "records": "Posts.csv text joined to Users.csv isPuppet",
            "bot": "isPuppet true (sockpuppet accounts)",
            "human": "isPuppet false and isObserver false (DartPost participants)",
            "excluded": "isObserver true accounts have no posts and are not treated as human labels",
            "positive_class": "bot",
            "split": "stratified holdout of 200 posts, seed 100. This is not an official TwiBot split.",
            **split_info,
        },
        "baseline": {
            "name": "Wei et al. BiLSTM with attention",
            "upstream_file": "LuoUndergradXJTU/TwiBot-22 src/Wei/Twibot-20/bilstm_attention.py",
            "embedding": "frozen GloVe Twitter 25d",
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "sentence_max_len": sentence_max_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "deviation": "LSTM initial state is zeros. Upstream samples it with torch.randn on every forward pass.",
        },
        "paraphrase_model": PARAPHRASE_MODEL,
        "n_paraphrases_unchanged_from_readable_text": unchanged,
        "original_200": original,
        "paraphrased_200": paraphrased,
        "delta_paraphrased_minus_original": {
            "precision": paraphrased["precision"] - original["precision"],
            "recall": paraphrased["recall"] - original["recall"],
            "f1": paraphrased["f1"] - original["f1"],
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["delta_paraphrased_minus_original"], indent=2))
    print("wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
