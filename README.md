# Fork: paraphrase check on one TwiBot text baseline

This repository is a fork. It is not a new bot detector.

The starting point is the public [TwiBot-20](https://github.com/BunsenFeng/TwiBot-20) sample and the text-only Wei baseline from the [TwiBot-22 benchmark stack](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/Wei/Twibot-20). The dataset paper is Shangbin Feng, Herun Wan, Ningnan Wang, Jundong Li, and Minnan Luo, “TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark,” CIKM 2021, [arXiv:2106.13088](https://arxiv.org/abs/2106.13088). The benchmark paper is Shangbin Feng et al., “TwiBot-22: Towards Graph-Based Twitter Bot Detection,” [arXiv:2206.04564](https://arxiv.org/abs/2206.04564). The reused model is Feng Wei and Uyen Trang Nguyen, “Twitter Bot Detection Using Bidirectional Long Short-term Memory Neural Networks and Word Embeddings,” [arXiv:2002.01336](https://arxiv.org/abs/2002.01336).

## What was added

One evaluation the benchmark does not report: after that text baseline is trained, how far precision, recall, and F1 move when 200 held-out texts are paraphrased and the labels stay the same.

`eval_paraphrase.py` trains the upstream BiLSTM-with-attention script, paraphrases the 200 held-out posts with `Qwen/Qwen2.5-0.5B-Instruct` (greedy decoding), and writes:

- `outputs/paraphrases.jsonl`, original text and paraphrase side by side, with the original label
- `results.json`, precision, recall, and F1 on the original 200 and the paraphrased 200

Bot is the positive class, matching the upstream script (`[0, 1]` for bot).

On this run, using the DartPost holdout described below and not TwiBot-20, the same checkpoint scored the original 200 posts at precision 0.739, recall 0.293, and F1 0.420 (17 true positives, 6 false positives, 41 false negatives). After paraphrasing, those figures were precision 0.511, recall 0.397, and F1 0.447 (23 true positives, 22 false positives, 35 false negatives). Precision fell by 0.228, recall rose by 0.103, and F1 rose by 0.027. None of the 200 paraphrases was unchanged from the readable source text. Exact values are in `results.json`.

## TwiBot-20 files in this checkout

`TwiBot-20_sample.json` is only a sample. It has no `label` field. The upstream README says the full dataset is withheld for privacy and requires permission from shangbin@cs.washington.edu. This fork does not use that sample as training data, does not invent an official split, and does not report TwiBot-20 scores.

## Data that was actually used

[Zenodo 10.5281/zenodo.14795885](https://doi.org/10.5281/zenodo.14795885) is the DartPost release for Saurabh Kumar, Valerio La Gatta, Andrea Pugliese, Andrew Pulver, V.S. Subrahmanian, Jiazhi Zhang, and Youzhi Zhang, “Reinforcement-Learning Based Covert Social Influence Operations,” The Web Conference 2025, [doi:10.1145/3696410.3714729](https://doi.org/10.1145/3696410.3714729). The deposit is MIT-licensed and includes post text (`Posts.csv`) plus account roles (`Users.csv`). It is not a survey-totals file.

Labels used here:

- bot: `isPuppet` true (80 sockpuppet accounts; 505 posts)
- human: `isPuppet` false and `isObserver` false (225 participants; 1,239 posts)
- excluded: `isObserver` true accounts, which have no posts

There is no official train/test split in the deposit. The 200 evaluation posts are a stratified sample (seed 100) drawn in proportion to those post counts. The remaining posts are the training set. The user table also contains emails and password hashes. Those fields are not copied into `outputs/` or `results.json`. Nothing is scraped from Twitter. The covert-influence agent from that paper is not reimplemented.

## Limits

- Text only. One existing baseline. No graph model and no new detector.
- Two hundred paraphrases, not the full test pool and not the official TwiBot-20 split.
- The checked-in Wei script freezes GloVe Twitter 25-dimensional vectors, uses a 50,000-word vocabulary, hidden size 16, 3 LSTM layers, and truncates each text to 64 tokens. The upstream readme mentions different sizes (embedding 200, hidden 64, vocabulary 5,000). This fork runs the script.
- Upstream sampling of a random LSTM initial state on every forward pass is replaced with zeros, so the original-versus-paraphrase gap is not that noise. The architecture and loss are unchanged.
- The upstream trainer still keeps the checkpoint with the highest accuracy on the original held-out loader.
- Embedded `data:image` blobs are removed only from the paraphrase prompt. The saved original text is the post as released. If a decode is empty, emoji and other symbols are stripped and the model is called again.
- `Qwen2.5-0.5B-Instruct` is small. Some paraphrases shift details (for example, “years” to “decades”) instead of staying inside a similarity threshold.
- Scores in `results.json` come from this run. They are not a claim about TwiBot-20.

## Earlier paraphrase and bot-detection work

Guo and Zhou, “Text–relation joint modeling improves the robustness of Chinese social bot detection against paraphrasing attacks,” Scientific Reports, 1 October 2026, [doi:10.1038/s41598-026-68571-x](https://doi.org/10.1038/s41598-026-68571-x). They paraphrase posts at three budgets and score account-level bot detectors. On their Weibo data, a text-only MacBERT + MLP baseline falls from 0.867 to 0.733 macro-F1 under the default sentence-level paraphrase (a 13.4-point drop). Their relation-aware model falls from 0.891 to 0.835 (5.6 points). They also report smaller Tier-2 drops on TwiBot-22: 9.6 points for MacBERT + MLP, 6.8 for BotMoE, and 4.2 for their model. This fork does not train those models.

Gaurang Katyal, “Account-history features for social bot detection in the era of large language models,” [arXiv:2606.26127](https://arxiv.org/abs/2606.26127). Rewriting bot posts so that URL, hashtag, mention, and casing rates match human posts drops a content classifier from ROC-AUC 0.842 to 0.785. Account-history features stay essentially flat. That rewrite matches surface statistics. It is not a semantic paraphrase of 200 held-out texts, and the TwiBot-20 check in that preprint is a 100-account score-distribution note on the public sample.

Krishna et al., “Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense” (NeurIPS 2023), is about AI-text detectors such as watermarking and DetectGPT, not social-bot account labels. Guo and Zhou cite it as the reason paraphrasing is a practical evasion tool. It is not a bot-detection result.

## License

Upstream TwiBot-20 and TwiBot-22 are MIT. Files added in this fork, including `eval_paraphrase.py` and the small path changes under `baseline/wei/`, are MIT as well. See `LICENSE`.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# glove.twitter.27B.25d.txt must be at data/glove/glove.twitter.27B.25d.txt
# unzip 10.5281/zenodo.14795885 under data/zenodo/
.venv/bin/python eval_paraphrase.py
```
