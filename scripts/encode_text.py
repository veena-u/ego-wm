"""Encode every episode description with Wan's umT5 text encoder (run once, on the GPU).

Writes {"texts": [str], "emb": [Tensor(L_i, 4096) fp16]} with each embedding trimmed to its
real token length; the dataset zero-pads to 512 exactly like the Wan pipeline.

python scripts/encode_text.py --out /data/text_emb.pt
python scripts/encode_text.py --out data/text_emb_smoke.pt --tiny-random --limit 20   # Mac test
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="+", default=["data/index.csv"], help="one or more index.csv files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny-random", action="store_true", help="random small encoder: tests the code path only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

    texts = set()
    for path in a.index:
        with open(path) as f:
            texts |= {r["description"] for r in csv.DictReader(f) if r["description"]}
    texts = sorted(texts)[: a.limit]
    print(f"{len(texts)} unique descriptions")
    tok = AutoTokenizer.from_pretrained(a.repo, subfolder="tokenizer")
    if a.tiny_random:
        enc = UMT5EncoderModel(UMT5Config(vocab_size=tok.vocab_size + 300, d_model=64, d_kv=16, d_ff=128,
                                          num_layers=2, num_heads=4))
    else:
        enc = UMT5EncoderModel.from_pretrained(a.repo, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    enc = enc.to(a.device).eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch_size):
            batch = [clean(t) for t in texts[i:i + a.batch_size]]
            ids = tok(batch, padding="max_length", max_length=a.max_len, truncation=True,
                      add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
            h = enc(ids.input_ids.to(a.device), ids.attention_mask.to(a.device)).last_hidden_state
            for row, n in zip(h, ids.attention_mask.sum(1).tolist()):
                embs.append(row[:n].half().cpu())
    torch.save({"texts": texts, "emb": embs, "repo": a.repo, "tiny_random": a.tiny_random}, a.out)
    lens = [len(e) for e in embs]
    print(f"saved {a.out}: dim {embs[0].shape[-1]}, tokens min/mean/max {min(lens)}/{sum(lens) / len(lens):.1f}/{max(lens)}")


if __name__ == "__main__":
    main()
