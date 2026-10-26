#!/usr/bin/env python3
"""ojp-core/data/router-train-aug.tsv: the training requests plus noisy copies (typos,
lower case, dropped punctuation, chat shorthand), so a probe sees how people actually type."""
import random
from pathlib import Path
random.seed(5)
root = Path(__file__).resolve().parent.parent
rows = [l.rstrip("\n").split("\t", 1) for l in (root / "ojp-core/data/router-train.tsv").open()]
short = {"please": "pls", "thanks": "thx", "you": "u", "your": "ur", "what is": "whats", "what's": "whats", "how do I": "how2", "can you": "can u", "and": "n", "to": "2", "for": "4"}
def typo(w):
    if len(w) < 4: return w
    i = random.randrange(1, len(w) - 1)
    kind = random.choice(["swap", "drop", "dup"])
    if kind == "swap": return w[:i] + w[i + 1] + w[i] + w[i + 2:] if i + 2 <= len(w) else w
    if kind == "drop": return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]
def noisy(t):
    if random.random() < 0.5: t = t.lower()
    if random.random() < 0.4: t = t.rstrip(".?!")
    for a, b in short.items():
        if random.random() < 0.3: t = t.replace(a, b)
    ws = t.split(" ")
    for k in range(len(ws)):
        if random.random() < 0.08: ws[k] = typo(ws[k])
    return " ".join(ws)
out = list(rows)
seen = {t.lower() for _, t in rows}
for l, t in rows:
    for _ in range(2):
        n = noisy(t)
        if n.lower() not in seen:
            seen.add(n.lower()); out.append([l, n])
random.shuffle(out)
(root / "ojp-core/data/router-train-aug.tsv").write_text("".join(f"{l}\t{t}\n" for l, t in out))
print(len(out), "rows")
