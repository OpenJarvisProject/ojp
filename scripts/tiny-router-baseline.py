#!/usr/bin/env python3
"""How good is a router that is not a language model at all? Hashed word and character n-grams + softmax
regression (a few hundred KB, microseconds per request), trained on the same data as the probes, scored on the
same held-out sets. Usage: tiny-router-baseline.py (run from the ojp repo)."""
import re, zlib
from pathlib import Path
import numpy as np
D = Path("ojp-core/data"); DIM = 1 << 14
def feats(t):
    t = t.lower(); w = re.findall(r"[a-z0-9']+|[^\sa-z0-9]", t); f = list(w) + [a + " " + b for a, b in zip(w, w[1:])]
    s = " " + re.sub(r"\s+", " ", t) + " "
    f += [s[i:i + n] for n in (3, 4, 5) for i in range(len(s) - n + 1)]
    return f
def vec(rows):
    X = np.zeros((len(rows), DIM), dtype=np.float32)
    for i, (_, t) in enumerate(rows):
        for g in feats(t): X[i, zlib.crc32(g.encode()) % DIM] += 1
    X = np.log1p(X); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); return X
def load(name): return [tuple(l.rstrip("\n").split("\t", 1)) for l in (D / name).open() if "\t" in l]
def fit(X, y, K, l2=1e-4, iters=300, lr=2.0):
    W = np.zeros((X.shape[1], K), dtype=np.float32); b = np.zeros(K, dtype=np.float32); Y = np.eye(K, dtype=np.float32)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(y); W -= lr * (X.T @ G + l2 * W); b -= lr * G.sum(0)
    return W, b
def run(title, train, tests):
    labels = sorted({l for l, _ in train}); idx = {l: i for i, l in enumerate(labels)}
    X = vec(train); y = np.array([idx[l] for l, _ in train]); W, b = fit(X, y, len(labels))
    out = [f"{title}: train {len(train)}, {len(labels)} classes, model {W.nbytes // 1024} KB (float32)"]
    for name, rows in tests:
        rows = [(l, t) for l, t in rows if l in idx]
        p = (vec(rows) @ W + b).argmax(1); acc = np.mean([labels[i] == l for i, (l, _) in zip(p, rows)])
        out.append(f"  {name}: {acc:.1%} of {len(rows)}")
    print("\n".join(out))
run("router (4 routes)", load("router-train-aug.tsv"), [("held-out 80", load("router-xl.tsv")), ("wild 42", load("router-wild.tsv")), ("wild 44", load("router-wild2.tsv"))])
run("tool (30 classes)", load("tool-train.tsv"), [("held-out 73 (tool-eval phrasings)", load("tool-test.tsv"))])
