#!/usr/bin/env python3
"""Fit the tool probe: a linear classifier over tool names (plus "multi") on the
router model's hidden state at the router prompt.

  fit-tool-probe.py feats-train.jsonl feats-test.jsonl out.json

Rows are ["label", hidden]. Picks the L2 strength by 5-fold CV on the training
rows, reports held-out accuracy (train-only fit), then writes a probe fitted on
train + test. Output holds labels, weights [class][dim] and bias."""
import json, sys
import numpy as np

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r[0] for r in rows], np.array([r[1] for r in rows], dtype=np.float64)

def fit(X, y, K, l2, iters=500, lr=0.5):
    n, d = X.shape
    W = np.zeros((d, K)); b = np.zeros(K); Y = np.eye(K)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W); b -= lr * G.sum(0)
    return W, b

def acc(X, y, W, b): return float(((X @ W + b).argmax(1) == y).mean())

ltr, Xtr = load(sys.argv[1]); lte, Xte = load(sys.argv[2])
labels = sorted(set(ltr) | set(lte)); K = len(labels); idx = {l: i for i, l in enumerate(labels)}
ytr = np.array([idx[l] for l in ltr]); yte = np.array([idx[l] for l in lte])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
A, B = (Xtr - mu) / sd, (Xte - mu) / sd
rng = np.random.default_rng(0); perm = rng.permutation(len(ytr)); folds = np.array_split(perm, 5)
best = None
for l2 in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
    cv = []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        W, b = fit(A[tr], ytr[tr], K, l2); cv.append(acc(A[te], ytr[te], W, b))
    m = float(np.mean(cv))
    if best is None or m > best[0] + 1e-9: best = (m, l2)
cv_acc, l2 = best
W, b = fit(A, ytr, K, l2)
pred = (B @ W + b).argmax(1)
print("train n=%d classes=%d 5-fold CV %.3f (l2=%g)  held-out (n=%d) %.3f" % (len(ytr), K, cv_acc, l2, len(yte), float((pred == yte).mean())))
for i in np.where(pred != yte)[0][:12]:
    print("  miss: want %s got %s" % (labels[yte[i]], labels[pred[i]]))
if "--train-only" in sys.argv:  # honest evaluation of the shipped procedure on the held-out set
    Xall, yall = Xtr, ytr
else:
    Xall = np.concatenate([Xtr, Xte]); yall = np.concatenate([ytr, yte])
mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
W, b = fit((Xall - mu) / sd, yall, K, l2)
W2 = W / sd[:, None]; b2 = b - (mu / sd) @ W
json.dump({"version": 1, "labels": labels, "dim": int(W2.shape[0]), "weights": W2.T.round(6).tolist(), "bias": b2.round(6).tolist(), "cv_accuracy": round(cv_acc, 4)}, open(sys.argv[3], "w"))
print("wrote", sys.argv[3])
