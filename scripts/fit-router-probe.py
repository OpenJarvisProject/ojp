#!/usr/bin/env python3
"""Fit the router probe: a linear classifier on the router model's hidden state
at the answer position (ojp-cli route-feats ... short).

  fit-router-probe.py feats-train.jsonl feats-xl.jsonl out.json

Chooses the L2 strength by 5-fold cross-validation on the training features,
reports accuracy on the held-out XL features with a train-only fit, then
writes a probe fitted on train + XL. The probe is specific to the exact GGUF
the features came from."""
import json, sys
import numpy as np

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=np.float64)

def fit(X, y, l2, iters=500, lr=0.5):
    n, d = X.shape
    W = np.zeros((d, 4)); b = np.zeros(4); Y = np.eye(4)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W); b -= lr * G.sum(0)
    return W, b

def acc(X, y, W, b): return float(((X @ W + b).argmax(1) == y).mean())

ytr, Xtr = load(sys.argv[1]); yxl, Xxl = load(sys.argv[2])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
A, B = (Xtr - mu) / sd, (Xxl - mu) / sd
rng = np.random.default_rng(0); idx = rng.permutation(len(ytr)); folds = np.array_split(idx, 5)
best = None
for l2 in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
    cv = []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        W, b = fit(A[tr], ytr[tr], l2); cv.append(acc(A[te], ytr[te], W, b))
    m = float(np.mean(cv))
    if best is None or m > best[0] + 1e-9: best = (m, l2)
cv_acc, l2 = best
W, b = fit(A, ytr, l2)
print("train n=%d  5-fold CV %.3f (l2=%g)  held-out XL (n=%d) %.3f" % (len(ytr), cv_acc, l2, len(yxl), acc(B, yxl, W, b)))
Xall = np.concatenate([Xtr, Xxl]); yall = np.concatenate([ytr, yxl])
mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
W, b = fit((Xall - mu) / sd, yall, l2)
# Fold the standardisation into the weights: logits = x @ W2 + b2.
W2 = W / sd[:, None]; b2 = b - (mu / sd) @ W
json.dump({"version": 1, "labels": ["answer_directly", "coding", "reasoning", "device_control"],
           "dim": int(W2.shape[0]), "weights": W2.T.round(6).tolist(), "bias": b2.round(6).tolist(),
           "cv_accuracy": round(cv_acc, 4)}, open(sys.argv[3], "w"))
print("wrote", sys.argv[3])
