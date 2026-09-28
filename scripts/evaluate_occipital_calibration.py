import csv
import math
import os
import random


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results", "cv5_real8_post_registration_gaussian", "oof_predictions.csv")
FEAT = os.path.join(ROOT, "results", "scan_clusters_10", "feature_table.csv")
OUT = os.path.join(ROOT, "results", "occipital_calibration")
EPS = 1e-6


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def solve(a, b):
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[col][col] += 1e-6
            pivot = col
        a[col], a[pivot] = a[pivot], a[col]
        q = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= q
        for i in range(n):
            if i == col:
                continue
            q = a[i][col]
            if q:
                for j in range(col, n + 1):
                    a[i][j] -= q * a[col][j]
    return [a[i][n] for i in range(n)]


def fit_logistic(rows, feature_fn, ridge=1e-3):
    p = len(feature_fn(rows[0]))
    beta = [0.0] * p
    for _ in range(80):
        h = [[0.0] * p for _ in range(p)]
        g = [0.0] * p
        for r in rows:
            x = feature_fn(r)
            q = sigmoid(dot(beta, x))
            w = max(q * (1.0 - q), 1e-8)
            for i in range(p):
                g[i] += x[i] * (r["y"] - q)
                for j in range(p):
                    h[i][j] += w * x[i] * x[j]
        for i in range(1, p):
            h[i][i] += ridge
            g[i] -= ridge * beta[i]
        step = solve(h, g)
        maxstep = max(abs(v) for v in step)
        beta = [u + v for u, v in zip(beta, step)]
        if maxstep < 1e-7:
            break
    return beta


def fit_scale(rows):
    # One-parameter temperature scaling, p=sigmoid(logit(p)/T).
    # A bounded grid is robust for the small calibration set.
    best_t, best_loss = 1.0, float("inf")
    for i in range(601):
        log_t = -3.0 + 6.0 * i / 600.0
        t = math.exp(log_t)
        loss = 0.0
        for r in rows:
            q = min(max(sigmoid(r["z"] / t), 0.005), 0.995)
            loss += -(r["y"] * math.log(q) + (1 - r["y"]) * math.log(1 - q))
        if loss < best_loss:
            best_loss, best_t = loss, t
    return best_t


def nll(y, p):
    pp = [min(max(v, 0.005), 0.995) for v in p]
    return -sum(yy * math.log(v) + (1 - yy) * math.log(1 - v) for yy, v in zip(y, pp)) / len(y)


def auc(y, p):
    pairs = sorted(zip(p, y))
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j + 2) / 2.0
        rank_sum += rank * sum(pairs[k][1] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def standardizer(train):
    vals = [r["q"] for r in train]
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) or 1.0
    return lambda r: (r["q"] - m) / sd


def fit_method(name, train):
    zq = standardizer(train)
    if name == "platt":
        beta = fit_logistic(train, lambda r: [1.0, r["z"]])
        return lambda r: sigmoid(beta[0] + beta[1] * r["z"])
    if name == "temperature":
        t = fit_scale(train)
        return lambda r: sigmoid(r["z"] / t)
    if name == "occipital_platt":
        beta = fit_logistic(train, lambda r: [1.0, r["z"], zq(r), r["z"] * zq(r)])
        return lambda r: sigmoid(beta[0] + beta[1] * r["z"] + beta[2] * zq(r) + beta[3] * r["z"] * zq(r))
    if name == "quality_gated":
        vals = sorted(r["q"] for r in train)
        cuts = [vals[len(vals) // 3], vals[2 * len(vals) // 3]]
        groups = []
        for lo, hi in [(-float("inf"), cuts[0]), (cuts[0], cuts[1]), (cuts[1], float("inf"))]:
            g = [r for r in train if lo <= r["q"] <= hi]
            beta = fit_logistic(g, lambda r: [1.0, r["z"]])
            groups.append((lo, hi, beta))
        def pred(r):
            for lo, hi, beta in groups:
                if lo <= r["q"] <= hi:
                    return sigmoid(beta[0] + beta[1] * r["z"])
            return sigmoid(r["z"])
        return pred
    raise ValueError(name)


def main():
    os.makedirs(OUT, exist_ok=True)
    feats = {}
    with open(FEAT, newline="") as f:
        for r in csv.DictReader(f):
            feats[r["uid"]] = r
    rows = []
    with open(PRED, newline="") as f:
        for r in csv.DictReader(f):
            uid = r["uid"]
            q = math.log(max(float(feats[uid]["occipital_mean"]), 1e-9))
            p = float(r["equal_ensemble"])
            rows.append({"uid": uid, "fold": int(r["fold"]), "y": int(r["actual"]), "p": p, "z": logit(p), "q": q})

    methods = ["platt", "temperature", "occipital_platt", "quality_gated"]
    # Outer OOF evaluation. Within each outer training portion, choose the best
    # method by a deterministic 4-way inner split, then refit it on all outer train rows.
    final = {m: [] for m in methods}
    chosen = []
    for outer in sorted(set(r["fold"] for r in rows)):
        train = [r for r in rows if r["fold"] != outer]
        test = [r for r in rows if r["fold"] == outer]
        inner_scores = {}
        for method in methods:
            scores = []
            for inner in sorted(set(r["fold"] for r in train)):
                a = [r for r in train if r["fold"] != inner]
                b = [r for r in train if r["fold"] == inner]
                pred = fit_method(method, a)
                scores.append(nll([r["y"] for r in b], [pred(r) for r in b]))
            inner_scores[method] = sum(scores) / len(scores)
        best = min(methods, key=lambda m: inner_scores[m])
        chosen.append((outer, best, inner_scores[best]))
        for method in methods:
            pred = fit_method(method, train)
            for r in test:
                final[method].append((r, pred(r)))

    summary = []
    yall = [r["y"] for r in rows]
    pbase = [r["p"] for r in rows]
    summary.append(("original_equal_ensemble", nll(yall, pbase), auc(yall, pbase)))
    for m in methods:
        rr = sorted(final[m], key=lambda x: x[0]["uid"])
        yy = [r["y"] for r, _ in rr]
        pp = [p for _, p in rr]
        summary.append((m, nll(yy, pp), auc(yy, pp)))

    with open(os.path.join(OUT, "nested_oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["uid", "actual", "fold"] + methods)
        byuid = {r["uid"]: r for r in rows}
        predmaps = [{r["uid"]: p for r, p in final[m]} for m in methods]
        for uid in sorted(byuid):
            r = byuid[uid]
            w.writerow([uid, r["y"], r["fold"]] + [pm[uid] for pm in predmaps])
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "log_loss", "auroc"])
        w.writerows(summary)
    with open(os.path.join(OUT, "outer_selected_methods.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["outer_fold", "selected_method", "inner_log_loss"]); w.writerows(chosen)
    print("Nested calibration complete")
    for m, ll, aa in summary:
        print(f"{m}: log_loss={ll:.6f}, AUROC={aa:.6f}")
    print("outer selections:", ", ".join(f"fold {f}: {m}" for f, m, _ in chosen))


if __name__ == "__main__":
    main()
