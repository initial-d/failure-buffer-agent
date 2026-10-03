"""Analysis of the ground-truth patch task: accuracy, two-stage repair, randomised probability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from fba.metrics.leakage import summarize_diff  # noqa: E402

NAME = {"gpt-4o-mini": "GPT-4o-mini", "qwen3.5-27b": "Qwen3.5-27B", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}
SG = ["none", "snapshot", "canary", "killswitch"]
LOSS = {"none": 4.0, "snapshot": 2.0, "canary": 1.0, "killswitch": 0.25}
ST = ROOT / "results/statistics/groundtruth"
ST.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)


def load():
    conds = []
    for f in ("stage1", "main", "stage2"):
        p = ROOT / "data/generated/groundtruth" / f"{f}.jsonl"
        if p.exists():
            conds += [json.loads(l) for l in open(p)]
    meta = pd.DataFrame(conds).drop(columns=["system", "prompt", "prompt_hash"])
    rows = []
    for f in sorted((ROOT / "results/raw/groundtruth").glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                rows.append({"condition_id": r["condition_id"], "model": f.stem, **r["parsed_response"]})
    df = pd.DataFrame(rows).merge(meta.drop(columns=[c for c in ["model"] if c in meta.columns]),
                                  on="condition_id")
    df["truth"] = df.truth.astype(float)
    if "deploy" in df:
        df["deploy"] = df.deploy.astype(float)
    return df


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    df = load()
    out = {}
    print(df.groupby(["model", "arm"]).size().unstack().to_string())

    # ---------------------------------------------------------------- 1. belief and accuracy
    print("\n=== 1. Stated probability and accuracy by safeguard (vanilla arm) ===")
    rows = []
    s1 = df[df.arm == "belief_nobuf"]
    for m in NAME:
        b0 = s1[s1.model == m]
        rows.append({"model": NAME[m], "condition": "no safeguard sentence", "mean_p": b0.success_probability.mean(),
                     "brier": brier(b0.success_probability, b0.truth), "logloss": logloss(b0.success_probability, b0.truth),
                     "brier_on_failures": brier(b0[b0.truth == 0].success_probability, 0)})
        for arm in ("vanilla", "instruction"):
            for sg in SG:
                g = df[(df.model == m) & (df.arm == arm) & (df.safeguard == sg) & df.success_probability.notna()]
                rows.append({"model": NAME[m], "condition": f"{arm}/{sg}", "mean_p": g.success_probability.mean(),
                             "brier": brier(g.success_probability, g.truth),
                             "logloss": logloss(g.success_probability, g.truth),
                             "brier_on_failures": brier(g[g.truth == 0].success_probability, 0)})
    acc = pd.DataFrame(rows)
    acc.to_csv(ST / "accuracy.csv", index=False)
    print(acc.to_string(index=False))

    print("\n=== paired contrasts kill switch - no rollback (per item) ===")
    rows = []
    for m in NAME:
        for arm in ("vanilla", "instruction"):
            g = df[(df.model == m) & (df.arm == arm) & df.success_probability.notna()]
            w = g.pivot_table(index="instance_id", columns="safeguard", values="success_probability")
            t = g.groupby("instance_id").truth.first()
            for name, d in (("p", w["killswitch"] - w["none"]),
                            ("brier", (w["killswitch"] - t) ** 2 - (w["none"] - t) ** 2),
                            ("brier_failed_items", ((w["killswitch"] - t) ** 2 - (w["none"] - t) ** 2)[t == 0]),
                            ("p_failed_items", (w["killswitch"] - w["none"])[t == 0]),
                            ("p_correct_items", (w["killswitch"] - w["none"])[t == 1])):
                rows.append({"model": NAME[m], "arm": arm, "measure": name, **summarize_diff(d, "")})
    con = pd.DataFrame(rows)
    con.to_csv(ST / "belief_contrasts.csv", index=False)
    print(con[["model", "arm", "measure", "n", "mean", "ci_low", "ci_high", "p_perm"]].to_string(index=False))

    # ---------------------------------------------------------------- 2. two-stage repair
    print("\n=== 2. Decisions and realised utility by arm and safeguard ===")
    dec = df[df.deploy.notna() & df.arm.isin(["vanilla", "instruction", "twostage"])].copy()
    dec["utility"] = np.where(dec.deploy == 1, np.where(dec.truth == 1, 1.0, -dec.loss), 0.0)
    rows = []
    for (m, arm, sg), g in dec.groupby(["model", "arm", "safeguard"]):
        rows.append({"model": NAME[m], "arm": arm, "safeguard": sg, "deploy_rate": g.deploy.mean(),
                     "deploy_rate_failed": g[g.truth == 0].deploy.mean(),
                     "deploy_rate_correct": g[g.truth == 1].deploy.mean(), "utility": g.utility.mean(), "n": len(g)})
    # benchmark policies with the true outcome unknown: always/never deploy, and a calibrated-threshold oracle
    for m in NAME:
        s = s1[s1.model == m].set_index("instance_id")
        for sg in SG:
            y = s.truth
            thr = LOSS[sg] / (1 + LOSS[sg])
            base = y.mean()
            for pol, d in (("never", np.zeros(len(y))), ("always", np.ones(len(y))),
                           ("threshold on stage-1 belief", (s.success_probability > thr).astype(float))):
                u = np.where(d == 1, np.where(y == 1, 1.0, -LOSS[sg]), 0.0)
                rows.append({"model": NAME[m], "arm": f"policy: {pol}", "safeguard": sg, "deploy_rate": d.mean(),
                             "deploy_rate_failed": d[y.values == 0].mean(), "deploy_rate_correct": d[y.values == 1].mean(),
                             "utility": u.mean(), "n": len(y)})
    util = pd.DataFrame(rows)
    util["safeguard"] = pd.Categorical(util.safeguard, SG, ordered=True)
    util.to_csv(ST / "utility.csv", index=False)
    for col in ("deploy_rate", "deploy_rate_failed", "utility"):
        print(f"\n--- {col}")
        print(util.pivot_table(index=["model", "arm"], columns="safeguard", values=col, observed=True).to_string())

    print("\n=== paired: utility(twostage) - utility(vanilla), and deploy of failed patches ===")
    rows = []
    for m in NAME:
        g = dec[dec.model == m]
        for sg in SG:
            w = g[g.safeguard == sg].pivot_table(index="instance_id", columns="arm", values="utility")
            wd = g[(g.safeguard == sg) & (g.truth == 0)].pivot_table(index="instance_id", columns="arm", values="deploy")
            for other in ("vanilla", "instruction"):
                rows.append({"model": NAME[m], "safeguard": sg, "contrast": f"twostage-{other}", "measure": "utility",
                             **summarize_diff(w["twostage"] - w[other], "")})
                rows.append({"model": NAME[m], "safeguard": sg, "contrast": f"twostage-{other}",
                             "measure": "deploy_failed", **summarize_diff(wd["twostage"] - wd[other], "")})
    rep = pd.DataFrame(rows)
    rep.to_csv(ST / "repair.csv", index=False)
    print(rep[["model", "safeguard", "contrast", "measure", "n", "mean", "ci_low", "ci_high", "p_perm"]].to_string(index=False))

    # does the two-stage arm keep the legitimate action shift?
    print("\n=== legitimate shift: deploy(killswitch) - deploy(none) within each arm ===")
    rows = []
    for m in NAME:
        for arm in ("vanilla", "instruction", "twostage"):
            w = dec[(dec.model == m) & (dec.arm == arm)].pivot_table(index="instance_id", columns="safeguard", values="deploy")
            rows.append({"model": NAME[m], "arm": arm, **summarize_diff(w["killswitch"] - w["none"], "")})
    sh = pd.DataFrame(rows)
    sh.to_csv(ST / "action_shift.csv", index=False)
    print(sh[["model", "arm", "n", "mean", "ci_low", "ci_high", "p_perm"]].to_string(index=False))

    # ---------------------------------------------------------------- 3. randomised probability
    print("\n=== 3. Randomised probability: deploy rate by assigned p and safeguard ===")
    gv = df[df.arm.str.startswith("given") & df.deploy.notna()].copy()
    print(gv.pivot_table(index=["model", "given_p"], columns="safeguard", values="deploy").reindex(columns=SG).to_string())
    rows = []
    for (m, sg), g in gv.groupby(["model", "safeguard"]):
        X = np.c_[np.ones(len(g)), g.given_p]
        beta = np.linalg.lstsq(X, g.deploy.values, rcond=None)[0][1]
        bs = []
        ids = g.instance_id.unique()
        by = {i: x for i, x in g.groupby("instance_id")}
        rng = np.random.default_rng(1)
        for _ in range(1000):
            s = pd.concat([by[i] for i in rng.choice(ids, len(ids))])
            bs.append(np.linalg.lstsq(np.c_[np.ones(len(s)), s.given_p], s.deploy.values, rcond=None)[0][1])
        rows.append({"model": NAME[m], "safeguard": sg, "slope": beta, "lo": np.quantile(bs, .025),
                     "hi": np.quantile(bs, .975)})
    slopes = pd.DataFrame(rows)
    slopes.to_csv(ST / "causal_slopes.csv", index=False)
    print("\n--- causal slope of deploy on the assigned probability (per unit p)")
    print(slopes.to_string(index=False))

    # implied belief-mediated action shift = slope at kill switch x belief inflation under kill switch
    print("\n=== implied action shift from the belief inflation (randomised slope x belief shift) ===")
    rows = []
    for m in NAME:
        sl = slopes[(slopes.model == NAME[m]) & (slopes.safeguard == "killswitch")].iloc[0]
        bshift = con[(con.model == NAME[m]) & (con.arm == "vanilla") & (con.measure == "p")].iloc[0]
        total = sh[(sh.model == NAME[m]) & (sh.arm == "vanilla")].iloc[0]
        two = sh[(sh.model == NAME[m]) & (sh.arm == "twostage")].iloc[0]
        rows.append({"model": NAME[m], "belief_shift": bshift["mean"], "slope_kill": sl.slope,
                     "implied_mediated": sl.slope * bshift["mean"],
                     "implied_lo": sl.lo * bshift["ci_low"], "implied_hi": sl.hi * bshift["ci_high"],
                     "total_shift_vanilla": total["mean"], "shift_twostage": two["mean"],
                     "observed_gap": total["mean"] - two["mean"]})
    imp = pd.DataFrame(rows)
    imp.to_csv(ST / "implied_mediation.csv", index=False)
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
