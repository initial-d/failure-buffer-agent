"""Two offline analyses.

(1) Programmatic threshold baseline. The separated system's decision rule is
    a(B) = 1[p_hat > p*(B)].  Our two-stage arm instead asks the model to decide.  Here we drive the
    decision from the same buffer-free belief with the programmatic rule, which separates epistemic error
    from decision-rule execution error.

(2) Equivalence tests. Claims of the form "the effect disappears" are backed by non-significance, which is
    not evidence of absence.  We test equivalence to zero with a margin delta, using the paired bootstrap
    interval of the effect.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fba.metrics.leakage import paired_bootstrap_ci  # noqa: E402
import analyze_gt as A  # noqa: E402

ST = ROOT / "results/statistics"
MARGIN = 0.02          # practical equivalence margin on a probability scale


def equivalence(d, margin=MARGIN, n_boot=5000):
    d = np.asarray(pd.Series(d).dropna(), float)
    if len(d) == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, equivalent=False)
    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot)
    return dict(n=len(d), mean=float(d.mean()), lo=lo, hi=hi, equivalent=bool(lo > -margin and hi < margin))


def threshold_baseline():
    df = A.load()
    p0 = df[df.arm == "belief_nobuf"].set_index(["model", "instance_id"]).success_probability
    recs = []
    for _, r in df[df.arm == "belief_nobuf"].iterrows():
        recs.append({"model": r.model, "instance_id": r.instance_id, "p0": r.success_probability,
                     "truth": float(r.truth)})
    base = pd.DataFrame(recs)
    # decisions the supplied-probability arm actually made, for the same p grid
    gv = df[df.arm.str.startswith("given") & df.deploy.notna()].copy()
    rows = []
    for m, g in base.groupby("model"):
        for sg, loss in A.LOSS.items():
            pstar = loss / (1 + loss)
            prog = (g.p0 > pstar).astype(float).values
            util = np.where(prog == 1, np.where(g.truth.values == 1, 1.0, -loss), 0.0)
            # the model's own two-stage decision, same items and safeguard
            two = df[(df.model == m) & (df.arm == "twostage") & (df.safeguard == sg)] \
                .set_index("instance_id").deploy.reindex(g.instance_id.values).astype(float)
            util_llm = np.where(two.values == 1, np.where(g.truth.values == 1, 1.0, -loss), 0.0)
            # always / never
            util_never = np.zeros(len(g))
            util_always = np.where(g.truth.values == 1, 1.0, -loss)
            rows.append({"model": m, "safeguard": sg, "loss": loss, "pstar": pstar, "n": len(g),
                         "deploy_programmatic": float(prog.mean()), "utility_programmatic": float(util.mean()),
                         "deploy_twostage_llm": float(np.nanmean(two.values)),
                         "utility_twostage_llm": float(np.nanmean(util_llm)),
                         "utility_never": float(util_never.mean()), "utility_always": float(util_always.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(ST / "groundtruth/programmatic.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print("=== programmatic threshold on the buffer-free belief vs the LLM decision stage ===")
    print(out.to_string(index=False))
    print("\n=== paired: utility(programmatic) - utility(LLM two-stage), by model and safeguard ===")
    for m, g in out.groupby("model"):
        d = g.utility_programmatic - g.utility_twostage_llm
        print(f"  {m:22s} mean {d.mean():+.3f} over {len(g)} safeguards")
    print("\n=== how often do the two disagree? ===")
    disag = []
    for m in base.model.unique():
        for sg in A.LOSS:
            pstar = A.LOSS[sg] / (1 + A.LOSS[sg])
            g = base[base.model == m]
            prog = (g.p0 > pstar).astype(float)
            two = df[(df.model == m) & (df.arm == "twostage") & (df.safeguard == sg)] \
                .set_index("instance_id").deploy.reindex(g.instance_id.values).astype(float)
            m_ = (prog.values != two.values)
            disag.append({"model": m, "safeguard": sg, "disagreement": float(np.nanmean(m_)),
                          "n": int(np.sum(~np.isnan(two.values)))})
    dd = pd.DataFrame(disag)
    dd.to_csv(ST / "groundtruth/rule_error.csv", index=False)
    print(dd.to_string(index=False))
    return out, dd


def equivalence_checks():
    print("\n\n=== equivalence to zero (margin %.2f) ===" % MARGIN)
    rows = []
    import analyze_new_studies as ans
    raw = ans.load_study("gradient")
    for (m, n), gg in raw.groupby(["model", "n_cases"]):
        w = gg.pivot_table(index="instance_id", columns="buffer_level", values="p")
        rows.append({"family": "gradient", "model": m, "cell": f"n={n}",
                     **equivalence(w["high"] - w["low"])})
    s1 = pd.read_csv(ST / "study1_pilot/bbl.csv")
    hl = s1[(s1.value == "p") & (s1.metric == "H-L")]
    for _, r in hl[hl.ambiguity.isin(["precise", "sparse"])].iterrows():
        rows.append({"family": "study1", "model": r.model, "cell": f"{r.ambiguity}/{r.indep}",
                     "n": int(r["n"]), "mean": r["mean"], "lo": r.ci_low, "hi": r.ci_high,
                     "equivalent": bool(r.ci_low > -MARGIN and r.ci_high < MARGIN)})
    c = pd.read_csv(ST / "confirmatory.csv")
    for _, r in c[c.ambiguity == "sparse"].iterrows():
        rows.append({"family": "confirmatory", "model": r.model, "cell": "sparse/exog",
                     "n": int(r["n"]), "mean": r["mean"], "lo": r.ci_low, "hi": r.ci_high,
                     "equivalent": bool(r.ci_low > -MARGIN and r.ci_high < MARGIN)})
    b = pd.read_csv(ST / "study3_pilot/bfs.csv")
    for _, r in b[(b.metric == "posterior H-L") & (b.feedback == "neg_8of10")].iterrows():
        rows.append({"family": "feedback", "model": r.model, "cell": f"{r.ambiguity}/8of10",
                     "n": int(r["n"]), "mean": r["mean"], "lo": r.ci_low, "hi": r.ci_high,
                     "equivalent": bool(r.ci_low > -MARGIN and r.ci_high < MARGIN)})
    t = pd.read_csv(ST / "budget_paired.csv")
    for _, r in t[t.contrast == "direct->b512"].iterrows():
        rows.append({"family": "budget", "model": r.model, "cell": f"{r.ambiguity}/direct->512",
                     "n": int(r["n"]), "mean": r["mean"], "lo": r.ci_low, "hi": r.ci_high,
                     "equivalent": bool(r.ci_low > -MARGIN and r.ci_high < MARGIN)})
    df = pd.DataFrame(rows)
    df.to_csv(ST / "equivalence.csv", index=False)
    print(df.to_string(index=False))
    print("\nshare equivalent within margin, by family:")
    print(df.groupby("family").equivalent.agg(["mean", "size"]).round(3).to_string())
    return df


if __name__ == "__main__":
    threshold_baseline()
    equivalence_checks()
