import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fba.metrics.leakage import add_cond_key, bbl_table, control_table, wide_main  # noqa: E402

MODELS = None
BUFFER_CODE = {"low": -1, "medium": 0, "high": 1}


def load(cfg) -> pd.DataFrame:
    conds = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/generated" / cfg["study"] / "conditions.jsonl")])
    conds = conds.drop(columns=["slots", "system", "prompt"])
    frames = []
    for f in sorted((ROOT / "results/raw" / cfg["study"]).glob("*.jsonl")):
        if MODELS and f.stem not in MODELS:
            continue
        recs = [json.loads(l) for l in open(f)]
        ok = {}
        for r in recs:
            if r["status"] == "ok":
                ok[r["condition_id"]] = r
        rows = [{"condition_id": cid, "model": f.stem,
                 "p": r["parsed_response"]["success_probability"],
                 "lo": r["parsed_response"]["lower_bound"], "hi": r["parsed_response"]["upper_bound"],
                 "reasoning_tokens": (r["token_usage"].get("completion_tokens_details") or {}).get("reasoning_tokens")}
                for cid, r in ok.items()]
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames).merge(conds, on="condition_id", how="left")
    df["mid"] = (df.lo + df.hi) / 2
    df["width"] = df.hi - df.lo
    return add_cond_key(df)


def mixed_model(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df.condition_type == "main"].copy()
    main["B"] = main.buffer_level.map(BUFFER_CODE)
    main["E"] = main.explicit_independence.astype(int)
    eps = 1e-3
    main["logit_p"] = np.log(main.p.clip(eps, 1 - eps) / (1 - main.p.clip(eps, 1 - eps)))
    rows = []
    for model, g in main.groupby("model"):
        for dv in ["p", "logit_p"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(f"{dv} ~ B * C(ambiguity, Treatment('precise')) + B:E + E + C(domain)",
                                  g, groups=g["instance_id"]).fit(reml=True)
            for term in fit.params.index:
                if term.startswith("B") or term == "E":
                    rows.append({"model": model, "dv": dv, "term": term, "coef": fit.params[term],
                                 "se": fit.bse[term], "p": fit.pvalues[term]})
    return pd.DataFrame(rows)


def plots(df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    w = wide_main(df, "p")
    models = sorted(w.model.unique())
    ambs = ["precise", "sparse", "qualitative"]
    fig, axes = plt.subplots(len(models), 3, figsize=(12, 3.6 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, a in enumerate(ambs):
            ax = axes[i, j]
            g = w[(w.model == m) & (w.ambiguity == a) & (w.indep == "imp")]
            xs = np.array([0, 1, 2])
            for _, r in g.iterrows():
                ax.plot(xs, r[["low", "medium", "high"]].values, color="0.6", alpha=0.35, lw=0.8)
            ax.plot(xs, g[["low", "medium", "high"]].mean().values, color="C3", lw=2.5, marker="o")
            ax.set_xticks(xs, ["low", "med", "high"])
            ax.set_title(f"{m} | {a} (implicit)", fontsize=10)
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel("elicited P(success)")
    fig.tight_layout()
    fig.savefig(out / "study1_paired_plot.png", dpi=150)
    plt.close(fig)

    # Forest: H-L by model x domain x ambiguity (implicit + explicit)
    w["d"] = w["high"] - w["low"]
    agg = w.groupby(["model", "ambiguity", "indep", "domain"]).d.agg(["mean", "sem", "count"]).reset_index()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 7), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0, i]
        g = agg[agg.model == m].sort_values(["ambiguity", "indep", "domain"]).reset_index(drop=True)
        y = np.arange(len(g))
        colors = [{"imp": "C0", "exp": "C1", "exog": "C2"}[e] for e in g.indep]
        ax.errorbar(g["mean"], y, xerr=1.96 * g["sem"].fillna(0), fmt="none", ecolor=colors)
        ax.scatter(g["mean"], y, c=colors, zorder=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, [f"{a[:4]}|{e}|{d}" for a, e, d in
                          zip(g.ambiguity, g.indep, g.domain)], fontsize=7)
        ax.set_xlabel("BBL = p(high) - p(low)")
        ax.set_title(m)
    fig.tight_layout()
    fig.savefig(out / "study1_forest.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/study1.yaml"))
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    global MODELS
    MODELS = args.models
    cfg = yaml.safe_load(open(args.config))
    df = load(cfg)
    stat_dir = ROOT / "results/statistics" / cfg["study"]
    stat_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/processed" / f"{cfg['study']}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
    print("responses per model:\n", df.groupby("model").size(), "\n")

    bbl = bbl_table(df)
    bbl.to_csv(stat_dir / "bbl.csv", index=False)
    hl = bbl[(bbl.value == "p") & (bbl.metric == "H-L")].copy()
    hl["cell"] = hl.apply(lambda r: f"{r['mean']:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}]{'*' if r.p_perm < .05 else ' '}", axis=1)
    print("=== BBL: p(high) - p(low), mean [95% CI], * p_perm<.05 ===")
    print(hl.pivot(index="model", columns=["ambiguity", "indep"], values="cell").to_string())
    for metric in ["H-none", "L-none"]:
        t = bbl[(bbl.value == "p") & (bbl.metric == metric) & (bbl.indep == "imp")]
        print(f"\n=== {metric} (implicit) ===")
        print(t.pivot(index="model", columns="ambiguity", values="mean").to_string())
    cw = bbl[(bbl.value == "width") & (bbl.ambiguity == "qualitative")]
    print("\n=== Credal width H-L (qualitative) ===")
    print(cw.pivot(index="model", columns="indep", values="mean").to_string())

    bbl_dom = bbl_table(df, by=("model", "domain", "indep"))
    bbl_dom.to_csv(stat_dir / "bbl_by_domain.csv", index=False)
    d = bbl_dom[(bbl_dom.value == "p") & (bbl_dom.metric == "H-L") & (bbl_dom.indep == "imp")]
    print("\n=== BBL by domain (implicit, pooled over ambiguity) ===")
    print(d.pivot(index="model", columns="domain", values="mean").to_string())

    ctrl = pd.concat([control_table(df, a).assign(ambiguity=a) for a in cfg["control_ambiguity"]])
    ctrl.to_csv(stat_dir / "controls.csv", index=False)
    print("\n=== Controls (mean paired difference) ===")
    print(ctrl.pivot(index=["ambiguity", "model"], columns="metric", values="mean").to_string())

    mm = mixed_model(df)
    mm.to_csv(stat_dir / "mixed_effects.csv", index=False)

    plots(df, ROOT / "results/figures" / cfg["study"])
    print(f"\nfigures -> results/figures/{cfg['study']}")


if __name__ == "__main__":
    main()
