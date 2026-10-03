"""Non-API mechanism checks: (1) signal standardised by prompt-level noise, (2) reasoning-trace audit.

Rationale. Raw p-unit effects cannot be compared across output formats, because writing a number and
choosing Yes/No are different measurements.  Both share the same prompt-level noise, so we standardise each
effect by the standard deviation of paired differences *within a fixed condition* (which is pure prompt
noise: nothing about the model's input differs between the two members of a noise pair).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_study1 as a1  # noqa: E402
from fba.metrics.leakage import add_cond_key, summarize_diff  # noqa: E402

TRACE_MODELS = {"glm-5.3": "GLM-5.3", "kimi-k3": "Kimi-K3",
                "deepseek-v4.1-flash-think": "DeepSeek-V4.1-Flash + thinking",
                "qwen3.5-27b-think": "Qwen3.5-27B + thinking"}

# Trace patterns. COVER = the trace reasons about what happens after a failure.
COVER = [
    r"\broll ?back", r"\bblue.?green", r"\bkill switch", r"\bcanary", r"\bcheckpoint", r"\bsnapshot",
    r"\bbackup", r"\bpoint.in.time", r"\binsurance", r"\binsured", r"\breimburse", r"\bcash reserve",
    r"\breserve", r"\binsolven", r"\bbankrupt", r"\bshut down", r"\bsurvive", r"\babsorb",
    r"\bcan be (?:restored|reverted|undone)", r"\bwithout (?:lasting|permanent) (?:effect|damage)",
    r"\bno customer", r"\bno data loss", r"\bno lasting", r"\bmitigat", r"\brecover",
]
# FRAME = the trace treats the buffer as changing the odds of the outcome itself (the error we study).
FRAME = [
    r"(?:likel|probab|chance|risk|odds)[^.]{0,60}\b(?:higher|lower|increase|decrease|reduce|improve|boost)",
    r"\b(?:higher|lower|increase|decrease|reduce|improve)s?[^.]{0,40}(?:likelihood|probability|chance|odds)",
    r"\bmore likely to succeed\b", r"\bless likely to fail\b",
    r"\b(?:given|because of|thanks to|with) the (?:kill switch|rollback|blue.?green|backup|insurance|reserve|snapshot)[^.]{0,40}(?:likel|probab|success|work)",
    r"\bcan afford to be more (?:optimistic|aggressive|confident)",
    r"\bsafety net[^.]{0,40}(?:higher|more likely|increase)",
    r"\bcomfortable being more (?:optimistic|confident)",
]
ACTION = [r"\bdeploy\b", r"\bshould (?:proceed|go ahead|run|deploy)\b", r"\bnot (?:a )?decision\b",
          r"\bwhether to (?:deploy|proceed|go)\b"]
HEDGE = [r"\bguess", r"\bno(?:t)? (?:statistics|data|numbers)", r"\buncertain", r"\brange", r"\bwide interval"]


def counts(text, pats):
    if not text:
        return 0
    t = text.lower()
    return sum(1 for p in pats if re.search(p, t))


def standardised():
    """Effect in units of prompt-level noise, per model x ambiguity x format."""
    cfg = yaml.safe_load(open(ROOT / "configs/study1.yaml"))
    a1.MODELS = None
    df = a1.load(cfg)
    rows = []
    # noise scale: SD of paired differences between two irrelevant code-name variants (identical evidence)
    noise = {}
    for (m, amb), g in df[df.condition_type == "noise"].groupby(["model", "ambiguity"]):
        piv = g.pivot_table(index="instance_id", columns="cond_key", values="p")
        cols = sorted(piv.columns)
        d = (piv[cols[0]] - piv[cols[-1]]).dropna()
        noise[(m, amb)] = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    # For each evidence type, the buffer levels actually collected: precise has no random-allocation
    # condition in the design, so it is compared on imp/exp; the other two have all three.
    INDEP = {"precise": ("imp", "exp"), "sparse": ("imp", "exog", "exp"), "qualitative": ("imp", "exog", "exp")}
    for (m, amb), g in df[df.condition_type.isin(["main", "exog"])].groupby(["model", "ambiguity"]):
        for ind in INDEP[amb]:
            gg = g[g.indep == ind]
            if gg.empty:
                continue
            w = gg.pivot_table(index="instance_id", columns="buffer_level", values="p")
            s = summarize_diff(w["high"] - w["low"], "")
            rows.append({"model": m, "ambiguity": amb, "indep": ind, "p_effect": s["mean"], "p": s["p_perm"],
                         "noise_sd": noise.get((m, amb), np.nan),
                         "z": s["mean"] / noise[(m, amb)] if noise.get((m, amb), 0) > 0 else np.nan})
    # forecasts
    for f in sorted((ROOT / "results/raw/forecast_clean").glob("*.jsonl")):
        name = f.stem
        recs = {}
        for l in open(f):
            r = json.loads(l)
            if r["status"] == "ok":
                iid, amb, tag = r["condition_id"].split("__")
                recs[r["condition_id"]] = {"instance_id": iid, "ambiguity": amb, "tag": tag,
                                           "yes": float(r["parsed_response"]["answer"] == "yes")}
        d = pd.DataFrame(recs.values())
        for amb, g in d.groupby("ambiguity"):
            piv = g.pivot_table(index="instance_id", columns="tag", values="yes")
            nz = float((piv["noise0"] - piv["noise1"]).std(ddof=1))
            for ind in ("imp", "exog", "exp"):
                s = summarize_diff(piv[f"high_{ind}"] - piv[f"low_{ind}"], "")
                rows.append({"model": name, "ambiguity": amb, "indep": ind, "p_effect": s["mean"],
                             "p": s["p_perm"], "noise_sd": nz,
                             "z": s["mean"] / nz if nz > 0 else np.nan, "format": "forecast"})
    out = pd.DataFrame(rows)
    out["format"] = out["format"].fillna("probability") if "format" in out.columns else "probability"
    out.to_csv(ROOT / "results/statistics/effect_sizes.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print("=== Effect in prompt-noise SD units (z) ===")
    for (m, ind), g in out[out.indep != "exp"].groupby(["model", "indep"]):
        cells = {f"{r.ambiguity}/{r.format}": f"{r.z:.2f}" for r in g.itertuples()}
        print(f"  {m:26s} {ind:5s} " + "  ".join(f"{k}={v}" for k, v in sorted(cells.items())))
    z = out.set_index(["model", "ambiguity", "indep", "format"]).z
    print("\n=== Compression ratio: z(forecast) / z(numeric probability), paired within model ===")
    for amb in ("precise", "qualitative"):
        for ind in ("imp", "exp", "exog"):
            try:
                a = z.xs((amb, ind, "forecast"), level=("ambiguity", "indep", "format"))
                b = z.xs((amb, ind, "probability"), level=("ambiguity", "indep", "format"))
            except KeyError:
                continue
            r = (a / b).dropna()
            r = r[np.isfinite(r) & (b.reindex(r.index) > 0.02)]
            if len(r):
                print(f"  {amb:12s} {ind:4s} " + "  ".join(f"{k}={v:.1f}" for k, v in r.items()))
    return out


def traces():
    frames = []
    for key, label in TRACE_MODELS.items():
        path = ROOT / "results/raw/study1_pilot" / f"{key}.jsonl"
        if not path.exists():
            continue
        for l in open(path):
            r = json.loads(l)
            if r["status"] != "ok":
                continue
            parts = r["condition_id"].split("__")
            iid, amb, ctype, lvl, ind = parts[0], parts[1], parts[2], parts[3], parts[-1]
            if ctype not in ("main", "exog") or amb not in ("precise", "sparse", "qualitative"):
                continue
            rc = r.get("reasoning_content") or ""
            frames.append({"model": key, "label": label, "instance_id": iid, "ambiguity": amb,
                           "condition_type": ctype, "buffer_level": lvl, "indep": ind,
                           "p": r["parsed_response"]["success_probability"], "len": len(rc),
                           "cover": counts(rc, COVER), "frame": counts(rc, FRAME),
                           "action": counts(rc, ACTION), "hedge": counts(rc, HEDGE)})
    df = pd.DataFrame(frames)
    df["cover_any"] = (df.cover > 0).astype(int)
    df["frame_any"] = (df.frame > 0).astype(int)
    df.to_csv(ROOT / "results/processed/traces_study1.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\n=== Trace audit (shares of responses whose reasoning mentions ...) ===")
    print(df.groupby(["label", "ambiguity"]).agg(n=("p", "size"), mean_len=("len", "mean"),
                                                 cover=("cover_any", "mean"), frame=("frame_any", "mean"),
                                                 hedge=("hedge", "mean")).to_string())
    print("\n=== Trace mentions vs the size of the buffer effect (same model, qualitative evidence) ===")
    rows = []
    for label, g in df[df.ambiguity == "qualitative"].groupby("label"):
        piv = g.pivot_table(index=["instance_id", "indep"], columns="buffer_level", values="p")
        cov = g[g.buffer_level == "high"].groupby(["instance_id", "indep"]).cover_any.first()
        j = piv.join(cov).dropna(subset=["cover_any"])
        j["d"] = j["high"] - j["low"]
        for ind, gg in j.groupby(level="indep"):
            a, b = gg[gg.cover_any == 1]["d"], gg[gg.cover_any == 0]["d"]
            rows.append({"label": label, "indep": ind, "n_mentions": int(gg.cover_any.sum()),
                         "effect_if_mentions": a.mean() if len(a) else np.nan,
                         "effect_if_silent": b.mean() if len(b) else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    standardised()
    traces()
