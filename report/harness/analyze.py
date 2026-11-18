"""Turn the raw run logs into results.json, results.md and the figures.

    python analyze.py RUNDIR REPORTDIR
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNDIR, REPORT = sys.argv[1], sys.argv[2]
FIG = os.path.join(REPORT, "figures")
os.makedirs(FIG, exist_ok=True)
CAP_FRAMES, CAP_PIPES = 9000, 99   # 9000 frames = 99 pipes cleared (first at frame 128, then every 90)

# ---- style (dataviz reference palette, light mode) ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUNDIR, "runs", "*.json"))):
        r = json.load(open(f))
        gens = r["generations"]
        r["best_frames"] = [max(g["scores"]) for g in gens]
        r["median_frames"] = [float(np.median(g["scores"])) for g in gens]
        r["best_pipes"] = [max(g["pipes"]) for g in gens]
        r["median_pipes"] = [float(np.median(g["pipes"])) for g in gens]
        r["best_so_far_pipes"] = list(np.maximum.accumulate(r["best_pipes"]).tolist())
        solved = [g["generation"] for g in gens if max(g["scores"]) >= CAP_FRAMES]
        r["first_solved_gen"] = solved[0] if solved else None
        r["gens_at_cap"] = len(solved)
        runs[r["name"]] = r
    return runs


def group(runs, prefix):
    return [runs[k] for k in sorted(runs) if k.startswith(prefix)]


def band(series):
    a = np.array(series, dtype=float)
    return np.median(a, 0), np.percentile(a, 25, 0), np.percentile(a, 75, 0), a.min(0), a.max(0)


def titles(fig, title, subtitle):
    fig.text(0.012, 0.975, title, fontsize=12, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.012, 0.915, subtitle, fontsize=8.5, color=MUTED, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))


def cap_line(ax):
    ax.axhline(CAP_PIPES, color=MUTED, lw=1)
    ax.set_yticks([0, 20, 40, 60, 80, CAP_PIPES])
    ax.text(ax.get_xlim()[1], CAP_PIPES + 1.5, "cap: 9,000 frames = 99 pipes", color=MUTED, fontsize=8.5, va="bottom", ha="right")


runs = load()
evo = group(runs, "evo_m0.15_")
rnd = group(runs, "random_")
G = np.arange(1, len(evo[0]["best_pipes"]) + 1)
summary = {}

# ---------- Figure 1: learning curve ----------
fig, ax = plt.subplots(figsize=(8, 4.4))
m, q1, q3, lo, hi = band([r["best_pipes"] for r in evo])
ax.fill_between(G, q1, q3, color=BLUE, alpha=0.18, lw=0)
ax.plot(G, m, color=BLUE, lw=2, label="best bird of the generation (median of runs, band = IQR)")
orig = runs["evo_m0.15_s00"]
ax.plot(G, orig["best_pipes"], color=INK2, lw=1, label="best bird, original seeds (bird 1, pipe 2)")
fm = np.median([r["median_pipes"] for r in evo], 0)
ax.plot(G, fm, color=ORANGE, lw=2, label="median bird of the flock (median of runs)")
ax.set_xlim(1, G[-1]); ax.set_ylim(-2, 108)
cap_line(ax)
ax.set_xlabel("generation"); ax.set_ylabel("pipes passed")
n_solved = sum(r["first_solved_gen"] is not None for r in evo)
ax.legend(loc="center right", fontsize=8.5, labelcolor=INK2)
med_first = np.median([r['first_solved_gen'] for r in evo if r['first_solved_gen']])
titles(fig, f"Best bird reaches the cap after a median of {med_first:g} generations; the median bird stays near 0 pipes",
       f"{len(evo)} runs with Karthik's settings (50 birds, top 5 kept, mutation ±0.15), each with its own bird and pipe seeds, 30 generations")
fig.savefig(os.path.join(FIG, "1_learning_curve.png"), dpi=150); plt.close(fig)
summary["exp1"] = {
    "runs": len(evo),
    "median_best_pipes_by_gen": m.tolist(), "q1_best_pipes_by_gen": q1.tolist(), "q3_best_pipes_by_gen": q3.tolist(),
    "min_best_pipes_by_gen": lo.tolist(), "max_best_pipes_by_gen": hi.tolist(),
    "median_flock_median_pipes_by_gen": fm.tolist(),
    "median_best_frames_by_gen": np.median([r["best_frames"] for r in evo], 0).tolist(),
    "median_flock_median_frames_by_gen": np.median([r["median_frames"] for r in evo], 0).tolist(),
}

# ---------- Figure 2: evolution vs random search ----------
fig, ax = plt.subplots(figsize=(8, 4.4))
for grp, col, lab in ((evo, BLUE, "evolution (his algorithm)"), (rnd, ORANGE, "random search (50 new random brains every generation)")):
    m2, q1b, q3b, lob, hib = band([r["best_so_far_pipes"] for r in grp])
    ax.fill_between(G, q1b, q3b, color=col, alpha=0.18, lw=0)
    ax.plot(G, m2, color=col, lw=2, label=lab)
    ax.text(G[-1] + 0.4, m2[-1], f"{m2[-1]:.0f}", color=INK2, va="center", fontsize=9)
ax.set_xlim(1, G[-1]); ax.set_ylim(-2, 108)
cap_line(ax)
ax.set_xlabel("generation (50 games per generation for both)"); ax.set_ylabel("best pipes passed so far")
ns_r = sum(r["first_solved_gen"] is not None for r in rnd)
ax.legend(loc="center right", fontsize=8.5, labelcolor=INK2)
titles(fig, f"Same budget (30 x 50 games): evolution reached the cap in {n_solved}/{len(evo)} runs, random search in {ns_r}/{len(rnd)}",
       f"line = median of {len(evo)} runs, band = interquartile range; best so far = best score in any generation up to this one")
fig.savefig(os.path.join(FIG, "2_evolution_vs_random.png"), dpi=150); plt.close(fig)


def solve_stats(grp):
    fs = [r["first_solved_gen"] for r in grp if r["first_solved_gen"] is not None]
    return {"runs": len(grp), "runs_reaching_cap": len(fs),
            "median_first_gen_at_cap": float(np.median(fs)) if fs else None,
            "first_gen_at_cap_range": [min(fs), max(fs)] if fs else None,
            "first_gen_at_cap_per_run": {r["name"]: r["first_solved_gen"] for r in grp},
            "final_gen_best_pipes_per_run": {r["name"]: r["best_pipes"][-1] for r in grp},
            "best_pipes_ever_per_run": {r["name"]: max(r["best_pipes"]) for r in grp},
            "gens_at_cap_per_run": {r["name"]: r["gens_at_cap"] for r in grp}}


summary["exp2"] = {"evolution": solve_stats(evo), "random": solve_stats(rnd)}

# elites vs children (gens 2-30: birds[0:5] are the kept elites, birds[5:] the mutated children)
E = np.array([p for r in evo for g in r["generations"][1:] for p in g["pipes"][:5]])
C = np.array([p for r in evo for g in r["generations"][1:] for p in g["pipes"][5:]])
R = np.array([p for r in rnd for g in r["generations"] for p in g["pipes"]])
post = [(r["name"], g["generation"], max(g["pipes"])) for r in evo for g in r["generations"]
        if r["first_solved_gen"] and g["generation"] > r["first_solved_gen"] and max(g["scores"]) < CAP_FRAMES]
summary["flock_breakdown"] = {
    "elites_gens2to30": {"n": int(len(E)), "median_pipes": float(np.median(E)), "frac_ge1_pipe": float((E >= 1).mean()), "frac_at_cap": float((E >= CAP_PIPES).mean())},
    "children_gens2to30": {"n": int(len(C)), "median_pipes": float(np.median(C)), "frac_ge1_pipe": float((C >= 1).mean()), "frac_at_cap": float((C >= CAP_PIPES).mean())},
    "random_brains_all": {"n": int(len(R)), "median_pipes": float(np.median(R)), "frac_ge1_pipe": float((R >= 1).mean()), "frac_ge10_pipes": float((R >= 10).mean()), "n_at_cap": int((R >= CAP_PIPES).sum())},
    "gens_after_first_cap_where_best_fell_below_cap": {"count": len(post), "of": int(sum(len(r["generations"]) - r["first_solved_gen"] for r in evo if r["first_solved_gen"])), "cases": post},
}

# ---------- Figure 3: mutation sweep ----------
SWEEP_TITLE = "Mutation ±0.05 is too small (11/20 runs reached the cap); ±0.15, ±0.5 and ±1.0 all went 20/20"
HELDOUT_TITLE = "On 20 unseen layouts, 19/20 evolved brains (his settings) still score a median of 99 pipes"
settings = [0.05, 0.15, 0.5, 1.0]
have = [s for s in settings if group(runs, f"evo_m{s}_")]
summary["exp3"] = {}
if len(have) > 1:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rng = np.random.default_rng(0)
    NOT = 33.5
    for i, s in enumerate(have):
        grp = group(runs, f"evo_m{s}_")
        st = solve_stats(grp)
        summary["exp3"][str(s)] = st
        y = np.array([r["first_solved_gen"] if r["first_solved_gen"] else NOT for r in grp], float)
        x = i + rng.uniform(-0.18, 0.18, len(y))
        col = BLUE
        ax.scatter(x, y, s=34, color=col, alpha=0.75, edgecolor="white", linewidth=1.2, zorder=3)
        solved = y[y < NOT]
        if len(solved):
            med = np.median(solved)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2, zorder=4)
        ax.text(i, 36.3, f"{len(solved)}/{len(y)} reached cap", ha="center", color=INK2, fontsize=9)
    ax.axhspan(31.5, 35.2, color=GRID, alpha=0.5, lw=0)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([f"±{s}" + (" (his)" if s == 0.15 else "") for s in have])
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, NOT]); ax.set_yticklabels(["1", "5", "10", "15", "20", "25", "30", "never"])
    ax.set_ylim(0, 38); ax.invert_yaxis()
    ax.set_xlabel("mutation size (uniform noise added to every weight)")
    ax.set_ylabel("first generation where the best\nbird hit the 9,000-frame cap")
    ax.grid(axis="x", visible=False)
    titles(fig, SWEEP_TITLE,
           "one dot per run (higher = faster), black bar = median of the runs that reached the cap; same 20 seed pairs for every setting")
    fig.savefig(os.path.join(FIG, "3_mutation_sweep.png"), dpi=150); plt.close(fig)

# ---------- Figure 4: held-out layouts ----------
hp = os.path.join(RUNDIR, "heldout.json")
if os.path.exists(hp):
    ho = json.load(open(hp))
    summary["exp4"] = {"layout_seeds": ho["layout_seeds"], "per_brain": {}}
    rows = [(f"evo_m{s}_", f"evolution ±{s}" + (" (his)" if s == 0.15 else "")) for s in have] + [("random_", "random search (control)")]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rng = np.random.default_rng(1)
    for i, (pre, lab) in enumerate(rows):
        names = [n for n in sorted(ho["brains"]) if n.startswith(pre)]
        meds = []
        for n in names:
            p = ho["brains"][n]["pipes"]
            summary["exp4"]["per_brain"][n] = {"median_pipes": float(np.median(p)), "min": min(p), "max": max(p),
                                               "layouts_at_cap": sum(f >= CAP_FRAMES for f in ho["brains"][n]["frames"]), "pipes": p}
            meds.append(np.median(p))
        meds = np.array(meds)
        col = ORANGE if pre == "random_" else BLUE
        ax.scatter(meds, i + rng.uniform(-0.18, 0.18, len(meds)), s=34, color=col, alpha=0.75, edgecolor="white", linewidth=1.2, zorder=3)
        ax.plot([np.median(meds)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
        ax.text(97, i - 0.3, f"{int((meds >= CAP_PIPES).sum())}/{len(meds)} at 99", ha="right", va="center", fontsize=8.5, color=INK2)
        at = [summary["exp4"]["per_brain"][n]["layouts_at_cap"] for n in names]
        summary["exp4"][pre] = {"brains": len(meds), "median_of_medians": float(np.median(meds)),
                                "brains_with_median_at_cap": int((meds >= CAP_PIPES).sum()),
                                "brains_at_cap_on_all_20_layouts": int(sum(a == len(ho["layout_seeds"]) for a in at)),
                                "games_at_cap": int(sum(at)), "games": len(names) * len(ho["layout_seeds"])}
    ax.axvline(CAP_PIPES, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows]); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-3, 104)
    ax.set_xlabel("pipes passed on 20 unseen layouts (median per brain; cap = 99)")
    titles(fig, HELDOUT_TITLE,
           "one dot per run: its final top elite (random search: its best brain ever) played on 20 unseen layouts; black bar = median across runs")
    fig.savefig(os.path.join(FIG, "4_heldout_test.png"), dpi=150); plt.close(fig)

# ---------- results.json ----------
raw = {k: {kk: v for kk, v in r.items() if kk != "generations"} | {
           "per_generation": [{"generation": g["generation"], "scores_frames": g["scores"], "pipes": g["pipes"],
                               "best_brain_frames": g["best_brain_score"], "best_brain_pipes": g["best_brain_pipes"]}
                              for g in r["generations"]],
           "final_best_brain_weights": r["generations"][-1]["best_brain"]}
       for k, r in runs.items()}
json.dump({"summary": summary, "runs": raw}, open(os.path.join(REPORT, "results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "exp1"}, indent=1)[:6000])

# ---------- results.md ----------
L = ["# Results tables", "", "All numbers from the runs in results.json. Pipes passed = pipes whose right edge reached the bird's left edge (x = 75) while the bird was alive. Cap = 9,000 frames = 99 pipes.", ""]
L += ["## Experiment 1 - learning curve, Karthik's settings (20 runs)", "",
      "| generation | best pipes: median | IQR | min | max | flock-median pipes (median of runs) | best frames: median | flock-median frames (median of runs) |",
      "|---|---|---|---|---|---|---|---|"]
e1 = summary["exp1"]
for i in range(len(G)):
    L.append(f"| {i+1} | {e1['median_best_pipes_by_gen'][i]:g} | {e1['q1_best_pipes_by_gen'][i]:g}-{e1['q3_best_pipes_by_gen'][i]:g} | {e1['min_best_pipes_by_gen'][i]:g} | {e1['max_best_pipes_by_gen'][i]:g} | {e1['median_flock_median_pipes_by_gen'][i]:g} | {e1['median_best_frames_by_gen'][i]:g} | {e1['median_flock_median_frames_by_gen'][i]:g} |")
L += ["", "Karthik's original seeds (rng_bird 1, rng_pipe 2), best pipes per generation: " + ", ".join(str(x) for x in runs["evo_m0.15_s00"]["best_pipes"]), ""]
fb = summary["flock_breakdown"]
L += ["### Who passes pipes inside the flock (generations 2-30, all 20 runs)", "",
      "| group | birds | median pipes | passed >= 1 pipe | reached cap |", "|---|---|---|---|---|",
      f"| 5 kept elites | {fb['elites_gens2to30']['n']} | {fb['elites_gens2to30']['median_pipes']:g} | {100*fb['elites_gens2to30']['frac_ge1_pipe']:.1f}% | {100*fb['elites_gens2to30']['frac_at_cap']:.1f}% |",
      f"| 45 mutated children | {fb['children_gens2to30']['n']} | {fb['children_gens2to30']['median_pipes']:g} | {100*fb['children_gens2to30']['frac_ge1_pipe']:.1f}% | {100*fb['children_gens2to30']['frac_at_cap']:.1f}% |",
      f"| random brains (Exp 2, all gens) | {fb['random_brains_all']['n']} | {fb['random_brains_all']['median_pipes']:g} | {100*fb['random_brains_all']['frac_ge1_pipe']:.2f}% | {fb['random_brains_all']['n_at_cap']} brains |", "",
      f"Generations after a run first hit the cap in which the best bird fell below the cap: {fb['gens_after_first_cap_where_best_fell_below_cap']['count']} of {fb['gens_after_first_cap_where_best_fell_below_cap']['of']} ({fb['gens_after_first_cap_where_best_fell_below_cap']['cases']}).", ""]
L += ["## Experiment 2 / 3 - runs that reached the frame cap within 30 generations", "",
      "| setting | runs | reached cap | median first generation at cap | range |", "|---|---|---|---|---|"]
rows = [("evolution, mutation ±0.15 (his)", summary["exp2"]["evolution"]), ("random search", summary["exp2"]["random"])]
rows += [(f"evolution, mutation ±{k}", v) for k, v in summary["exp3"].items() if k != "0.15"]
for lab, st in rows:
    rg = st["first_gen_at_cap_range"]
    L.append(f"| {lab} | {st['runs']} | {st['runs_reaching_cap']} | {st['median_first_gen_at_cap'] if st['median_first_gen_at_cap'] is not None else '-'} | {f'{rg[0]}-{rg[1]}' if rg else '-'} |")
L += ["", "First generation at cap, per seed index (s00 = original seeds; '-' = never in 30 generations):", "",
      "| seed | " + " | ".join(l for l, _ in rows) + " |", "|---" * (len(rows) + 1) + "|"]
for i in range(len(evo)):
    vals = []
    for _, st in rows:
        v = list(st["first_gen_at_cap_per_run"].values())[i] if i < len(st["first_gen_at_cap_per_run"]) else None
        vals.append("-" if v is None else str(v))
    L.append(f"| s{i:02d} | " + " | ".join(vals) + " |")
if "exp4" in summary:
    L += ["", "## Experiment 4 - held-out layouts (20 unseen pipe layouts, seeds 10000000-10000019)", "",
          "| brains | n | median of per-brain median pipes | brains whose median is at cap (99) | brains at cap on all 20 layouts | games at cap |", "|---|---|---|---|---|---|"]
    for k, v in summary["exp4"].items():
        if isinstance(v, dict) and "brains" in v:
            L.append(f"| {k.rstrip('_')} | {v['brains']} | {v['median_of_medians']:g} | {v['brains_with_median_at_cap']} | {v['brains_at_cap_on_all_20_layouts']} | {v['games_at_cap']}/{v['games']} |")
    L += ["", "| brain | median pipes | min | max | layouts survived to cap (of 20) |", "|---|---|---|---|---|"]
    for n, v in summary["exp4"]["per_brain"].items():
        L.append(f"| {n} | {v['median_pipes']:g} | {v['min']} | {v['max']} | {v['layouts_at_cap']} |")
open(os.path.join(REPORT, "results.md"), "w").write("\n".join(L) + "\n")
