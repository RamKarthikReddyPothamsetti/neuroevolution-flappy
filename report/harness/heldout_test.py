"""Experiment 4: play saved brains on pipe layouts never used in training.

For each test layout seed (10_000_000 + k, k = 0..19) we run ONE generation of a
text-substituted copy of flappy.py in which the flock is replaced by the saved
brains (one bird per brain; birds never interact, so a flock of test brains is
the same as testing each alone). Same game rules, same 9000-frame cap.

Brains tested:
  - evolution runs: the top elite of the LAST generation (gen 30) of each run
  - random-search runs: the best brain seen in any generation (highest frames;
    earliest generation on ties) -- a control

    python heldout_test.py /path/to/flappy.py OUTDIR [--layouts 20]
"""
import argparse, glob, json, os, subprocess, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_variant import build

BIRDS_LINE = "birds = [Bird(rng_bird, rng_color) for i in range(50)]\n"


def pick_brains(outdir):
    names, brains = [], []
    for f in sorted(glob.glob(os.path.join(outdir, "runs", "*.json"))):
        r = json.load(open(f))
        gens = r["generations"]
        if r["random_search"]:
            g = max(gens, key=lambda g: g["best_brain_score"])  # max() keeps the earliest on ties
        else:
            g = gens[-1]
        names.append(r["name"])
        brains.append(g["best_brain"])
    return names, brains


def play(job):
    original, outdir, seed, brains_path = job
    work = os.path.join(outdir, "heldout")
    script = os.path.join(work, f"test_{seed}.py")
    log = os.path.join(work, f"test_{seed}.jsonl")
    src = build(open(original).read(), 1, seed, 0.15, 1, False, log)
    assert src.count(BIRDS_LINE) == 1
    src = src.replace(BIRDS_LINE,
        "import json as _j2\n"
        f"birds = [Bird(rng_bird, rng_color, NN(rng_bird, [np.array(l) for l in b])) for b in _j2.load(open({brains_path!r}))]   # harness: saved brains\n")
    open(script, "w").write(src)
    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1", SDL_AUDIODRIVER="dummy")
    subprocess.run([sys.executable, script], cwd=work, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    g = json.loads(open(log).readline())
    return seed, g["scores"], g["pipes"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("outdir")
    ap.add_argument("--layouts", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, "heldout"), exist_ok=True)
    names, brains = pick_brains(a.outdir)
    bp = os.path.join(a.outdir, "heldout", "brains.json")
    json.dump(brains, open(bp, "w"))
    seeds = [10_000_000 + k for k in range(a.layouts)]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(play, [(a.original, a.outdir, s, bp) for s in seeds])
    out = {"layout_seeds": seeds, "brains": {}}
    for i, n in enumerate(names):
        out["brains"][n] = {"frames": [r[1][i] for r in res], "pipes": [r[2][i] for r in res]}
    json.dump(out, open(os.path.join(a.outdir, "heldout.json"), "w"))
    print("tested", len(names), "brains on", len(seeds), "layouts")
