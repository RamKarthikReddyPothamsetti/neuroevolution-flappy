"""Run the training experiments (1: evolution, 2: random search, 3: mutation sweep).

Each run = one process running a text-substituted copy of flappy.py
(see make_variant.py). Runs are spread over all CPU cores.

    python run_experiments.py /path/to/flappy.py OUTDIR [--seeds 20] [--only evo,random,sweep]

Seed scheme: run i (i = 0..N-1) uses rng_bird = 1 + 10*i, rng_pipe = 2 + 10*i,
so i = 0 is Karthik's original seeds (1, 2). rng_color stays 3.
"""
import argparse, json, os, subprocess, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_variant import build

GENERATIONS = 30


def run_one(job):
    name, original, outdir, bird_seed, pipe_seed, mutation, random_search = job
    work = os.path.join(outdir, "runs")
    os.makedirs(work, exist_ok=True)
    script = os.path.join(work, name + ".py")
    log = os.path.join(work, name + ".jsonl")
    src = open(original).read()
    open(script, "w").write(build(src, bird_seed, pipe_seed, mutation, GENERATIONS, random_search, log))
    t0 = time.time()
    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1", SDL_AUDIODRIVER="dummy")
    subprocess.run([sys.executable, script], cwd=work, env=env,
                   stdout=open(os.path.join(work, name + ".out"), "w"), stderr=subprocess.DEVNULL, check=True)
    gens = [json.loads(l) for l in open(log)]
    return {"name": name, "bird_seed": bird_seed, "pipe_seed": pipe_seed, "mutation": mutation,
            "random_search": random_search, "seconds": round(time.time() - t0, 1), "generations": gens}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("outdir")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--only", default="evo,random,sweep")
    a = ap.parse_args()
    which = a.only.split(",")
    jobs = []
    for i in range(a.seeds):
        bs, ps = 1 + 10 * i, 2 + 10 * i
        if "evo" in which:
            jobs.append((f"evo_m0.15_s{i:02d}", a.original, a.outdir, bs, ps, 0.15, False))
        if "random" in which:
            jobs.append((f"random_s{i:02d}", a.original, a.outdir, bs, ps, 0.15, True))
        if "sweep" in which:
            for m in (0.05, 0.5, 1.0):
                jobs.append((f"evo_m{m}_s{i:02d}", a.original, a.outdir, bs, ps, m, False))
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            json.dump(r, open(os.path.join(a.outdir, "runs", r["name"] + ".json"), "w"))
            print(f"{r['name']}: {r['seconds']} s  (elapsed {time.time() - t0:.0f} s)", flush=True)
    print(f"total wall time {time.time() - t0:.0f} s on {os.cpu_count()} cores")
