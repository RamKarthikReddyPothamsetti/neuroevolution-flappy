"""Build an experiment copy of flappy.py by exact text substitution.

The original file is never modified. Every substitution must match exactly
once, otherwise this script stops with an error, so the list below is the
complete list of changes made for the experiments.

Usage (normally called from run_experiments.py):
    python make_variant.py ORIGINAL OUT.py --bird-seed 1 --pipe-seed 2 \
        --mutation 0.15 --generations 30 [--random-search] --log run.jsonl
"""
import argparse

INSTRUMENT_HELPERS = '''
# ---- experiment instrumentation (harness, not part of the original) ----
import json as _json
_LOG = open(LOG_PATH, "w")
_last_rights = {}
def _count_cleared(pipes, birds):
    # A pipe is "cleared" on the frame its right edge reaches the bird's
    # left edge (x <= spawn_x - radius = 75). From then on the game no longer
    # checks collisions against it. Every bird still alive at that moment gets +1.
    for p in pipes:
        r = p.rect_top.right
        prev = _last_rights.get(id(p), r + Pipe.speed)
        if prev > Bird.spawn_x - Bird.radius >= r:
            for b in birds:
                if b.is_alive:
                    b.pipes += 1
        _last_rights[id(p)] = r
def _log_generation(generation, birds):
    ranked_ = sorted(birds, key = lambda b : b.score)   # same sort as next_generation
    best_bird = ranked_[-1]                             # = the top elite
    _LOG.write(_json.dumps({
        "generation": int(generation),
        "scores": [int(b.score) for b in birds],
        "pipes": [int(b.pipes) for b in birds],
        "best_brain_score": int(best_bird.score),
        "best_brain_pipes": int(best_bird.pipes),
        "best_brain": [l.tolist() for l in best_bird.nn.layers],
    }) + "\\n")
    _LOG.flush()
    _last_rights.clear()
# ---- end instrumentation ----
'''


def build(src, bird_seed, pipe_seed, mutation, generations, random_search, log_path):
    subs = [
        # 1. run length
        ("MAX_GENERATIONS = 30 ", f"MAX_GENERATIONS = {generations} "),
        # 2. seeds (colour seed left at 3; it only affects drawing colours)
        ("rng_bird = np.random.default_rng(1)", f"rng_bird = np.random.default_rng({bird_seed})"),
        ("rng_pipe = np.random.default_rng(2)", f"rng_pipe = np.random.default_rng({pipe_seed})"),
        # 3. mutation size
        ("    mutation_variation = 0.15", f"    mutation_variation = {mutation}"),
        # 4. per-bird pipe counter
        ("        self.score = 0\n", "        self.score = 0\n        self.pipes = 0   # harness\n"),
        # 5. count cleared pipes each frame (before birds move, after pipes move)
        ("    next_pipe = next((pipe for pipe in pipes if pipe.rect_top.right > Bird.spawn_x - Bird.radius), None)\n",
         "    next_pipe = next((pipe for pipe in pipes if pipe.rect_top.right > Bird.spawn_x - Bird.radius), None)\n"
         "    _count_cleared(pipes, birds)   # harness\n"),
        # 6. log every generation (scores, pipes, best brain weights)
        ('        print(f"gen {generation}: best {best} median {median}")\n',
         '        print(f"gen {generation}: best {best} median {median}")\n'
         '        _log_generation(generation, birds)   # harness\n'),
        # 7. helper definitions inserted just before the game starts
        ("pipes = [Pipe(rng_pipe)]\nbirds = [Bird(rng_bird, rng_color) for i in range(50)]\n",
         f"LOG_PATH = {log_path!r}\n" + INSTRUMENT_HELPERS +
         "pipes = [Pipe(rng_pipe)]\nbirds = [Bird(rng_bird, rng_color) for i in range(50)]\n"),
    ]
    if random_search:
        # 8. random-search baseline: no inheritance, 50 brand-new random brains each generation
        subs.append(("def next_generation(birds):\n",
                     "def next_generation(birds):\n"
                     "    return [Bird(rng_bird, rng_color) for i in range(50)]   # harness: random search\n"))
    out = src
    for old, new in subs:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"substitution target found {n} times (expected 1): {old!r}")
        out = out.replace(old, new)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("out")
    ap.add_argument("--bird-seed", type=int, default=1)
    ap.add_argument("--pipe-seed", type=int, default=2)
    ap.add_argument("--mutation", type=float, default=0.15)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--random-search", action="store_true")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    src = open(a.original).read()
    open(a.out, "w").write(build(src, a.bird_seed, a.pipe_seed, a.mutation,
                                 a.generations, a.random_search, a.log))
