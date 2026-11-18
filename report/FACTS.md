# Fact sheet: neuroevolution Flappy Bird

**Code: Karthik. Experiment harness and figures: produced with Claude.**
Bullet facts only, for Karthik to rewrite in his own words. Every number below comes from `results.json` / `results.md`. The original `flappy.py` was not modified (md5 `a4324772576b71610a4026534a7560e9`).

---

## 1. What the project is

- A single-file Python game (pygame + NumPy, ~180 lines) in which 50 Flappy Bird agents, each controlled by a tiny neural network, learn to fly through pipes by **neuroevolution**: no gradients, no backprop. The good networks are kept and randomly mutated copies of them make up the next generation.
- Written by Karthik. The only lines in `flappy.py` not written by him are tagged `(added with Claude)`: a headless switch (`HEADLESS`, which sets `SDL_VIDEODRIVER=dummy`), a per-generation frame cap (`MAX_FRAMES = 9000`), a run cap (`MAX_GENERATIONS = 30`) and printing the median score.
- Drawing style: black background, white rectangular pipes, birds as pastel circles (random colour per bird, `rng_color` seed 3).

## 2. How it works (exact numbers from the code)

**Game**
- Screen 1280 x 720 px.
- Bird: fixed x = 100, spawns at y = 100, radius 25. It is drawn as a circle, but collisions use a 50 x 50 square `Rect` centred on the bird.
- Physics per frame: `y += velocity`, then `velocity += 1` (gravity = 1 px/frame^2). A flap sets `velocity = -10`.
- Pipes: width 75 px, gap 200 px, move 10 px/frame to the left, a new pipe every 90 frames (so pipes are 900 px apart), spawn at x = 1280. Top-pipe height = random integer in [150, 370] (`margin` 150).
- The bird dies if its rect overlaps the top or bottom rect of the *next* pipe (the first pipe whose right edge is > 75 = bird's left edge), or if its centre y goes < 0 or > 720.
- All 50 birds in a generation see the **same** pipe layout. Each generation gets a **new** layout, drawn from one `rng_pipe` (seed 2) that keeps running across generations.
- A generation ends when every bird is dead or after 9,000 frames (the cap). 9,000 frames = 99 pipes (the first pipe is cleared at frame 128, then one every 90 frames).

**Brain (class `NN`)**
- 4 inputs, each scaled:
  1. `y / 720`: height (0 to 1 while on screen)
  2. `velocity / 10`: vertical speed (-1 right after a flap; not bounded when falling)
  3. `(next pipe's right edge - 100) / 1280`: horizontal distance to the end of the next pipe
  4. `(gap centre - y) / 720`: vertical offset from the bird to the middle of the gap
- Layers: 4 inputs + bias -> **6 hidden** (tanh) -> 6 + bias -> **1 output** (tanh). Bias is handled by putting a constant 1 in front of each layer's input (`np.hstack((np.ones(1), outputs))`), so the weight matrices are **5 x 6** and **7 x 1**.
- Weight count: 30 + 7 = **37** numbers. That is the whole genome.
- Initial weights: `uniform(-1, 1)` (from `rng_bird`, seed 1).
- Decision rule: flap if the output > 0. Evaluated every frame.

**Evolution**
- Population: 50 birds.
- Fitness: `score` = frames survived (+1 per frame alive). Pipes passed are not counted directly.
- Selection: sort by score and keep the **top 5** unchanged (elitism). They are re-played next generation on a new layout.
- Reproduction: each of the 5 parents produces 9 children (45 total) by adding `uniform(-0.15, 0.15)` noise to **every** one of the 37 weights (`mutation_variation = 0.15`). No crossover.
- Next generation = 5 elites + 45 mutants = 50.

## 3. Results (each tied to a figure)

Setup for all experiments: 20 runs per setting. Run i uses `rng_bird = 1 + 10i` and `rng_pipe = 2 + 10i` (i = 0..19; i = 0 is Karthik's original seeds 1/2). 30 generations, 9,000-frame cap, everything else as in his code. Total compute: about 20 minutes on 2 CPU cores.

**Sanity check:** an unchanged copy with `MAX_GENERATIONS = 4` printed exactly `gen 1: best 115 median 29.0`, `gen 2: best 1287`, `gen 3: best 1111`, `gen 4: best 932`. The instrumented copy reproduces the original's 30-generation printout line for line.

**Exp 1: learning curve, his settings** -> `figures/1_learning_curve.png`
- **20/20 runs** had a best bird that reached the 9,000-frame cap (99 pipes) within 30 generations.
- First generation at the cap: **median 7.5, range 1-25**. Without seed s10 (see the next bullet): median 9 (19 runs).
- Seed s10 hit the cap in **generation 1**, before any evolution: one of its 50 initial random brains was already good. Generation 1 is identical in every experiment for a given seed, so s10 is a lucky draw, not evidence that evolution works.
- Karthik's original seeds (s00): best pipes by generation = 0, 13, 11, 9, 18, 23, 25, 7, 31, 5, 84, 33, 67, 70, then **99 from generation 15 on** (cap at gen 15, held for all 16 remaining generations).
- Median across runs of the best bird's pipes: gen 1 = 0, gen 5 = 25, gen 7 = 78, gen 9 = 99. From gen 11 on: 99 every generation.
- **The typical bird stays bad**: the flock median stays at 0-1 pipes in every generation (median-of-runs flock-median frames: 28.5 in gen 1, about 115-211 later). Frame ~115 is roughly when a bird reaches the first pipe.
- Inside the flock (gens 2-30, all 20 runs; `results.md`, "Who passes pipes"):
  - 5 kept elites (2,900 bird-games): median 99 pipes, 89.4% pass at least 1 pipe, 58.0% reach the cap
  - 45 mutated children (26,100): median 0 pipes, 39.8% pass at least 1 pipe, 12.3% reach the cap
  - So the progress is carried by the elites. Most ±0.15 mutants of a capped parent do not even clear pipe 1.
- Stability after solving: in only **3 of 407** generations after a run first hit the cap did the best bird fall below the cap (best = 29, 32, 52 pipes; runs s05 and s06).

**Exp 2: evolution vs random search, same budget** -> `figures/2_evolution_vs_random.png`
- Random search = every generation is 50 brand-new random brains (no inheritance). Same seeds, same 30 x 50 = 1,500 games per run.
- Reached the cap: **evolution 20/20 runs, random search 3/20** (s10 at gen 1, s06 and s11 at gen 9).
- Best pipes so far after 30 generations, median of runs: **evolution 99, random search 7.5** (the figure label rounds it to 8).
- Over all 30,000 random brains: 0.94% passed at least 1 pipe, 0.04% passed 10 or more, and **3** reached the cap (about 1 in 10,000).
- Honest reading: the task is easy enough that pure luck sometimes solves it (3/20 runs), but evolution solves it reliably (20/20).

**Exp 3: mutation-size sweep** -> `figures/3_mutation_sweep.png`
- `mutation_variation` in {0.05, 0.15 (his), 0.5, 1.0}, the same 20 seed pairs each.

| mutation | runs that reached the cap | median first gen at cap | range |
|---|---|---|---|
| ±0.05 | 11/20 | 12 | 1-23 |
| ±0.15 (his) | 20/20 | 7.5 | 1-25 |
| ±0.5 | 20/20 | 9 | 1-18 |
| ±1.0 | 20/20 | 10 | 1-22 |

- ±0.05 is clearly too small within 30 generations (9 runs never reached the cap).
- ±0.15, ±0.5 and ±1.0 overlap heavily. With 20 seeds you **cannot** say which of these three is fastest. Do not claim his 0.15 is "optimal".

**Exp 4: held-out layouts ("did it learn the game or one lucky layout?")** -> `figures/4_heldout_test.png`
- Each run's final top elite (gen 30) played 20 new pipe layouts never seen in training (`rng_pipe` seeds 10,000,000-10,000,019), same rules, same 9,000-frame cap. Random-search control: each run's best brain from any generation.

| brains (20 each) | median of per-brain median pipes | brains with median = 99 | brains at cap on all 20 layouts | games at cap (of 400) |
|---|---|---|---|---|
| evolution ±0.05 | 60.25 | 10 | 7 | 193 |
| **evolution ±0.15 (his)** | **99** | **19** | **16** | **376** |
| evolution ±0.5 | 99 | 20 | 19 | 399 |
| evolution ±1.0 | 99 | 19 | 10 | 343 |
| random search (control) | 3 | 2 | 0 | 47 |

- His original-seed brain (s00): **99 pipes on all 20 unseen layouts**.
- His settings, the 4 brains that were not perfect: s01 (min 27, 16/20 at cap), s05 (min 94, 19/20), s13 (min 2, 13/20), s19 (median 62.5, min 5, 8/20).
- The ±0.05 results split in two: brains either play perfectly or barely pass a pipe (the median of 60 sits between the two groups).
- The random-search control's "best" brains (chosen by one lucky score) mostly fail on new layouts (median 3 pipes). The evolved brains do not. This is the evidence that evolution found a general controller, not one memorised layout.
- Caveat: "general" here means the same game with new random pipe heights. Nothing was tested with a different gap, speed or physics.

**Demo** -> `figures/demo.gif`: the first ~600 frames (about 9 s at 30 fps, every 2nd frame) of generation 30 with his original seeds, drawn in his own style. 640 x 360, 1.25 MB.

## 4. Limitations (honest list)

- **One random layout per generation makes selection noisy.** A bird is ranked on a single layout, so a brain can be kept for being lucky, and a good brain can be dropped after one bad layout. (Measured: after reaching the cap, the best bird fell below it in only 3/407 generations, so in practice it rarely hurt here.)
- **Fitness counts frames, not pipes.** In this game the two always agree: pipes = floor((frames - 128) / 90) + 1 for frames >= 128, else 0. This held for all 150,000 bird-games logged. Because pipes come at a fixed rhythm, frames rank birds the same way pipes do, and also break ties *between* pipes (surviving longer after the last pipe counts). That stops being true if the pipe spacing or speed ever varies.
- **The fitness saturates at the cap.** Once several birds survive 9,000 frames they tie, and which 5 are kept depends on list order (Python's `sorted` is stable and the top 5 are the last 5), not on skill.
- **The out-of-bounds check uses the bird's centre, not its edge.** A bird can be half off-screen (top or bottom) and still alive.
- **Collision box vs drawing:** collisions use the 50 x 50 square, but the bird is drawn as a radius-25 circle, so a corner can "hit" a pipe the circle does not touch.
- **Fall speed is unbounded**, so the velocity input `velocity / 10` can go well above 1 (falling freely from the top to the bottom gives a speed of about 38 px/frame, i.e. an input of about 3.8). The other inputs stay within about [-1, 1].
- **`while len(new_birds) < 50`** only lands on exactly 50 because 45 is divisible by 5 (each pass of the inner loop adds 5 children). With a different elite count (e.g. 4 or 6), the population would overshoot 50.
- **No saving or loading of the best brain** in his version. A trained network disappears when the program exits. (The harness saves weights only for the experiments.)
- Every child mutates **all 37 weights** at once (there is no per-weight mutation rate). Most children of a capped parent fail at pipe 1 (see Exp 1).
- Experiment scale: 20 seeds per setting, 30 generations, 1 task. Differences between ±0.15, ±0.5 and ±1.0 are within noise.

## 5. What the experiment harness changed (exact list)

All changes are text substitutions applied to a **copy** of `flappy.py` (`harness/make_variant.py`; each target must match exactly once). The game rules are untouched.
1. `MAX_GENERATIONS = 30` kept at 30 (1 for the held-out test). The sanity check used 4.
2. `rng_bird` seed `1` -> `1 + 10i`, `rng_pipe` seed `2` -> `2 + 10i` (i = 0..19; i = 0 = original). `rng_color` left at 3.
3. `mutation_variation = 0.15` -> 0.05 / 0.5 / 1.0 (Exp 3 only).
4. Added `self.pipes = 0` to `Bird.__init__` (a counter, not used by the game).
5. Added one call per frame, right after `next_pipe` is chosen: when a pipe's right edge reaches the bird's left edge (x <= 75, i.e. the game stops checking collisions with that pipe), every bird still alive gets +1 pipe. Note: this counts a pipe as passed when it is fully *cleared*, which is slightly stricter than "the right edge passes the bird's centre x = 100".
6. Added one logging call after his `print(...)` at the end of each generation: writes every bird's frames and pipes, plus the top-ranked bird's weights (ranked with the same `sorted(..., key=score)` as his `next_generation`), to a JSONL file.
7. A block of helper functions (for 5 and 6) inserted just before `pipes = [Pipe(rng_pipe)]`.
8. **Random-search baseline only:** first line of `next_generation` replaced so it returns 50 new random `Bird`s (no parents, no mutation).
9. **Held-out test only** (`harness/heldout_test.py`): the initial flock line is replaced by one bird per saved brain, `rng_pipe` = test seed, 1 generation. Birds do not interact, so testing them as one flock = testing each alone. Check: the gen-1 best brain of the original run replayed on layout seed 2 scores exactly 115 frames, as in training.
10. **GIF only** (`harness/make_gif.py`): `HEADLESS = False` with `SDL_VIDEODRIVER=dummy` set in the environment (so his draw code runs without a window), plus frame capture of generation 30, stopping after 600 frames. Its printed training log matches the original run.

## 6. How to reproduce

```bash
pip install --break-system-packages pygame-ce numpy matplotlib pillow
# everything (about 20 min on 2 cores):
bash harness/run_all.sh /path/to/flappy.py /tmp/flappy-exp ./flappy-report
# or step by step:
python harness/run_experiments.py flappy.py /tmp/flappy-exp --seeds 20 --only evo,random   # Exp 1 + 2
python harness/run_experiments.py flappy.py /tmp/flappy-exp --seeds 20 --only sweep        # Exp 3
python harness/heldout_test.py    flappy.py /tmp/flappy-exp --layouts 20                   # Exp 4
python harness/analyze.py /tmp/flappy-exp ./flappy-report                                  # figures + tables
python harness/make_gif.py flappy.py ./flappy-report/figures/demo.gif                      # demo
```
- All runs are deterministic (fixed NumPy seeds). Per-run wall time: about 1 s for random search and 3-60 s for evolution.
- Files: `results.json` (summary + every bird's frames and pipes for every generation of every run + the final best weights), `heldout_raw.json` (pipes and frames for each of the 100 test brains on each of the 20 layouts), `results.md` (tables).
