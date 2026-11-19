# Neuroevolution Flappy Bird

My evolution-based Flappy Bird program is basically training a neural network to make the bird cross through a target number of 99 pipes in the least amount of generations possible.

![demo](report/figures/demo.gif)

## How it works

The inputs for the neural network are:
1. the height of the bird
2. its velocity
3. the distance to the nearest pipe to its right
4. the distance between the bird and that pipe's gap centre

For each generation, we sort by which weights enable the bird to stay alive the longest, and pick the top 5 birds' weights and use those ones to create mutated copies by adding noise, so the top 5 plus 45 mutated birds make the 50 birds for the next generation.

A generation ends when every bird is dead or after 9,000 frames (= 99 pipes, counted as "solved").

## Run it

```bash
pip install numpy pygame-ce
python flappy.py
```

## Results

These results were evaluated over 20 random seeds, and I used 30 generations for each seed to calculate the median number of pipes crossed per generation as well. Full tables in [`report/results.md`](report/results.md).

| Setting | Runs that reached the 99-pipe cap |
|---|---|
| Evolution, mutation ±0.15 (my settings) | 20 / 20 (median first generation: 7.5) |
| Random search (new random networks every generation, same budget) | 3 / 20 |
| Evolution, mutation ±0.05 | 11 / 20 |
| Evolution, mutation ±0.5 | 20 / 20 |
| Evolution, mutation ±1.0 | 20 / 20 |

---

The experiment harness and figures in `report/harness/` were written with AI assistance (Claude). The game, the neural network and the evolution code are my own.
