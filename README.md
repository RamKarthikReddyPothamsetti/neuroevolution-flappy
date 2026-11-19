My evolution-based flappy bird program is basically training training a neural network to make the bird cross through a target number of 99 pipes in the least amount of generations possible


demo

so basically the inputs for the nn are the height of the bird, its velocity, the distance to the nearest pipe to its right, the distance between that pipes gap-centre
for each generation, we sort by which weights enable the bird to stay alive the longest, and pick the top 5 birds' weights and use those ones to create mutated copies by adding noise to create a total of 50 new birds for the next generation
A generation ends when every bird is dead or after 9,000 frames (= 99 pipes, counted as "solved").

Run it
bash
pip install numpy pygame-ce
python flappy.py

Results

These results were evaluated over 20 random seeds, and I used 30 generations for each seed to calculate the median number of pipes crossed as well per generation. Full tables in report/results.md.

Setting	Runs that reached the 99-pipe cap
Evolution, mutation ±0.15 (my settings)	20 / 20 (median first generation: 7.5)
Random search (new random networks every generation, same budget)	3 / 20
Evolution, mutation ±0.05	11 / 20
Evolution, mutation ±0.5	20 / 20
Evolution, mutation ±1.0	20 / 20
