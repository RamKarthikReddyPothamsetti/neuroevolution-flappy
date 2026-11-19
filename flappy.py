import pygame
import numpy as np
from sys import exit
import os

WIDTH = 1280
HEIGHT = 720
PIPE_EVERY = 90
HEADLESS = True     
MAX_FRAMES = 9000 
MAX_GENERATIONS = 30  

if HEADLESS: os.environ["SDL_VIDEODRIVER"] = "dummy"  

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NeuroEvol Flappy Bird")
clock = pygame.time.Clock()

rng_bird = np.random.default_rng(1)
rng_pipe = np.random.default_rng(2)
rng_color = np.random.default_rng(3)

class NN:
    mutation_variation = 0.15

    def __init__(self, rng_obj, layers = None):
        if layers is None:
            weights_1 = rng_obj.uniform(-1, 1, (5, 6))
            weights_2 = rng_obj.uniform(-1, 1, (7, 1))
            layers = [weights_1, weights_2]
        self.layers = layers

    def forward_pass(self, inputs):
        outputs = inputs
        for layer in self.layers:
            inputs = np.hstack((np.ones(1), outputs))
            outputs = np.tanh(inputs @ layer)
        return outputs

    def mutate(self, rng_obj):
        new_layers = [layer + rng_obj.uniform(-NN.mutation_variation, NN.mutation_variation, layer.shape) for layer in self.layers]
        return NN(rng_obj, new_layers)


def next_generation(birds):
    ranked = sorted(birds, key = lambda b : b.score)
    parents = ranked[-5:]

    new_birds = [Bird(rng_bird, rng_color, p.nn) for p in parents]

    while len(new_birds) < 50:
        for parent in parents:
            child_nn = parent.nn.mutate(rng_bird)
            new_birds.append(Bird(rng_bird, rng_color, child_nn))

    return new_birds

class Bird:
    acceleration = 1
    spawn_x = 100
    spawn_y = 100
    radius = 25
    flap_velocity = 10

    def __init__(self, rng_bird, rng_color, nn = None):
        self.x = Bird.spawn_x
        self.y = Bird.spawn_y
        self.rect = pygame.Rect(self.x, self.y, 2 * Bird.radius, 2 * Bird.radius)
        self.rect.center = (self.x, self.y)
        self.velocity = 0
        self.is_alive = True
        self.color = (rng_color.integers(150, 256, size=1)[0], rng_color.integers(150, 256, size=1)[0], rng_color.integers(150, 256, size=1)[0])
        self.score = 0
        self.nn = nn if nn is not None else NN(rng_bird)

    def survive(self):
        if self.is_alive:
            self.y += self.velocity
            self.velocity += Bird.acceleration
            self.score += 1
        if self.y < 0 or self.y > HEIGHT:
            self.is_alive = False
        self.rect.center = (self.x, self.y)

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), Bird.radius)

    def flap(self):
        if self.is_alive:
            self.velocity = -Bird.flap_velocity

    def inputs(self, pipe):
        gap_center = pipe.top + Pipe.gap / 2
        return np.array([
            self.y / HEIGHT,
            self.velocity / Bird.flap_velocity,
            (pipe.rect_top.right - self.x) / WIDTH,
            (gap_center - self.y) / HEIGHT
        ])

    def decision(self, pipe):
        if self.nn.forward_pass(self.inputs(pipe))[0] > 0:
            self.flap()


class Pipe:
    width = 75
    margin = 150
    gap = 200
    speed = 10
    color = (255, 255, 255)

    def __init__(self, rng_pipe):
        self.top = rng_pipe.integers (Pipe.margin, HEIGHT - Pipe.margin - Pipe.gap + 1, size = 1)[0]
        self.bottom = HEIGHT - self.top - Pipe.gap
        self.rect_top = pygame.Rect(WIDTH, 0, Pipe.width, self.top)
        self.rect_bottom = pygame.Rect(WIDTH, HEIGHT - self.bottom, Pipe.width, self.bottom)

    def update_pos(self):
        self.rect_top.x -= Pipe.speed
        self.rect_bottom.x -= Pipe.speed

    def draw(self):
        pygame.draw.rect(screen, Pipe.color, self.rect_top)
        pygame.draw.rect(screen, Pipe.color, self.rect_bottom)


pipes = [Pipe(rng_pipe)]
birds = [Bird(rng_bird, rng_color) for i in range(50)]

generation = 1
frames_since_pipe = 0
frames_this_gen = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not HEADLESS: screen.fill((0, 0, 0))

    frames_since_pipe += 1
    frames_this_gen += 1
    if frames_since_pipe >= PIPE_EVERY:
        pipes.append(Pipe(rng_pipe))
        frames_since_pipe = 0

    for pipe in pipes:
            pipe.update_pos()
            if not HEADLESS: pipe.draw()
    pipes = [pipe for pipe in pipes if pipe.rect_top.right > 0]
    next_pipe = next((pipe for pipe in pipes if pipe.rect_top.right > Bird.spawn_x - Bird.radius), None)

    for bird in birds:
        if bird.is_alive:
            if next_pipe is not None:
                bird.decision(next_pipe)
                if (bird.rect.colliderect(next_pipe.rect_top) or bird.rect.colliderect(next_pipe.rect_bottom)):
                    bird.is_alive = False
            bird.survive()
            if not HEADLESS: bird.draw()

    if not any(bird.is_alive for bird in birds) or frames_this_gen >= MAX_FRAMES: 
        best = max(bird.score for bird in birds)
        median = np.median([bird.score for bird in birds])
        print(f"gen {generation}: best {best} median {median}")
        birds = next_generation(birds)
        pipes = [Pipe(rng_pipe)]
        frames_since_pipe = 0
        frames_this_gen = 0
        generation += 1
        if generation > MAX_GENERATIONS:
            break


    if not HEADLESS:
        pygame.display.update()
        clock.tick(600000)

pygame.quit()
