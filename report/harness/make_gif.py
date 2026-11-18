"""Optional demo GIF: train with Karthik's original seeds for 29 generations,
then record the first ~10 seconds of generation 30 (the trained flock) in his
own drawing style (black background, white pipes, coloured circle birds).

Changes to the copy (text substitution, game rules untouched):
  - HEADLESS = False so his draw() calls run, but SDL_VIDEODRIVER=dummy is set
    in the environment so no window opens
  - after pygame.display.update(), every 2nd frame of generation 30 is saved,
    and the program stops after 600 frames of generation 30

    python make_gif.py /path/to/flappy.py OUT.gif
"""
import os, subprocess, sys, tempfile

orig, out_gif = sys.argv[1], os.path.abspath(sys.argv[2])
src = open(orig).read()
cap = '''
        if generation == 30:   # harness: record
            if frames_this_gen % 2 == 0:
                _frames.append(pygame.image.tobytes(screen, "RGB"))
            if frames_this_gen >= 600:
                from PIL import Image as _I
                imgs = [_I.frombytes("RGB", (WIDTH, HEIGHT), b).resize((640, 360), _I.LANCZOS) for b in _frames]
                imgs = [im.quantize(colors=64, dither=_I.Dither.NONE) for im in imgs]
                imgs[0].save(OUT_GIF, save_all=True, append_images=imgs[1:], duration=33, loop=0, optimize=True)
                break
'''
subs = [
    ("HEADLESS = True ", "HEADLESS = False "),
    ("        pygame.display.update()\n", "        pygame.display.update()\n" + cap),
    ("pipes = [Pipe(rng_pipe)]\nbirds = [", f"_frames = []\nOUT_GIF = {out_gif!r}\npipes = [Pipe(rng_pipe)]\nbirds = ["),
]
for old, new in subs:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
d = tempfile.mkdtemp(dir="/tmp")
p = os.path.join(d, "gif_variant.py")
open(p, "w").write(src)
env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
subprocess.run([sys.executable, p], env=env, check=True, cwd=d)
print("wrote", out_gif, os.path.getsize(out_gif) // 1024, "KB")
