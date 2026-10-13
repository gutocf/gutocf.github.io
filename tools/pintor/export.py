import sys, os, json, glob
sys.argv = [sys.argv[0], sys.argv[1], '/dev/null', sys.argv[2]]
os.environ.setdefault('NECK_RATIO', '1.6')
exec(open(os.path.join(os.path.dirname(__file__), 'prep.py')).read().split("if __name__")[0])
OUTD = sys.argv[3]; os.makedirs(OUTD, exist_ok=True)
MAXH = 900
meta = {}
for b, B in bodies.items():
    comps = {h: compose(h, b) for h in heads}
    padmax = max(p for _, p in comps.values())
    imgs = {}
    for h, (c, p) in comps.items():
        full = np.full((c.shape[0] + padmax - p, c.shape[1]), 255, np.uint8)
        full[padmax - p:] = c
        imgs[h] = full
    # common crop across heads
    dark = np.zeros_like(next(iter(imgs.values())), bool)
    for im in imgs.values(): dark |= im < 200
    ys, xs = np.where(dark); m = 24
    y0, y1, x0, x1 = max(0, ys.min()-m), ys.max()+m, max(0, xs.min()-m), xs.max()+m
    sc = min(1.0, MAXH / (y1 - y0))
    # torso seed: below the V tip
    vt = B['hy1'] + padmax
    col = B['L'][:, B['cx']]; y = B['hy1'] + 1
    while col[y]: y += 1                   # thick V line
    while not col[y]: y += 1               # collar band
    while col[y]: y += 1                   # thin V line
    seed_y = y + padmax + 40
    for h, im in imgs.items():
        crop = Image.fromarray(im[y0:y1, x0:x1])
        crop = crop.resize((round(crop.width*sc), round(crop.height*sc)), Image.LANCZOS)
        crop.save(f'{OUTD}/{h}_{b}.png', optimize=True)
        # leak check: face centre must not be connected to the border
        L = np.array(crop) < 160; ext = exterior(L)
        H = heads[h]
        fx = round((B['cx'] - x0) * sc); fy = round((seed_y - y0) * sc)
        if ext[fy, fx]: print('LEAK torso', h, b)
    meta[b] = dict(w=crop.width, h=crop.height,
                   seed=[round((B['cx'] - x0) * sc), round((seed_y - y0) * sc)])
# head thumbnails
for h, H in heads.items():
    ys, xs = np.where(H['L']); m = 12
    im = Image.fromarray(H['g'][max(0,ys.min()-m):ys.max()+m, max(0,xs.min()-m):xs.max()+m])
    im = im.resize((round(im.width*260/im.height), 260), Image.LANCZOS)
    im.save(f'{OUTD}/rosto_{h}.png', optimize=True)
# objects
for f in sorted(glob.glob(f'{PKG}/objetos/*.jpeg')):
    g = load(f); L = g < TH; ys, xs = np.where(L); m = 12
    im = Image.fromarray(g[max(0,ys.min()-m):ys.max()+m, max(0,xs.min()-m):xs.max()+m])
    k = 420 / max(im.size); im = im.resize((round(im.width*k), round(im.height*k)), Image.LANCZOS)
    im.save(f'{OUTD}/obj_{os.path.basename(f)[:-5]}.png', optimize=True)
print(json.dumps(meta))
