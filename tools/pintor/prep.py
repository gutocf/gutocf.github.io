import sys, glob, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PKG = sys.argv[1]; OUT = sys.argv[2]
TH = 140

def load(f):
    return np.array(Image.open(f).convert('L')).astype(np.uint8)

def exterior(lines):
    """non-line pixels connected to the image border"""
    lab, n = ndi.label(~lines)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return np.isin(lab, list(border))

def runs(row):
    r=[];i=0;n=len(row)
    while i<n:
        if row[i]:
            j=i
            while j<n and row[j]: j+=1
            r.append((i,j-1)); i=j
        else: i+=1
    return r

# ---------- heads ----------
heads = {}
for f in sorted(glob.glob(f'{PKG}/craques/*.jpeg')):
    g = load(f); L = g < TH
    ys, xs = np.where(L); yb = ys.max()
    r = runs(L[yb-20])
    nl, nr = r[0][0], r[-1][1]          # neck outer walls
    c = (nl+nr)//2; colr = runs(L[:, c]); chin = colr[-2][1]
    heads[os.path.basename(f)[:-5]] = dict(g=g, L=L, ext=exterior(L), yb=yb, nl=nl, nr=nr, chin=chin,
                                           top=ys.min(), x0=xs.min(), x1=xs.max())

# ---------- bodies ----------
CX = {'chutando': 701, 'comemorando': 704, 'goleiro': 704, 'parado': 704}
VTIP = {'chutando': 170, 'comemorando': 172, 'goleiro': 172, 'parado': 85}
bodies = {}
for f in sorted(glob.glob(f'{PKG}/corpos/*.jpeg')):
    name = os.path.basename(f)[:-5]
    g = load(f).copy(); L = g < TH
    ys, xs = np.where(L)
    # collar centre column = column of topmost line pixel near middle of body bbox
    cxb = (xs.min()+xs.max())//2
    win = L[:, cxb-150:cxb+150]; wy, wx = np.where(win)
    ytop = wy.min(); cx = CX[name]
    if name == 'parado':
        # remove neck stump: everything above the shoulder line inside stump columns
        r = runs(L[ytop+18, cx-60:cx+60])
        sl, sr = r[0][0]+cx-60, r[-1][1]+cx-60
        # shoulder join row: first row where pixels just outside the walls are dark
        y = ytop
        while not (L[y, sl-6] or L[y, sr+6]): y += 1
        cut = y + 2
        g[:cut, sl-4:sr+5] = 255
        L = g < TH
    ext = exterior(L)
    lab, _ = ndi.label(~L)
    # seed just above the V tip (thick front line) at the collar centre column
    col = L[:, cx]; y = VTIP[name]
    while not col[y]: y += 1
    seed = (y-4, cx)
    vint = lab == lab[seed]
    if name == 'parado':
        vint &= np.arange(L.shape[0])[:, None] >= cut
        vint[:, :cx-33] = False; vint[:, cx+34:] = False; vint[y:, :] = False
    # behind-the-neck area: in each column of the V interior, everything from the top down to it
    vy, vx = np.where(vint)
    hole = np.zeros_like(L)
    for x in range(vx.min(), vx.max()+1):
        c = np.where(vint[:, x])[0]
        if len(c): hole[:c.max()+1, x] = True
    # neck width target = width of V interior at its upper part
    rowsw = [vint[yy].sum() for yy in range(vy.min(), vy.max()+1)]
    hy, hx = np.where(hole)
    band = np.zeros_like(L)
    bodies[name] = dict(g=g, L=L, ext=ext, hole=hole, band=band,
                        hx0=hx.min(), hx1=hx.max(), hy0=hy.min(), hy1=vy.max(), cx=cx, vw=max(rowsw), ctop=(cut if name=='parado' else ytop))
    print(name, 'vint', vx.min(), vx.max(), vy.min(), vy.max(), 'vw', max(rowsw), file=sys.stderr)

NECK_RATIO = float(os.environ.get('NECK_RATIO', '1.0'))
DROP = int(os.environ.get('DROP', '4'))

NECKSHOW = float(os.environ.get('NECKSHOW', '0.12'))
def compose(hname, bname):
    H, B = heads[hname], bodies[bname]
    s = B['vw'] * NECK_RATIO / (H['nr'] - H['nl'])
    # chin goes a little above the collar top; neck is stretched to reach below the V tip
    chin_t = B['ctop'] - NECKSHOW * B['vw'] * NECK_RATIO
    need = (B['hy1'] + DROP - chin_t) / s           # source px chin->bottom needed
    E = max(0, int(np.ceil(need - (H['yb'] - H['chin']))))
    r = H['yb'] - 20
    g = np.concatenate([H['g'][:r], np.repeat(H['g'][r:r+1], E, 0), H['g'][r:]])
    ex = np.concatenate([H['ext'][:r], np.repeat(H['ext'][r:r+1], E, 0), H['ext'][r:]])
    hh, hw = g.shape
    gimg = Image.fromarray(g).resize((round(hw*s), round(hh*s)), Image.LANCZOS)
    ext = Image.fromarray((ex*255).astype(np.uint8)).resize(gimg.size, Image.BILINEAR)
    hg = np.array(gimg).astype(float); he = np.array(ext).astype(float)/255
    ha = np.where(he > 0.5, (255-hg)/255, 1.0)
    ncx = (H['nl']+H['nr'])/2*s
    ox = round(B['cx'] - ncx); oy = round(chin_t - H['chin']*s)
    bh, bw = B['g'].shape
    pad = max(0, -oy) + 20
    W = bw; Hh = bh + pad
    canvas = np.full((Hh, W), 255.0)
    canvas[pad:, :] = B['g']
    front = (~B['ext']) & (~B['hole']) & (~B['band'])
    frontp = np.zeros((Hh, W), bool); frontp[pad:] = front
    y0 = oy+pad; x0 = ox
    hh2, hw2 = hg.shape
    ys = slice(max(0,y0), min(Hh, y0+hh2)); xs = slice(max(0,x0), min(W, x0+hw2))
    sub_h = hg[ys.start-y0:ys.stop-y0, xs.start-x0:xs.stop-x0]
    sub_a = ha[ys.start-y0:ys.stop-y0, xs.start-x0:xs.stop-x0]
    a = np.where(frontp[ys, xs], 0, sub_a)
    canvas[ys, xs] = canvas[ys, xs]*(1-a) + sub_h*a
    return canvas.astype(np.uint8), pad

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for b in bodies:
        for h in heads:
            Image.fromarray(compose(h, b)[0]).save(f'{OUT}/{h}_{b}.png')
