# Created by: Arena.ai Agent Mode (AI) - NightCity MTA:SA asset pipeline
# -----------------------------------------------------------------------------
# plan.py - the city plan: non-uniform street grid, districts, block recipes, expressways, river + bridges, sky bridges,
#           distant skyline.  Produces the model registry (deduplicated by specification) and the list of placements.
#
# City frame: origin = centre of the city, x east, y north, z = street level (0).  Everything is deterministic.
# -----------------------------------------------------------------------------
import numpy as np
from .ground import SW, CARR, SIDE, CURB, QUAY, RIVER_Z, BED_Z, cell_dims
from . import ground, bld, infra, terrain, tunnel as TUN
from .bldkit import POD_H

# ---------------------------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------------------------
XB = [56, 56, 56, 56, 88, 88, 128, 88, 72, 72, 72, 72]                 # block widths  (west -> east)
XC = ['S', 'N', 'N', 'N', 'A', 'S', 'A', 'A', 'S', 'S', 'A', 'S', 'S']  # 13 street lines (x = const)
YB = [88, 88, 88, 88, 88, 130, 72, 72, 128, 88, 72, 72]                # block depths  (south -> north), j = 5 is the river
YC = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'A', 'A', 'S', 'A', 'S', 'S']  # 13 street lines (y = const)
RIVER_J = 5
HERO = (6, 8)
# district letters, rows j = 11 (north) ... 0 (south)  (printed top = north)
DISTRICTS = [
    "RRRRRCCCEEEE",   # 11
    "RRMMQCCCEEEE",   # 10 (neighborhood park)
    "MMMMCCCCEEGE",   # 9
    "MMMMCPKCEEEE",   # 8
    "MMMMCCCCEEEE",   # 7
    "RMMWWWWWWWWW",   # 6
    "~~~~~~~~~~~~",   # 5  river
    "IIWWWWWWWWII",   # 4
    "IIIIIIIIIIII",   # 3
    "ITTFFIIYYFII",   # 2
    "ITTFFIIYYFII",   # 1
    "IIIIIIIIIIII",   # 0
]
DIST = {j: DISTRICTS[11 - j] for j in range(12)}

EXPRESS_EW = dict(line=8, level=12.0)         # y line 8 : east-west expressway, lower deck
EXPRESS_NS = dict(line=7, level=22.0)         # x line 7 : north-south expressway, upper deck (crosses over the lower one)
BRIDGE_LINES = {1: 'N', 4: 'A', 6: 'A', 8: 'S', 10: 'A'}   # x lines that cross the river (class used for the deck model)
HERO_BRIDGE_LINE = 6
# the road tunnel (cut-and-cover underpass): north - south under x line 9 (an 'S' street) NORTH of the river, from row 6 to row 9.  The road
# falls at 10 % through an open cut (64 m), enters the covered box tube right behind the portal header and runs flat 6.5 m below the
# street under three cross streets (two avenues and the lower expressway deck), then rises through the second open cut.  No water
# polygon lies above it: GTA treats everything below a water polygon as water, so the tunnel stays away from the river on purpose.
TUNNEL = dict(line=9, row_in=6, g=0.10, L_open=64.0, PT=1.2, flat_L=64.0, flat_n=5, variants=(0, 1, 2, 1, 0), start=15.0)

# The rideable Night City Metro: two vertically stacked, offset closed loops over the civic core.
# Level 1 dives through a lit west tunnel; level 2 is an open-air express 18 m above and 16 m north,
# with tall supports down to grade. Both loops use the native GTA train consist and call at MARKET,
# UNION and DOCKS, creating real inter-level service rather than a decorative second track.
METRO = dict(y=-5.0, ys=-55.0, z=13.0, x0=-468.0, x1=444.0, r=25.0, portal_x=-156.0,
             stops=(-84.0, 168.0, 396.0))
METRO2 = dict(y=11.0, ys=-39.0, z=31.0, x0=-468.0, x1=444.0, r=25.0,
              stops=(-84.0, 168.0, 396.0))

STYLE = {'C': 'core', 'K': 'core', 'P': 'core', 'E': 'ent', 'G': 'ent', 'M': 'market', 'R': 'res', 'W': 'ent', 'I': 'ind', 'T': 'ind', 'F': 'ind', 'Y': 'ind', 'Q': 'park', '~': 'ent'}
PAD = {'C': 'nc_plaza', 'K': 'nc_plaza', 'P': 'nc_plaza', 'E': 'nc_sidewalk', 'G': 'nc_sidewalk', 'M': 'nc_sidewalk', 'R': 'nc_sidewalk', 'W': 'nc_sidewalk', 'I': 'nc_concrete', 'T': 'nc_concrete', 'F': 'nc_concrete', 'Y': 'nc_concrete', 'Q': 'nc_grass'}

GLASSES = ['nc_glass_a', 'nc_glass_b', 'nc_glass_c', 'nc_glass_d', 'nc_glass_e', 'nc_glass_f', 'nc_glass_g', 'nc_glass_h']
LEDS = ['nc_strip_cyan', 'nc_strip_white', 'nc_light_magenta', 'nc_strip_cyan']


class Plan:
    def __init__(self):
        self.models = {}          # name -> dict(kind, builder, args, cat)
        self.keys = {}            # (builder, frozen args) -> name
        self.placements = []      # dict(model, x, y, z, rz, tag)
        self.water = []           # (x0, y0, x1, y1, z)
        self.river_water = []     # leading subset of water rectangles for validator / bank probes
        self.points = {}
        self.tour = []
        self.counts = {'g': 0, 'b': 0, 'i': 0, 's': 0}

    def model(self, cat, builder, args, hint='', name=None):
        key = (builder, tuple(sorted((k, v if not isinstance(v, (list, tuple)) else tuple(v)) for k, v in args.items())))
        if key in self.keys:
            return self.keys[key]
        self.counts[cat] += 1
        name = name or ('nc_%s%03d' % (cat, self.counts[cat]))
        self.models[name] = dict(cat=cat, builder=builder, args=dict(args), hint=hint)
        self.keys[key] = name
        return name

    def place(self, name, x, y, z, rz, tag):
        self.placements.append(dict(model=name, x=float(x), y=float(y), z=float(z), rz=float(rz), tag=tag))


def tunnel_geom(X, Y):
    """river tunnel in city coordinates -> dict(x, y_in, y_out, y_cov0, y_cov1, z_floor, parts=[...], holes=[(y0, y1) of the open cuts],
    prof=[(y, z) road level breakpoints]).  A part is dict(builder, args, y, z, rz)."""
    T = TUNNEL
    g, Lo, PT = T['g'], T['L_open'], T['PT']
    Lt = Lo + PT
    fl, fn = T['flat_L'], T['flat_n']
    x = X[T['line']]
    y_in = Y[T['row_in']] + T['start']           # behind the crosswalk of the street line in front of it
    z_floor = -g * Lt                            # road level at the portal header = flat road level of the tube
    parts = [dict(builder='tunnel_open', args=dict(L=Lo, pt=PT, g=g), y=y_in + Lt / 2, z=-g * Lt / 2, rz=0.0)]
    y = y_in + Lt
    y_a = y
    for k in range(fn):                          # covered box tube, flat
        parts.append(dict(builder='tunnel_tube', args=dict(L=fl, dz=0.0, variant=T['variants'][k % len(T['variants'])]), y=y + fl / 2, z=z_floor, rz=0.0))
        y += fl
    y_b = y
    parts.append(dict(builder='tunnel_open', args=dict(L=Lo, pt=PT, g=g), y=y_b + Lt / 2, z=-g * Lt / 2, rz=180.0))      # the open cut turned around
    y_out = y_b + Lt
    prof = [(y_in, 0.0), (y_a, z_floor), (y_b, z_floor), (y_out, 0.0)]
    return dict(x=x, y_in=y_in, y_out=y_out, y_cov0=y_a, y_cov1=y_b, y_a=y_a, y_b=y_b, z_floor=z_floor, parts=parts, prof=prof,
                holes=[(y_in, y_in + Lo), (y_b + PT, y_out)])


def tunnel_z(tg, y):
    """road level of the tunnel at world y"""
    ys = [p[0] for p in tg['prof']]
    zs = [p[1] for p in tg['prof']]
    return float(np.interp(y, ys, zs))


def tunnel_tour(tg):
    """camera keys that drive through the tunnel: eye x, y, z, look-at x, y, z, seconds to the next key (the eye follows the road profile)"""
    x = tg['x']

    def at(ye, yt, sec, h=1.7, ht=1.2):
        return (x, ye, tunnel_z(tg, ye) + h, x, yt, tunnel_z(tg, yt) + ht, sec)
    yi, yo, ya, yb = tg['y_in'], tg['y_out'], tg['y_a'], tg['y_b']
    return [
        (x, yi - 70.0, 7.0, x, yi + 40.0, tunnel_z(tg, yi + 40.0) + 0.5, 5.0),     # high over the approach, looking down into the cut
        at(yi + 25.0, yi + 125.0, 5.0),
        at(ya + 10.0, ya + 110.0, 8.0),                                            # through the mouth, down the tube
        at(ya + 130.0, ya + 230.0, 8.0),
        at(ya + 250.0, ya + 350.0, 6.0),
        at(yb - 40.0, yb + 60.0, 5.0),
        at(yb + 20.0, yo, 5.0),                                                    # up the second cut
        (x, yo + 8.0, 1.9, x, yo + 110.0, 2.6, 6.0),                               # out into the avenue
    ]


def grid_lines():
    W = [XB[i] + SW[XC[i]] / 2 + SW[XC[i + 1]] / 2 for i in range(12)]
    D = [YB[j] + SW[YC[j]] / 2 + SW[YC[j + 1]] / 2 for j in range(12)]
    X = [0.0]
    for w in W:
        X.append(X[-1] + w)
    Y = [0.0]
    for d in D:
        Y.append(Y[-1] + d)
    # centre the grid on the middle of the whole map
    cx = (X[0] + X[-1]) / 2
    cy = (Y[0] + Y[-1]) / 2
    X = [x - cx for x in X]
    Y = [y - cy for y in Y]
    return X, Y, W, D


# ---------------------------------------------------------------------------------------------
# palettes (fully specified building specs; the plan picks among them)
# ---------------------------------------------------------------------------------------------
def _palettes():
    rng = np.random.default_rng(2077)
    towers = []
    sizes = [(36, 36), (40, 36), (44, 38), (48, 40), (40, 44), (36, 44), (44, 44), (32, 32)]
    floors = [34, 40, 46, 52, 58, 64, 70, 78, 86, 96]
    for k in range(30):
        w, d = sizes[k % len(sizes)]
        f = floors[(k * 3 + int(rng.integers(0, 3))) % len(floors)]
        r = rng.random()
        mat = GLASSES[(k * 5 + int(rng.integers(0, 8))) % 8]
        led = LEDS[int(rng.integers(len(LEDS)))]
        if r < 0.58:
            towers.append(('tower_setback', dict(w=w, d=d, floors=f, mat=mat, seed=int(rng.integers(1, 9999)), crown_style=str(rng.choice(['spire', 'ring', 'prongs', 'flat', 'dish'])),
                                                 steps=int(rng.integers(1, 4)), scale_top=float(round(rng.uniform(0.55, 0.78), 2)), pod=int(rng.integers(1, 3)),
                                                 fin=int(rng.random() < 0.4), signs=int(rng.integers(2, 4)), led=led, octo=bool(rng.random() < 0.25))))
        elif r < 0.86:
            towers.append(('tower_cyl', dict(r=float(rng.choice([13, 14, 15, 16])), floors=f, mat=mat, seed=int(rng.integers(1, 9999)), ring_every=int(rng.choice([6, 8, 10])), led=led,
                                              crown_style=str(rng.choice(['ring', 'spire', 'flat'])))))
        else:
            towers.append(('tower_twin', dict(w=24, d=28, floors=min(f, 70), gap=float(rng.choice([16, 20])), mat=mat, seed=int(rng.integers(1, 9999)), led=led)))
    mids = []
    for k in range(18):
        w, d = [(36, 36), (40, 36), (32, 40), (44, 32), (36, 28)][k % 5]
        mids.append(('midrise', dict(w=w, d=d, floors=int(rng.integers(8, 20)), mat=GLASSES[(k * 3 + 1) % 8], seed=int(rng.integers(1, 9999)), signs=int(rng.integers(2, 4)),
                                     led=LEDS[int(rng.integers(len(LEDS)))])))
    tens = []
    for k in range(14):
        w, d = [(26, 22), (26, 24), (24, 22), (26, 26)][k % 4]
        tens.append(('tenement', dict(w=w, d=d, floors=int(rng.integers(6, 14)), mat=['nc_wall_tenement_a', 'nc_wall_tenement_b', 'nc_wall_brutal'][k % 3], seed=int(rng.integers(1, 9999)),
                                      escapes=int(rng.integers(1, 3)))))
    slabs = []
    for k in range(8):
        w, d = [(64, 18), (56, 18), (48, 16), (60, 20)][k % 4]
        slabs.append(('slab', dict(w=w, d=d, floors=int(rng.integers(14, 28)), mat=['nc_glass_b', 'nc_glass_g', 'nc_wall_brutal', 'nc_glass_h'][k % 4], seed=int(rng.integers(1, 9999)))))
    megas = [('megablock', dict(w=w, d=d, floors=int(f), seed=int(rng.integers(1, 9999)))) for (w, d, f) in ((48, 48, 30), (48, 44, 36), (52, 48, 32), (40, 40, 34))]
    warehouses = [('warehouse', dict(w=w, d=d, h=float(h), seed=int(rng.integers(1, 9999)))) for (w, d, h) in ((56, 36, 11), (60, 40, 12), (52, 34, 10), (64, 38, 13), (48, 32, 10))]
    return dict(towers=towers, mids=mids, tens=tens, slabs=slabs, megas=megas, warehouses=warehouses)


PAL = None


def pal():
    global PAL
    if PAL is None:
        PAL = _palettes()
    return PAL


def _footprint(arch, a):
    if arch == 'tower_setback':
        return a['w'] + 6, a['d'] + 6
    if arch == 'tower_cyl':
        return 2 * (a['r'] + 3.5), 2 * (a['r'] + 3.5)
    if arch == 'tower_twin':
        return 2 * a['w'] + a['gap'] + 6, a['d'] + 6
    if arch == 'arcology':
        return a['w'] + 8, a['w'] + 8
    if arch == 'plaza_gate':
        return a['w'], 9.0
    return a['w'], a['d']


def pick(items, bw, bd, target, rng, key=lambda it: it[1].get('floors', 0), margin=2.0, avoid=()):
    """choose the palette entry that fits (bw, bd) and whose size key is closest to `target`"""
    fit = [it for it in items if _footprint(it[0], it[1])[0] <= bw - margin and _footprint(it[0], it[1])[1] <= bd - margin]
    if not fit:
        fit = sorted(items, key=lambda it: _footprint(it[0], it[1])[0] * _footprint(it[0], it[1])[1])[:3]
    fit = [it for it in fit if (it[0], it[1].get('seed')) not in avoid] or fit
    fit.sort(key=lambda it: abs(key(it) - target))
    return fit[int(rng.integers(0, min(3, len(fit))))]


def _spline(anchors, spacing=12.0):
    """Catmull-Rom interpolation through geographic control points, sampled in metres."""
    A = np.asarray(anchors, dtype=np.float64)
    if len(A) < 2:
        return A.copy()
    padded = np.vstack([A[0], A, A[-1]])
    out = []
    for k in range(len(A) - 1):
        p0, p1, p2, p3 = padded[k:k + 4]
        chord = float(np.linalg.norm(p2[:2] - p1[:2]))
        n = max(2, int(np.ceil(chord / float(spacing))))
        t = np.arange(n, dtype=np.float64)[:, None] / n
        t2, t3 = t * t, t * t * t
        q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        out.append(q)
    out.append(A[-1:])
    return np.vstack(out)


def _add_curved_road(plan, road_paths, name, anchors, width=14.0, sidewalk=2.0,
                     kind='arterial', lamps=True, tag='road', spacing=12.0):
    xy = _spline(anchors, spacing=spacing)
    if xy.shape[1] > 2:
        xy = xy[:, :2]
    z = np.asarray(terrain.height(xy[:, 0], xy[:, 1]), dtype=np.float64) + 0.025
    path = np.column_stack([xy, z])
    road_paths.append(path)
    # Keep each DFF / COL comfortably inside RenderWare's signed local coordinate limits.
    stride = max(3, int(round(72.0 / spacing)))
    for k, start in enumerate(range(0, len(path) - 1, stride)):
        end = min(len(path) - 1, start + stride)
        chunk = path[start:end + 1]
        cx = float((chunk[:, 0].min() + chunk[:, 0].max()) / 2.0)
        cy = float((chunk[:, 1].min() + chunk[:, 1].max()) / 2.0)
        local = tuple(tuple(round(float(v), 3) for v in (p[0] - cx, p[1] - cy, p[2])) for p in chunk)
        args = dict(points=local, width=float(width), sidewalk=float(sidewalk), kind=kind,
                    seed=int(k), lamps=bool(lamps))
        model = plan.model('i', 'curved_road', args, hint='%s segment %d' % (name, k))
        plan.place(model, cx, cy, 0.0, 0.0, tag)
    return path


def _add_curved_highway(plan, name, anchors, width=20.0, spacing=12.0, tag='bridge'):
    sampled = _spline(anchors, spacing=spacing)
    xy = sampled[:, :2]
    deck = sampled[:, 2]
    ground_z = np.asarray(terrain.height(xy[:, 0], xy[:, 1]), dtype=np.float64)
    path = np.column_stack([xy, deck, ground_z])
    stride = max(3, int(round(72.0 / spacing)))
    for k, start in enumerate(range(0, len(path) - 1, stride)):
        end = min(len(path) - 1, start + stride)
        chunk = path[start:end + 1]
        cx = float((chunk[:, 0].min() + chunk[:, 0].max()) / 2.0)
        cy = float((chunk[:, 1].min() + chunk[:, 1].max()) / 2.0)
        local = tuple(tuple(round(float(v), 3) for v in (p[0] - cx, p[1] - cy, p[2], p[3])) for p in chunk)
        args = dict(points=local, width=float(width), support_spacing=42.0, seed=int(k))
        model = plan.model('i', 'curved_highway', args, hint='%s viaduct %d' % (name, k))
        plan.place(model, cx, cy, 0.0, 0.0, tag)
    return path


def _coastal_water(step=24.0):
    """Merge equal runs of offshore grid cells into a small, non-overlapping set of GTA water quads."""
    x0, y0, x1, y1 = terrain.WORLD
    step = float(step)
    nx, ny = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    xc = x0 + (np.arange(nx) + 0.5) * step
    active, rects = {}, []
    for j in range(ny):
        yc = y0 + (j + 0.5) * step
        # A half-cell tidal margin ensures every offshore point is covered even though
        # MTA water rectangles align to 24 m cells; at this contour the beach is still
        # below the -3 m water plane, so the shallow overlap does not flood visible land.
        offshore = np.asarray(terrain.land_distance(xc, yc)) < 12.0
        edges = np.diff(np.r_[False, offshore, False].astype(np.int8))
        runs = set(zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()))
        for run, first in list(active.items()):
            if run not in runs:
                rects.append((x0 + run[0] * step, y0 + first * step,
                              x0 + run[1] * step, y0 + j * step, terrain.SEA_Z))
                del active[run]
        for run in runs:
            active.setdefault(run, j)
    for run, first in active.items():
        rects.append((x0 + run[0] * step, y0 + first * step,
                      x0 + run[1] * step, y1, terrain.SEA_Z))
    # The 24 m cell grid begins on even world coordinates, as required by createWater().
    return [(int(round(a / 2) * 2), int(round(b / 2) * 2),
             int(round(c / 2) * 2), int(round(d / 2) * 2), float(z)) for a, b, c, d, z in rects if c > a and d > b]


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def build_plan():
    plan = Plan()
    X, Y, W, D = grid_lines()
    rng = np.random.default_rng(31337)
    P_ = pal()
    recent = []
    road_paths = []
    cx0 = X[HERO[0]] + W[HERO[0]] / 2
    cy0 = Y[HERO[1]] + D[HERO[1]] / 2
    plan.core_extent = (X[0], Y[0], X[-1], Y[-1])
    plan.extent = tuple(float(v) for v in terrain.WORLD)
    tg = tunnel_geom(X, Y)
    plan.tunnel = tg
    # the open cuts must lie inside the straight part of their street (between the crosswalks of the cross streets)
    for (hy0, hy1) in tg['holes']:
        jj = [j for j in range(12) if Y[j] < hy0 and hy1 < Y[j + 1]]
        assert len(jj) == 1, (hy0, hy1)
        j = jj[0]
        lo = Y[j] + CARR[YC[j]] / 2 + 4.0 + 1.0
        hi = Y[j + 1] - CARR[YC[j + 1]] / 2 - 4.0 - 1.0
        assert lo <= hy0 and hy1 <= hi, ('tunnel cut leaves its street', hy0, hy1, lo, hi)
    skyblocks = {}                 # (i, j) -> list of (x, y, half_w, half_d, top_z) for the sky bridge search
    for j in range(12):
        for i in range(12):
            ch = DIST[j][i]
            cls = (XC[i], XC[i + 1], YC[j], YC[j + 1])
            bw, bd = XB[i], YB[j]
            cxm = X[i] + W[i] / 2
            cym = Y[j] + D[j] / 2
            style = STYLE[ch]
            # block centre in world coordinates
            bcx = cxm + (SW[cls[0]] / 2 - SW[cls[1]] / 2) / 2
            bcy = cym + (SW[cls[2]] / 2 - SW[cls[3]] / 2) / 2
            alley = None
            if ch == 'M':
                alley = ('x', 0.0, 8.0)
            spec = dict(bw=bw, bd=bd, cls=cls, style=style, pad=PAD.get(ch, 'nc_sidewalk'), alley=alley, river=(ch == '~'), seed=1000 + 13 * bw + bd)
            if spec['river']:
                spec['seed'] = 3000 + bw
                gp = []
                if i in BRIDGE_LINES:                  # bridge on the west edge of this cell
                    gp.append((-W[i] / 2, -W[i] / 2 + SW[XC[i]] / 2))
                if (i + 1) in BRIDGE_LINES:            # ... and on the east edge
                    gp.append((W[i] / 2 - SW[XC[i + 1]] / 2, W[i] / 2))
                spec['gaps'] = tuple((float(a), float(b)) for a, b in gp)
            holes = []
            for (hy0, hy1) in tg['holes']:
                if Y[j] < hy0 and hy1 < Y[j + 1]:
                    if i + 1 == TUNNEL['line']:
                        holes.append(('e', round(hy0 - cym, 3), round(hy1 - cym, 3)))
                    if i == TUNNEL['line']:
                        holes.append(('w', round(hy0 - cym, 3), round(hy1 - cym, 3)))
            if holes:
                spec['holes'] = tuple(holes)
            gname = plan.model('g', 'cell', spec, hint='%s %dx%d' % (ch, bw, bd))
            plan.place(gname, cxm, cym, 0.0, 0.0, 'ground')
            if ch == '~':
                # river water: between the quay walls, even integer coordinates (GTA water grid)
                yw0 = cym + (-D[j] / 2 + CARR[cls[2]] / 2 + SIDE[cls[2]] + QUAY)
                yw1 = cym + (D[j] / 2 - CARR[cls[3]] / 2 - SIDE[cls[3]] - QUAY)
                river_rect = (int(round(X[i] / 2.0)) * 2, int(np.floor(yw0 / 2) * 2), int(round(X[i + 1] / 2.0)) * 2, int(np.ceil(yw1 / 2) * 2), RIVER_Z)
                plan.water.append(river_rect)   # neighbours share the same even grid line
                plan.river_water.append(river_rect)
                continue
            dist_c = np.hypot(bcx - cx0, bcy - cy0)
            rr = np.random.default_rng(int(rng.integers(1, 10**9)))
            blds = []                # (arch, args, x, y, rz) relative to the block centre
            if ch in 'CKP':
                if ch == 'K':
                    blds.append(('arcology', dict(w=110, floors=70, seed=77), 0.0, 0.0, 0.0))
                elif ch == 'P':
                    blds.append(('plaza_gate', dict(w=56, h=70.0, seed=5), 0.0, 0.0, 90.0))
                else:
                    target = 96 - dist_c / 14.0 + float(rr.uniform(-8, 8))
                    arch, a = pick(P_['towers'], bw, bd, target, rr, avoid=recent[-6:])
                    recent.append((arch, a.get('seed')))
                    fw, fd = _footprint(arch, a)
                    rz = 0.0 if (fw <= bw - 2 and fd <= bd - 2) else 90.0
                    blds.append((arch, a, 0.0, 0.0, rz))
            elif ch == 'E':
                r = rr.random()
                if r < 0.34:
                    arch, a = pick(P_['towers'], bw, bd, 40 + float(rr.uniform(-6, 16)), rr, avoid=recent[-6:])
                    recent.append((arch, a.get('seed')))
                    blds.append((arch, a, 0.0, 0.0, 0.0 if _footprint(arch, a)[0] <= bw - 2 and _footprint(arch, a)[1] <= bd - 2 else 90.0))
                elif r < 0.78:
                    arch, a = pick(P_['mids'], bw - 4, bd - 4, 14 + float(rr.uniform(-4, 5)), rr, avoid=recent[-4:])
                    recent.append((arch, a.get('seed')))
                    blds.append((arch, a, 0.0, 0.0, float(rr.choice([0, 90]))))
                else:
                    g = ('garage', dict(w=float(rr.choice([48, 52, 56])), d=float(rr.choice([32, 36])), levels=int(rr.integers(5, 9)), seed=int(rr.integers(1, 99))))
                    blds.append((g[0], g[1], 0.0, 0.0, float(rr.choice([0, 90]))))
            elif ch == 'G':
                blds.append(('garage', dict(w=56.0, d=40.0, levels=8, seed=3), 0.0, -bd * 0.2, 0.0))
                arch, a = pick(P_['mids'], bw - 4, bd * 0.4, 12, rr)
                blds.append((arch, a, 0.0, bd * 0.25, 0.0))
            elif ch == 'W':
                if rr.random() < 0.6:
                    arch, a = pick(P_['mids'], bw - 4, bd - 4, 10 + float(rr.uniform(-2, 8)), rr, avoid=recent[-4:])
                    recent.append((arch, a.get('seed')))
                    blds.append((arch, a, 0.0, 0.0, 0.0))
                else:
                    arch, a = pick(P_['slabs'], bw - 4, bd - 4, 18, rr)
                    blds.append((arch, a, 0.0, 0.0, 0.0 if bw >= bd else 90.0))
            elif ch == 'M':
                ds = (bd - 8.0) / 2.0
                for row in (-1, 1):
                    for col in (-1, 1):
                        arch, a = pick(P_['tens'], bw / 2 - 1, ds - 1.5, int(rr.integers(6, 14)), rr)
                        blds.append((arch, a, col * bw / 4.0, row * (4.0 + ds / 2.0), float(rr.choice([0, 90, 180, 270]))))
            elif ch == 'R':
                if rr.random() < 0.5:
                    arch, a = pick(P_['megas'], bw - 4, bd - 4, 32, rr, key=lambda it: it[1]['floors'])
                    blds.append((arch, a, 0.0, 0.0, 0.0))
                else:
                    arch, a = pick(P_['slabs'], bw - 6, bd * 0.45, 20, rr)
                    blds.append((arch, a, 0.0, -bd * 0.22, 0.0))
                    arch2, a2 = pick(P_['slabs'], bw - 6, bd * 0.45, 16, rr)
                    blds.append((arch2, a2, 0.0, bd * 0.24, 180.0))
            elif ch == 'I':
                if rr.random() < 0.65:
                    arch, a = pick(P_['warehouses'], bw - 4, bd - 4, 11, rr, key=lambda it: it[1]['h'])
                    blds.append((arch, a, 0.0, -bd * 0.18 if bd > 70 else 0.0, 0.0))
                    if bd >= 84:
                        arch, a = pick(P_['warehouses'], bw - 4, bd * 0.4, 10, rr, key=lambda it: it[1]['h'])
                        blds.append((arch, a, 0.0, bd * 0.27, 0.0))
                else:
                    blds.append(('container_yard', dict(w=float(min(bw - 4, 80)), d=float(min(bd - 4, 60)), seed=int(rr.integers(1, 99))), 0.0, 0.0, 0.0))
            elif ch == 'T':
                blds.append(('tank_farm', dict(w=float(bw - 6), d=float(min(bd - 6, 70)), seed=int(rr.integers(1, 99))), 0.0, 0.0, 0.0))
            elif ch == 'F':
                blds.append(('factory', dict(w=float(bw - 4), d=float(min(bd - 4, 70)), seed=int(rr.integers(1, 99))), 0.0, 0.0, 0.0))
            elif ch == 'Y':
                blds.append(('container_yard', dict(w=float(bw - 4), d=float(min(bd - 4, 64)), seed=int(rr.integers(1, 99))), 0.0, 0.0, 0.0))
            for arch, a, dx, dy, rz in blds:
                bname = plan.model('b', arch, a, hint=arch)
                plan.place(bname, bcx + dx, bcy + dy, CURB, rz, 'bld')
                fw, fd = _footprint(arch, a)
                if rz % 180 == 90:
                    fw, fd = fd, fw
                skyblocks.setdefault((i, j), []).append((bcx + dx, bcy + dy, fw / 2, fd / 2, arch, a))
    # ------------------------------------------------------------------ regional topography, shoreline and new metropolitan facilities
    # A continuous 384 m terrain quilt extends the compact historic street grid into a
    # real coastal region.  Each 24 m tile has a matching walkable triangle COL mesh.
    wx0, wy0, wx1, wy1 = terrain.WORLD
    terrain_step, terrain_size = 24.0, terrain.TILE
    nx_tiles = int(round((wx1 - wx0) / terrain_size))
    ny_tiles = int(round((wy1 - wy0) / terrain_size))
    terrain_holes = tuple((r[0], r[1], r[2], r[3]) for r in plan.river_water)
    terrain_holes += tuple((tg['x'] - 12.0, hy0, tg['x'] + 12.0, hy1) for hy0, hy1 in tg['holes'])
    for tj in range(ny_tiles):
        ty = float(wy0 + (tj + 0.5) * terrain_size)
        for ti in range(nx_tiles):
            tx = float(wx0 + (ti + 0.5) * terrain_size)
            a = dict(size=terrain_size, step=terrain_step, lod=False, cx=tx, cy=ty, holes=terrain_holes)
            m = plan.model('g', 'terrain_tile', a, hint='terrain tile %d,%d' % (ti, tj))
            plan.place(m, tx, ty, 0.0, 0.0, 'terrain')
            lod_args = dict(size=terrain_size, step=48.0, lod=True, cx=tx, cy=ty, holes=terrain_holes)
            lod = plan.model('g', 'terrain_tile', lod_args, hint='terrain LOD tile %d,%d' % (ti, tj))
            plan.place(lod, tx, ty, 0.0, 0.0, 'terrain_lod')

    # Hillside access loop: the road climbs in switchbacks instead of cutting through
    # the north-west ridge.  The lookout is set on the crest above the alluvial valley.
    ridge_path = _add_curved_road(plan, road_paths, 'ridge parkway',
        [(-600.0, 520.0), (-720.0, 610.0), (-860.0, 595.0), (-980.0, 710.0),
         (-870.0, 825.0), (-1040.0, 920.0), (-1190.0, 875.0), (-1360.0, 990.0),
         (-1320.0, 1110.0), (-1450.0, 1190.0), (-1320.0, 1270.0)],
        width=11.0, sidewalk=1.0, kind='mountain', lamps=False, spacing=10.0)
    lookout_z = float(terrain.height(-1280.0, 1240.0))
    lookout = plan.model('b', 'observation_tower', dict(height=74.0, seed=8), hint='north ridge lookout')
    plan.place(lookout, -1280.0, 1240.0, lookout_z, 0.0, 'landmark')
    coast_tower_z = float(terrain.height(1545.0, -430.0))
    coast_tower = plan.model('b', 'observation_tower', dict(height=44.0, seed=23), hint='east headland light')
    plan.place(coast_tower, 1545.0, -430.0, coast_tower_z, 0.0, 'landmark')

    # Regional stadium district on a deliberately graded plateau, with its public gate
    # tied into a modern roundabout and the existing east-side avenue.
    stadium = plan.model('b', 'stadium', dict(w=118.0, d=92.0, height=34.0, seed=17), hint='civic stadium')
    plan.place(stadium, 900.0, 300.0, 8.0, 0.0, 'landmark')
    roundabout = plan.model('i', 'roundabout', dict(radius=24.0, width=9.0, seed=4), hint='stadium circle')
    plan.place(roundabout, 760.0, 310.0, 8.0, 0.0, 'road')
    _add_curved_road(plan, road_paths, 'stadium boulevard',
        [(600.0, 310.0), (650.0, 310.0), (695.0, 310.0), (730.0, 310.0)],
        width=15.0, sidewalk=2.2, kind='arterial')
    _add_curved_road(plan, road_paths, 'stadium gate approach',
        [(785.0, 310.0), (820.0, 279.0), (855.0, 247.0), (900.0, 233.0)],
        width=13.0, sidewalk=1.8, kind='arterial')
    _add_curved_road(plan, road_paths, 'east neighborhood connector',
        [(760.0, 338.0), (795.0, 410.0), (900.0, 490.0), (1060.0, 530.0),
         (1230.0, 490.0), (1385.0, 370.0), (1480.0, 190.0), (1470.0, 20.0)],
        width=12.0, sidewalk=1.7, kind='arterial')

    # Airport: runway and aprons sit on a graded, flood-safe pad east of the city.
    terminal = plan.model('b', 'airport_terminal', dict(w=116.0, d=46.0, height=22.0, seed=31), hint='regional airport terminal')
    plan.place(terminal, 1030.0, -360.0, -0.9, 0.0, 'bld')
    apron = plan.model('i', 'airport_apron', dict(length=240.0, width=78.0, seed=3), hint='airport apron')
    plan.place(apron, 1030.0, -450.0, -0.9, 0.0, 'runway')
    for k in range(10):
        variant = 0 if k == 0 else (1 if k == 9 else 2)
        runway = plan.model('i', 'runway', dict(L=96.0, width=64.0, variant=variant), hint='airport runway section')
        plan.place(runway, 598.0 + 96.0 * k, -545.0, -0.9, 0.0, 'runway')
    _add_curved_road(plan, road_paths, 'airport boulevard',
        [(600.0, -430.0), (680.0, -430.0), (780.0, -420.0), (900.0, -420.0), (1030.0, -420.0)],
        width=16.0, sidewalk=2.2, kind='arterial')
    _add_curved_road(plan, road_paths, 'airport taxiway',
        [(600.0, -606.0), (800.0, -610.0), (1030.0, -606.0), (1260.0, -602.0), (1480.0, -595.0)],
        width=15.0, sidewalk=0.0, kind='industrial', lamps=False)
    _add_curved_road(plan, road_paths, 'airport runway connector',
        [(1140.0, -480.0), (1140.0, -510.0), (1145.0, -545.0), (1150.0, -580.0)],
        width=13.0, sidewalk=0.0, kind='industrial', lamps=False)
    _add_curved_road(plan, road_paths, 'airport east loop',
        [(1030.0, -420.0), (1220.0, -420.0), (1435.0, -418.0), (1535.0, -445.0),
         (1555.0, -490.0), (1550.0, -585.0), (1490.0, -685.0), (1370.0, -760.0),
         (1210.0, -800.0), (1040.0, -800.0), (930.0, -780.0)],
        width=12.0, sidewalk=1.5, kind='coastal')

    # A port-to-city coastal road skirts the inner southern shore and serves the
    # container terminal.  Its branch keeps heavy port traffic out of the town centre.
    _add_curved_road(plan, road_paths, 'south coast arterial',
        [(160.0, -690.0), (185.0, -760.0), (120.0, -850.0), (-40.0, -940.0),
         (-235.0, -1035.0), (-430.0, -1080.0), (-610.0, -1050.0), (-740.0, -950.0),
         (-780.0, -830.0), (-710.0, -730.0), (-600.0, -690.0)],
        width=15.0, sidewalk=2.0, kind='coastal')
    _add_curved_road(plan, road_paths, 'port access spur',
        [(-610.0, -1050.0), (-622.0, -1028.0), (-622.0, -990.0), (-622.0, -950.0),
         (-615.0, -916.0), (-580.0, -898.0), (-500.0, -894.0), (-420.0, -894.0),
         (-360.0, -910.0), (-325.0, -940.0), (-320.0, -968.0)],
        width=12.0, sidewalk=1.2, kind='industrial', lamps=False)
    _add_curved_road(plan, road_paths, 'wharf access spur',
        [(-430.0, -1080.0), (-418.0, -1066.0), (-410.0, -1052.0), (-410.0, -1040.0)],
        width=12.0, sidewalk=1.2, kind='industrial', lamps=False)
    _add_curved_road(plan, road_paths, 'airport to harbor link',
        [(930.0, -780.0), (790.0, -820.0), (620.0, -858.0), (450.0, -862.0),
         (300.0, -828.0), (185.0, -760.0)],
        width=13.0, sidewalk=1.4, kind='coastal')

    # Curved bay viaduct: a separate, genuinely elevated deck with support COL reaching
    # the tidal seabed and smooth approaches to both coastal roads.
    _add_curved_highway(plan, 'south bay causeway',
        [(230.0, -806.0, -0.7), (315.0, -831.0, -0.2), (390.0, -852.0, 3.5),
         (470.0, -864.0, 7.2), (560.0, -867.0, 8.0), (650.0, -858.0, 7.6),
         (735.0, -837.0, 5.4), (820.0, -810.0, 2.4), (910.0, -780.0, -0.6)],
        width=19.0, spacing=10.0, tag='bridge')

    # Port campus: container stacks, customs warehouse, two operational gantry cranes
    # and a long pile-supported wharf running beyond the natural shoreline.
    yard = plan.model('b', 'container_yard', dict(w=174.0, d=116.0, seed=44), hint='south harbor container yard')
    plan.place(yard, -520.0, -990.0, -0.8, 0.0, 'bld')
    customs = plan.model('b', 'warehouse', dict(w=88.0, d=46.0, h=15.0, seed=45), hint='port customs shed')
    plan.place(customs, -300.0, -1000.0, -0.8, 0.0, 'bld')
    pier = plan.model('i', 'pier', dict(length=360.0, width=18.0, deck=0.2, seed=6), hint='deep-water cargo pier')
    plan.place(pier, -410.0, -1220.0, -0.8, 0.0, 'bridge')
    for x, seed in ((-545.0, 11), (-330.0, 12)):
        crane = plan.model('i', 'port_crane', dict(height=46.0, span=54.0, depth=34.0, seed=seed), hint='harbor gantry crane')
        plan.place(crane, x, -1175.0, -0.8, 0.0, 'landmark')

    # Reusable opaque low-poly canopy clusters form belts of mixed coastal forest, not
    # an alpha billboard carpet.  Trees are kept clear of streets, terminals and airfields.
    terrain_placements = []
    for p in plan.placements:
        if p['tag'] == 'terrain':
            terrain_placements.append((p['x'], p['y']))
    road_xy = road_paths
    def near_roads(px, py, limit=30.0):
        for line in road_xy:
            aa, bb = line[:-1, :2], line[1:, :2]
            vv = bb - aa
            ll = np.sum(vv * vv, axis=1)
            tt = np.clip(np.sum((np.array([px, py]) - aa) * vv, axis=1) / np.maximum(ll, 1e-8), 0.0, 1.0)
            qq = aa + vv * tt[:, None]
            if float(np.min(np.sum((qq - np.array([px, py])) ** 2, axis=1))) < limit * limit:
                return True
        return False
    tree_rng = np.random.default_rng(88219)
    forest_groups = [
        (190, (-2140.0, -1020.0, 860.0, 1740.0), ('pine', 'broadleaf'), 95.0, 190.0),
        (95, (-2200.0, -650.0, -1250.0, 900.0), ('broadleaf', 'pine'), 45.0, 150.0),
        (90, (1280.0, -150.0, 1535.0, 1220.0), ('pine', 'broadleaf'), 25.0, 150.0),
        (45, (420.0, -1260.0, 1200.0, -760.0), ('palm', 'broadleaf'), 18.0, 85.0),
    ]
    tree_count = 0
    for count, (xa, ya, xb, yb), kinds, min_land, max_land in forest_groups:
        made, attempts = 0, 0
        while made < count and attempts < count * 45:
            attempts += 1
            tx = float(tree_rng.uniform(xa, xb))
            ty = float(tree_rng.uniform(ya, yb))
            dl = float(terrain.land_distance(tx, ty))
            if not (min_land <= dl <= max_land) or near_roads(tx, ty, 28.0):
                continue
            # Keep a clear urban bench, terminal approach, stadium campus, runway and port apron.
            if (abs(tx) < 630 and abs(ty) < 720) or (370 < tx < 1690 and -735 < ty < -265):
                continue
            if (-760 < tx < -70 and -1235 < ty < -825) or (650 < tx < 1150 and 110 < ty < 490):
                continue
            if any((tx - qx) ** 2 + (ty - qy) ** 2 < 19.0 ** 2 for qx, qy in terrain_placements[-tree_count - 20:]):
                continue
            # Avoid unusually steep ledges for tree roots; cliff faces remain bare rock.
            hz = [float(terrain.height(tx + dx, ty + dy)) for dx, dy in ((-8, 0), (8, 0), (0, -8), (0, 8))]
            if max(hz) - min(hz) > 18.0:
                continue
            kind = kinds[int(tree_rng.integers(len(kinds)))]
            if kind == 'palm':
                height = float(tree_rng.choice([9.0, 11.0, 13.0]))
            else:
                height = float(tree_rng.choice([7.0, 9.0, 11.0, 13.0, 15.0]))
            variant = int(tree_rng.integers(4))
            tree = plan.model('g', 'tree', dict(kind=kind, height=height, variant=variant), hint=kind + ' canopy')
            plan.place(tree, tx, ty, float(terrain.height(tx, ty)), 0.0, 'vegetation')
            terrain_placements.append((tx, ty))
            made += 1
            tree_count += 1

    # Coast water is generated after the narrow civic river so NC_RIVER_COUNT remains
    # a stable, testable leading subset.  24 m shoreline cells are vertically merged.
    plan.water.extend(_coastal_water(24.0))

    # ------------------------------------------------------------------ expressways
    ew, ns = EXPRESS_EW, EXPRESS_NS
    for i in range(12):
        L = int(round(W[i]))
        m = plan.model('i', 'expressway', dict(L=L, level=ew['level'], seed=i % 3, pier=True), hint='ew')
        plan.place(m, X[i] + W[i] / 2, Y[ew['line']], 0.0, 0.0, 'infra')
    for j in range(12):
        L = int(round(D[j]))
        if j == RIVER_J:
            m = plan.model('i', 'expressway', dict(L=L, level=ns['level'], seed=j % 3, pier=False, river=True), hint='ns-river')
        else:
            m = plan.model('i', 'expressway', dict(L=L, level=ns['level'], seed=j % 3, pier=True), hint='ns')
        plan.place(m, X[ns['line']], Y[j] + D[j] / 2, 0.0, 90.0, 'infra')
    # ------------------------------------------------------------------ bridges over the river
    j = RIVER_J
    span = int(round(D[j]))
    water_span = int(round(D[j] - (CARR[YC[j]] / 2 + SIDE[YC[j]] + QUAY) - (CARR[YC[j + 1]] / 2 + SIDE[YC[j + 1]] + QUAY)))
    for line, cls in BRIDGE_LINES.items():
        if line == HERO_BRIDGE_LINE:
            m = plan.model('i', 'bridge_cable', dict(span=span, water=water_span, cls='A', seed=3), hint='hero bridge')
        else:
            m = plan.model('i', 'bridge_girder', dict(span=span, water=water_span, cls=cls, seed=line), hint='bridge %s' % cls)
        plan.place(m, X[line], (Y[j] + Y[j + 1]) / 2, 0.0, 0.0, 'bridge')
    # ------------------------------------------------------------------ the river road tunnel
    for pt_ in tg['parts']:
        m = plan.model('i', pt_['builder'], pt_['args'], hint='tunnel')
        plan.place(m, tg['x'], pt_['y'], pt_['z'], pt_['rz'], 'tunnel')
    # ------------------------------------------------------------------ the Night City Metro (closed loop)
    MY, MYS, MZ = METRO['y'], METRO['ys'], METRO['z']
    x0m, x1m = METRO['x0'], METRO['x1']
    span = plan.model('i', 'rail_deck', dict(variant=0), hint='metro span')
    span_clear = plan.model('i', 'rail_deck', dict(variant=1), hint='metro span (bridge crossing)')
    bx = [X[line] for line in BRIDGE_LINES]                  # bridge decks: no pier beside them
    ns = int(round((x1m - x0m) / 48.0))
    for yy in (MY, MYS):                                     # north straight + south return
        for k in range(ns):
            cx = x0m + 24.0 + 48.0 * k
            near = min(min(abs(cx - 12.0 - b), abs(cx + 12.0 - b)) for b in bx)
            plan.place(span if near >= 18.0 else span_clear, cx, yy, MZ, 0.0, 'metro')
    curve = plan.model('i', 'rail_curve', dict(), hint='metro loop turn')
    plan.place(curve, x1m, (MY + MYS) / 2, MZ, 0.0, 'metro')       # east turn (bulges east)
    plan.place(curve, x0m, (MY + MYS) / 2, MZ, 180.0, 'metro')     # west turn (bulges west)
    tube = plan.model('i', 'rail_tube', dict(), hint='metro tube')
    nt = int(round((METRO['portal_x'] - x0m) / 24.0))
    for k in range(nt):
        plan.place(tube, x0m + 12.0 + 24.0 * k, MY, MZ, 0.0, 'metro')
    portal = plan.model('i', 'rail_portal', dict(), hint='metro portal')
    plan.place(portal, METRO['portal_x'], MY, MZ, 0.0, 'metro')
    for v, sx in enumerate(METRO['stops']):
        st = plan.model('i', 'rail_station', dict(variant=v), hint='metro station %d' % v)
        plan.place(st, sx, MY, MZ, 0.0, 'metro')
    sig = plan.model('i', 'rail_signal', dict(), hint='metro signal')
    for sx in (-126.0, -30.0, 228.0, 400.0):
        plan.place(sig, sx, MY + 3.2, MZ, 0.0, 'metro')
    plan.metro = METRO

    # The second, upper metro level is a distinct rideable loop. Its 18 m higher rail datum,
    # 16 m lateral offset and grade-reaching pylons keep both native train consists clear.
    M2Y, M2YS, M2Z = METRO2['y'], METRO2['ys'], METRO2['z']
    span_upper = plan.model('i', 'rail_deck', dict(variant=2), hint='upper metro span with tall piers')
    for yy in (M2Y, M2YS):
        for k in range(ns):
            cx = x0m + 24.0 + 48.0 * k
            near = min(min(abs(cx - 12.0 - b), abs(cx + 12.0 - b)) for b in bx)
            plan.place(span_clear if near < 18.0 else span_upper, cx, yy, M2Z, 0.0, 'metro')
    curve_upper = plan.model('i', 'rail_curve', dict(variant=2), hint='upper metro loop turn')
    plan.place(curve_upper, x1m, (M2Y + M2YS) / 2, M2Z, 0.0, 'metro')
    plan.place(curve_upper, x0m, (M2Y + M2YS) / 2, M2Z, 180.0, 'metro')
    for v, sx in enumerate(METRO2['stops']):
        st = plan.model('i', 'rail_station', dict(variant=v, support_floor=-31.0), hint='upper metro station %d' % v)
        plan.place(st, sx, M2Y, M2Z, 0.0, 'metro')
    for sx in (-126.0, -30.0, 228.0, 400.0):
        plan.place(sig, sx, M2Y + 3.2, M2Z, 0.0, 'metro')
    plan.metro2 = METRO2
    # ------------------------------------------------------------------ sky bridges between neighbouring core / entertainment towers
    sky = []
    for (i, j), lst in skyblocks.items():
        if (i + 1, j) in skyblocks and DIST[j][i] in 'CE' and DIST[j][i + 1] in 'CE':
            for (x0, y0, hw0, hd0, a0, s0) in lst:
                for (x1, y1, hw1, hd1, a1, s1) in skyblocks[(i + 1, j)]:
                    f0, f1 = s0.get('floors', 0), s1.get('floors', 0)
                    if a0.startswith('tower') and a1.startswith('tower') and min(f0, f1) >= 46 and abs(y0 - y1) < 6:
                        gap = (x1 - hw1) - (x0 + hw0)
                        if 8 < gap < 64:
                            sky.append(((x0 + hw0 - 1.0 + x1 - hw1 + 1.0) / 2, (y0 + y1) / 2, gap + 2.0, min(f0, f1)))
    sky.sort(key=lambda s: -s[3])
    used = 0
    for (sx, sy, L, f) in sky[:5]:
        z = 40.0 + 3.8 * int(f * 0.35)
        m = plan.model('i', 'skybridge', dict(length=int(round(L)), seed=used), hint='skybridge')
        plan.place(m, sx, sy, z, 0.0, 'sky')
        used += 1
    # ------------------------------------------------------------------ distant skyline strips (short chunks keep local vertices / collision sane)
    x0, y0, x1, y1 = plan.extent
    skyline_chunk = 420.0
    skyline_model = plan.model('s', 'skyline_strip', dict(L=int(skyline_chunk), seed=11, rows=2), hint='distant skyline')
    for k, cx in enumerate(np.arange(x0 + skyline_chunk / 2, x1, skyline_chunk)):
        m = skyline_model if k % 4 == 0 else plan.model('s', 'skyline_strip', dict(L=int(skyline_chunk), seed=11 + k % 4, rows=2), hint='distant skyline')
        plan.place(m, float(cx), y1 - 90.0, 0.0, 0.0, 'skyline')
        plan.place(m, float(cx), y0 + 90.0, 0.0, 180.0, 'skyline')
    for k, cy in enumerate(np.arange(y0 + skyline_chunk / 2, y1, skyline_chunk)):
        m = plan.model('s', 'skyline_strip', dict(L=int(skyline_chunk), seed=21 + k % 4, rows=2), hint='distant skyline')
        plan.place(m, x1 - 90.0, float(cy), 0.0, 90.0, 'skyline')
        plan.place(m, x0 + 90.0, float(cy), 0.0, -90.0, 'skyline')
    # ------------------------------------------------------------------ the sky dome (a camera-following sphere the sky shader draws on)
    dome = plan.model('g', 'skydome', dict(), hint='sky dome', name='nc_skydome')
    plan.place(dome, 0.0, 0.0, 350.0, 0.0, 'sky')
    # ------------------------------------------------------------------ named points: spawn, tour, viewpoints
    plan.points = dict(
        spawn=(cx0 - 0.0, cy0 - D[HERO[1]] / 2 + 12.0, CURB + 1.0),
        plaza=(cx0 - 94.0, cy0, CURB + 1.0),
        avenue=(X[4], cy0 - 200.0, 1.0),
        market=(X[1] + 4.0, Y[8] + 40.0, 1.0),
        quay=(cx0, Y[RIVER_J] + CARR[YC[RIVER_J]] / 2 + SIDE[YC[RIVER_J]] + 4.0, CURB + 1.0),
        bridge=(X[HERO_BRIDGE_LINE], (Y[RIVER_J] + Y[RIVER_J + 1]) / 2, CURB + 1.0),
        expressway=(X[3], Y[ew['line']], ew['level'] + 1.0),
        industrial=(X[8] + 60, Y[2] + 80, 1.0),
        metro_market=(METRO['stops'][0], METRO['y'] + 8.5, METRO['z'] + 1.0),
        metro_union=(METRO['stops'][1], METRO['y'] + 8.5, METRO['z'] + 1.0),
        metro_docks=(METRO['stops'][2], METRO['y'] + 8.5, METRO['z'] + 1.0),
        metro_upper_market=(METRO2['stops'][0], METRO2['y'] + 8.5, METRO2['z'] + 1.0),
        metro_upper_union=(METRO2['stops'][1], METRO2['y'] + 8.5, METRO2['z'] + 1.0),
        metro_upper_docks=(METRO2['stops'][2], METRO2['y'] + 8.5, METRO2['z'] + 1.0),
        metro_portal=(METRO['portal_x'] + 16.0, METRO['y'] - 10.0, METRO['z'] + 2.0),
        metro_loop=(METRO['x0'] - 12.0, (METRO['y'] + METRO['ys']) / 2, METRO['z'] + 2.0),
        metro_upper_loop=(METRO2['x0'] - 12.0, (METRO2['y'] + METRO2['ys']) / 2, METRO2['z'] + 2.0),
        tunnel_in=(tg['x'], tg['y_in'] - 14.0, 1.0),
        tunnel_out=(tg['x'], tg['y_out'] + 14.0, 1.0),
        tunnel_mid=(tg['x'], (tg['y_a'] + tg['y_b']) / 2, tg['z_floor'] + 1.0),
        harbor=(-520.0, -990.0, 1.0),
        pier=(-410.0, -1220.0, 1.0),
        airport=(1030.0, -360.0, 2.0),
        runway=(1030.0, -545.0, 2.0),
        stadium=(900.0, 233.0, 9.0),
        lookout=(-1280.0, 1240.0, lookout_z + 2.0),
        headland=(1545.0, -430.0, coast_tower_z + 2.0),
        causeway=(560.0, -867.0, 9.0),
    )
    yr = (Y[RIVER_J] + Y[RIVER_J + 1]) / 2
    hb = X[HERO_BRIDGE_LINE]
    plan.tour = [      # eye x, y, z, look-at x, y, z, seconds to the next key
        (X[10], Y[2] - 60.0, 300.0, cx0, cy0, 90.0, 14.0),                      # high aerial over the harbour side, the core in the distance
        (hb + 120.0, yr - 230.0, 70.0, hb, yr, 45.0, 12.0),                      # over the river towards the cable-stayed bridge
        (hb, yr - 70.0, 7.0, hb, yr + 330.0, 40.0, 14.0),                        # along the bridge deck, the core ahead
        (X[4], Y[7] + 20.0, 3.5, X[4], Y[10], 60.0, 12.0),                       # avenue canyon between the towers
        (cx0 - 140.0, cy0 - 120.0, 3.0, cx0, cy0, 150.0, 12.0),                  # plaza, looking up at the arcology
        (cx0 + 70.0, cy0 - 30.0, 40.0, cx0, cy0, 140.0, 10.0),                   # climb along the arcology
        (X[2], Y[ew['line']], ew['level'] + 3.0, X[9], Y[ew['line']], ew['level'] + 9.0, 14.0),   # along the lower expressway deck
        (X[1] + 3.0, Y[8] + 6.0, 3.0, X[1] + 3.0, Y[8] + 220.0, 14.0, 14.0),     # neon market lane
        (X[9] + 50.0, Y[2] + 30.0, 11.0, X[9] + 50.0, Y[3] + 140.0, 46.0, 12.0), # industrial district: stacks, tanks, flare
    ] + tunnel_tour(tg) + [                                                    # through the river tunnel
        (METRO['stops'][0] - 120.0, METRO['y'] + 26.0, METRO['z'] + 16.0, METRO['stops'][0], METRO['y'], METRO['z'] + 4.0, 12.0),
        # approach MARKET station along the viaduct
        (METRO['stops'][0] - 34.0, METRO['y'] + 9.0, METRO['z'] + 2.6, METRO['stops'][0] + 44.0, METRO['y'] + 8.0, METRO['z'] + 2.2, 11.0),
        # on the platform, looking down the line
        (METRO['portal_x'] - 150.0, METRO['y'], METRO['z'] + 2.2, METRO['portal_x'] + 10.0, METRO['y'], METRO['z'] + 2.6, 11.0),
        # through the lit tunnel tube towards the portal
        (X[3], Y[4] - 100.0, 140.0, cx0, cy0, 40.0, 10.0),                       # back up for a final look at the skyline
        (180.0, -780.0, 120.0, -520.0, -990.0, 18.0, 14.0),                    # the working cargo harbor
        (280.0, -840.0, 18.0, 560.0, -867.0, 8.0, 12.0),                       # south bay causeway and shoreline
        (-610.0, -1160.0, 42.0, -410.0, -1220.0, 18.0, 12.0),                   # cranes and deep-water pier
        (680.0, -500.0, 150.0, 1030.0, -380.0, 12.0, 14.0),                    # airport terminal approach
        (1200.0, -620.0, 86.0, 1030.0, -545.0, 0.0, 12.0),                      # runway threshold lights
        (700.0, 140.0, 72.0, 900.0, 300.0, 28.0, 14.0),                        # stadium bowl and civic campus
        (-1630.0, 780.0, 430.0, -1280.0, 1240.0, lookout_z + 10.0, 16.0),       # north ridge lookout
        (1900.0, 150.0, 250.0, 1545.0, -430.0, coast_tower_z + 10.0, 14.0),    # east headland
    ]
    plan.grid = dict(X=X, Y=Y, W=W, D=D)
    plan.hero = (cx0, cy0)
    return plan
