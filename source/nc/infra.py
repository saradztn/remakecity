# Created by: Arena.ai Agent Mode (AI) - NightCity MTA:SA asset pipeline
# -----------------------------------------------------------------------------
# infra.py - elevated expressways, river bridges (girder + cable stayed), glass sky bridges, distant skyline strips.
# Every function returns (Mesh, Col, meta).  Bridges / expressways are built along +x, centred on the origin.
# (No vehicles anywhere: only the road structures themselves.)
# -----------------------------------------------------------------------------
import numpy as np
from .mb import Mesh, Col, unit
from . import parts as P
from . import bldkit as K
from .bldkit import rect
from .ground import CURB, SW, CARR, SIDE, ROAD, RIVER_Z, BED_Z


def _cap_beam_pier(M, C, x, level, thick, ytop_w=18.0, base=0.0):
    h = level - thick
    M.box((x - 1.2, -1.2, base), (x + 1.2, 1.2, h - 1.8), 'nc_pillar', tile=(2.4, 6.0))
    M.box((x - 1.5, -ytop_w / 2, h - 1.8), (x + 1.5, ytop_w / 2, h), 'nc_concrete_dark', tile=(3, 3))
    M.box((x - 1.25, -1.25, base + 0.2), (x + 1.25, 1.25, base + 0.32), 'nc_strip_cyan', tile=(1, 1), emis=1.0)
    M.light((x, 0, base + 1.0), P.CYA, 0.7, 14.0)
    C.box((x - 1.2, -1.2, base), (x + 1.2, 1.2, h - 1.8))
    C.box((x - 1.5, -ytop_w / 2, h - 1.8), (x + 1.5, ytop_w / 2, h))


def expressway(L, level, seed=0, pier=True, river=False):
    """double carriageway elevated expressway segment, length L along x, deck top at z = level"""
    rng = np.random.default_rng(seed)
    M, C = Mesh(), Col()
    hw = 13.2
    th = 2.2
    for sy in (1, -1):
        M.ribbon([(-L / 2, sy * 6.6, level), (L / 2, sy * 6.6, level)], 6.0, 'nc_road_hwy', tile_v=8.0, u=(0.0, 1.0))
    # median + outer barriers (jersey profile as stepped boxes)
    for (ya, yb) in ((-0.6, 0.6),):
        M.box((-L / 2, ya, level), (L / 2, yb, level + 1.0), 'nc_concrete', tile=(4, 2))
    for sy in (1, -1):
        y0, y1 = (12.6, 13.2) if sy > 0 else (-13.2, -12.6)
        M.box((-L / 2, y0, level), (L / 2, y1, level + 0.95), 'nc_concrete', tile=(4, 2))
        ya, yb = (12.7, 13.1) if sy > 0 else (-13.1, -12.7)
        M.box((-L / 2, ya, level + 0.95), (L / 2, yb, level + 1.05), 'nc_strip_white', tile=(1, 1), emis=0.9)
    # box girder: underside + sides + LED line
    M.hquad(-L / 2, -hw, L / 2, hw, level - th, 'nc_deck_under', tile=(4, 4), up=False)
    M.wall_strip([(L / 2, hw), (-L / 2, hw)], level - th, level, 'nc_concrete', tile=(4, 2))
    M.wall_strip([(-L / 2, -hw), (L / 2, -hw)], level - th, level, 'nc_concrete', tile=(4, 2))
    for sy in (1, -1):
        y = sy * (hw + 0.02)
        M.box((-L / 2, min(y, y + sy * 0.12), level - th + 0.15), (L / 2, max(y, y + sy * 0.12), level - th + 0.35), 'nc_strip_cyan', tile=(1, 1), emis=1.0, skip=('-z',))
    # deck end faces
    # lamps on the median every ~24 m, twin arm
    nl = max(1, int(L // 24))
    for k in range(nl):
        x = -L / 2 + (k + 0.5) * L / nl
        M.merge(P.street_lamp(10.0, 4.2, False, True), (x, 0, level + 1.0), rz=90.0)
    if river:
        for fx in (-L / 6, L / 6):
            _cap_beam_pier(M, C, fx, level, th, base=BED_Z)
    elif pier:
        _cap_beam_pier(M, C, 0.0, level, th)
    C.box((-L / 2, -hw, level - th), (L / 2, hw, level))
    C.box((-L / 2, -0.6, level), (L / 2, 0.6, level + 1.0))
    C.box((-L / 2, 12.6, level), (L / 2, 13.2, level + 0.95))
    C.box((-L / 2, -13.2, level), (L / 2, -12.6, level + 0.95))
    return M, C, dict(kind='expressway', w=float(L), d=26.4, h=float(level + 12), dist=560.0)


def bridge_girder(span, cls, seed=0, water=None):
    """river bridge of road class `cls`, deck along +y over `span` (bank street centre to bank street centre), top at z = 0.
    The structure (parapets, lamps, girder, piers) exists over the water section only (`water` m, centred)."""
    M, C = Mesh(), Col()
    water = float(span if water is None else water)
    wd = SW[cls]
    car, side = CARR[cls], SIDE[cls]
    th = 1.8
    hw = wd / 2
    jn = CARR['S'] / 2                           # half carriageway of the bank street: this band is the intersection (plain asphalt, no raised pavement)
    ya, yb = -span / 2 + jn, span / 2 - jn
    if cls in ROAD:
        M.ribbon([(0, ya, 0.0), (0, yb, 0.0)], car / 2, ROAD[cls], tile_v=8.0, u=(0.0, 1.0))
    else:
        M.hquad(-car / 2, ya, car / 2, yb, 0.0, 'nc_alley', tile=(6, 6))
    M.hquad(-hw, -span / 2, hw, ya, 0.0, 'nc_asphalt', tile=(12.0, 12.0))
    M.hquad(-hw, yb, hw, span / 2, 0.0, 'nc_asphalt', tile=(12.0, 12.0))
    for sx in (1, -1):
        x0, x1 = (car / 2, hw) if sx > 0 else (-hw, -car / 2)
        M.hquad(x0, ya, x1, yb, CURB, 'nc_sidewalk', tile=(3.6, 3.6))
        xc = car / 2 * sx
        a, b = ((xc, yb), (xc, ya)) if sx > 0 else ((xc, ya), (xc, yb))
        M.wall(np.array(a), np.array(b), 0.0, CURB, 'nc_curb', tile=(1.6, 1.6))
        px = hw * sx
        M.box((min(px, px - sx * 0.4), -water / 2, CURB), (max(px, px - sx * 0.4), water / 2, CURB + 1.1), 'nc_concrete', tile=(4, 2))
        M.box((min(px - sx * 0.05, px - sx * 0.35), -water / 2, CURB + 1.1), (max(px - sx * 0.05, px - sx * 0.35), water / 2, CURB + 1.18), 'nc_strip_cyan', tile=(1, 1), emis=1.0)
        n = max(2, int(water // 26))
        for k in range(n):
            y = -water / 2 + (k + 0.5) * water / n
            M.merge(P.street_lamp(9.0, 2.6, False, False), (px - sx * 0.7, y, CURB), rz=(-90.0 if sx > 0 else 90.0) + 180.0)
    # girder: underside + outer faces (over the water only)
    M.hquad(-hw, -water / 2, hw, water / 2, -th, 'nc_deck_under', tile=(4, 4), up=False)
    M.wall_strip([(hw, -water / 2), (hw, water / 2)], -th, 0.0, 'nc_concrete_dark', tile=(4, 2))
    M.wall_strip([(-hw, water / 2), (-hw, -water / 2)], -th, 0.0, 'nc_concrete_dark', tile=(4, 2))
    for fy in (-1 / 6, 1 / 6):
        y = water * fy
        M.box((-hw * 0.7, y - 1.6, BED_Z), (hw * 0.7, y + 1.6, -th), 'nc_concrete_dark', tile=(3, 3))
        M.box((-hw * 0.7 - 0.02, y - 1.62, -th - 1.0), (hw * 0.7 + 0.02, y + 1.62, -th - 0.9), 'nc_strip_cyan', tile=(1, 1), emis=1.0)
        C.box((-hw * 0.7, y - 1.6, BED_Z), (hw * 0.7, y + 1.6, -th))
    C.box((-hw, -water / 2, -th), (hw, water / 2, 0.0))
    C.box((-hw, -span / 2, -0.5), (hw, span / 2, 0.0))
    for sx in (1, -1):                                    # the pavements stand 0.16 m above the road, the carriageway and the junction bands are at 0
        C.box((min(car / 2 * sx, hw * sx), ya, 0.0), (max(car / 2 * sx, hw * sx), yb, CURB))
    for sx in (1, -1):
        px = hw * sx
        C.box((min(px, px - sx * 0.4), -water / 2, CURB), (max(px, px - sx * 0.4), water / 2, CURB + 1.1))
    return M, C, dict(kind='bridge', w=float(wd), d=float(span), h=15.0, dist=560.0)


def bridge_cable(span, cls='A', seed=0, pylon_h=92.0, water=None):
    """hero cable-stayed bridge (H pylon at the middle of the span, harp cables, lit)"""
    M, C, meta = bridge_girder(span, cls, seed, water)
    hw = SW[cls] / 2
    rng = np.random.default_rng(seed + 1)
    ph = pylon_h
    for sx in (-1, 1):
        x = sx * (hw - 2.0)
        # tapered pylon leg
        M.loft(rect(3.2, 4.4, x, 0), CURB, rect(2.0, 3.0, x, 0), ph, 'nc_concrete', 'nc_concrete', tile=(4, 6))
        # LED edge lines up the leg
        for dy in (-1.5, 1.5):
            P.bar(M, (x + sx * 1.6, dy * 0.9, CURB + 0.5), (x + sx * 1.0, dy * 0.7, ph - 1), 0.14, 'nc_strip_cyan', emis=1.0)
        P.beacon_light(M, (x - 0.15, -0.15, ph))
        C.box((x - 1.6, -2.2, CURB), (x + 1.6, 2.2, ph * 0.5))
        C.box((x - 1.2, -1.8, ph * 0.5), (x + 1.2, 1.8, ph))
    for z in (ph * 0.45, ph * 0.8):
        M.box((-hw + 1.0, -1.4, z), (hw - 1.0, 1.4, z + 2.0), 'nc_concrete_dark', tile=(3, 3))
        C.box((-hw + 1.0, -1.4, z), (hw - 1.0, 1.4, z + 2.0))
    # harp of stay cables on both legs, both directions
    n = 9
    for sx in (-1, 1):
        x_leg, x_deck = sx * (hw - 2.0), sx * (hw - 0.6)
        for k in range(n):
            zt = ph * 0.52 + (ph * 0.44) * k / (n - 1)
            for sd in (-1, 1):
                yd = sd * (7.0 + 6.4 * k)
                if abs(yd) > (water or span) / 2 - 2:
                    continue
                M.tube(np.array([(x_leg, sd * 0.8, zt), (x_deck, yd, CURB + 1.2)]), 0.11, 4, 'nc_steel', tile=(4, 1), caps=False, emis=0.12, smooth=False)
    M.light((0, 0, ph * 0.6), P.COOL, 3.0, 80.0)
    meta.update(h=float(ph + 6), dist=800.0, hero=True)
    return M, C, meta


def skybridge(length, height=3.6, width=4.2, seed=0):
    """lit glass tube + steel truss between two buildings; along x, centred"""
    M, C = Mesh(), Col()
    rng = np.random.default_rng(seed)
    M.box((-length / 2, -width / 2, 0.0), (length / 2, width / 2, height), 'nc_glass_e', tile=(3.2 * 4, 3.8 * 8), emis=0.95, skip=('+x', '-x'), uvoff=(float(rng.integers(0, 4)) / 4, 0.0))
    M.box((-length / 2, -width / 2 - 0.3, -0.9), (length / 2, width / 2 + 0.3, 0.0), 'nc_metal_dark', tile=(2, 2))
    M.box((-length / 2, -width / 2 - 0.3, height), (length / 2, width / 2 + 0.3, height + 0.5), 'nc_metal_dark', tile=(2, 2))
    M.box((-length / 2, -width / 2 - 0.32, height + 0.5), (length / 2, -width / 2 - 0.2, height + 0.58), 'nc_strip_cyan', tile=(1, 1), emis=1.0)
    M.box((-length / 2, width / 2 + 0.2, height + 0.5), (length / 2, width / 2 + 0.32, height + 0.58), 'nc_strip_cyan', tile=(1, 1), emis=1.0)
    # truss diagonals under the tube
    n = max(2, int(length // 5))
    for k in range(n):
        x0 = -length / 2 + k * length / n
        x1 = x0 + length / n
        zz = -2.2
        P.bar(M, (x0, -width / 2 - 0.2, -0.9), (x0 + (x1 - x0) / 2, -width / 2 - 0.2, zz), 0.16, 'nc_steel')
        P.bar(M, (x0 + (x1 - x0) / 2, -width / 2 - 0.2, zz), (x1, -width / 2 - 0.2, -0.9), 0.16, 'nc_steel')
        P.bar(M, (x0, width / 2 + 0.2, -0.9), (x0 + (x1 - x0) / 2, width / 2 + 0.2, zz), 0.16, 'nc_steel')
        P.bar(M, (x0 + (x1 - x0) / 2, width / 2 + 0.2, zz), (x1, width / 2 + 0.2, -0.9), 0.16, 'nc_steel')
    M.light((0, 0, height * 0.6), P.COOL, 1.5, 28.0)
    C.box((-length / 2, -width / 2 - 0.3, -0.9), (length / 2, width / 2 + 0.3, height + 0.5))
    return M, C, dict(kind='skybridge', w=float(length), d=float(width), h=float(height + 3), dist=520.0)


def skyline_strip(L, seed, rows=2, hmin=70.0, hmax=250.0):
    """distant towers (no podiums, no collision): a long strip along +x, centred, depth ~ rows * 55 m"""
    rng = np.random.default_rng(seed)
    M, C = Mesh(), Col()
    mats = ['nc_glass_a', 'nc_glass_b', 'nc_glass_c', 'nc_glass_e', 'nc_glass_h', 'nc_glass_f', 'nc_glass_d']
    x = -L / 2
    while x < L / 2 - 20:
        w = float(rng.uniform(22, 46))
        for r in range(rows):
            d = float(rng.uniform(22, 40))
            h = float(rng.uniform(hmin, hmax)) * (1.0 + 0.25 * (r == 1))
            cy = r * 52.0 + float(rng.uniform(-6, 6))
            poly = rect(w, d, x + w / 2 + (r % 2) * 6, cy)
            m = mats[int(rng.integers(len(mats)))]
            K.tier(M, C, poly, 0.0, h, m, rng, emis=0.9, ledge_every=0, corners=False, col=False)
            if h > 160:
                P.beacon_light(M, (poly[:, 0].mean(), poly[:, 1].mean(), h))
            if rng.random() < 0.4:
                M.extrude(K.offset_poly(poly, 0.3), h - 0.7, h, 'nc_strip_cyan' if rng.random() < 0.6 else 'nc_strip_white', 'nc_strip_cyan', tile=(2, 1), emis=1.0)
        x += w + float(rng.uniform(2, 10))
    M.box((-L / 2 - 4, -30, -3.0), (L / 2 + 4, rows * 52.0 + 30, 0.0), 'nc_concrete_dark', tile=(8, 8))
    return M, C, dict(kind='skyline', w=float(L), d=float(rows * 52.0 + 60), h=float(hmax * 1.3 + 30), dist=1500.0, nocol=True)


# ---------------------------------------------------------------------------------------------
# Regional road hierarchy: grade-following collectors, elevated beltway, roundabouts, airport and port
# ---------------------------------------------------------------------------------------------
def _road_strip_vertices(path, half_width, offset=0.0, zoff=0.0):
    """Two edges and upward collision faces for a swept road / sidewalk strip."""
    Pth = np.asarray(path, dtype=np.float64)
    delta = Pth[1:, :2] - Pth[:-1, :2]
    tang = np.zeros((len(Pth), 2), dtype=np.float64)
    tang[:-1] += unit(delta)
    tang[1:] += unit(delta)
    tang = unit(tang)
    left = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    center = Pth[:, :2] + left * float(offset)
    z = Pth[:, 2] + float(zoff)
    L = np.column_stack([center + left * half_width, z])
    R = np.column_stack([center - left * half_width, z])
    verts = np.empty((len(Pth) * 2, 3), dtype=np.float64)
    verts[0::2], verts[1::2] = L, R
    faces = []
    for i in range(len(Pth) - 1):
        l0, r0, l1, r1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces.extend(((r0, r1, l1), (r0, l1, l0)))
    return verts, np.asarray(faces, dtype=np.int64), center, left


def _curb_quads(M, path, half_width, height=0.16):
    Pth = np.asarray(path, dtype=np.float64)
    for side in (-1.0, 1.0):
        v, f, center, left = _road_strip_vertices(Pth, half_width, 0.0, 0.0)
        edge = center + left * (half_width * side)
        Q = []
        for i in range(len(Pth) - 1):
            a, b = edge[i], edge[i + 1]
            za, zb = Pth[i, 2], Pth[i + 1, 2]
            Q.append([[a[0], a[1], za], [b[0], b[1], zb], [b[0], b[1], zb + height], [a[0], a[1], za + height]])
        if Q:
            M.quads(np.asarray(Q), 'nc_curb', emis=0.0)


def curved_road(points, width=12.0, sidewalk=2.6, kind='arterial', seed=0, lamps=True):
    """Paved, collision-backed terrain road around a sampled (x,y,z) spline."""
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 3 or len(path) < 2:
        raise ValueError('curved_road requires at least two (x,y,z) points')
    half = float(width) / 2.0
    roadmat = 'nc_road_hwy' if kind == 'expressway' else ('nc_road_ave' if width >= 14 else 'nc_road_str')
    M, C = Mesh(), Col()
    M.ribbon(path[:, :3], half, roadmat, tile_v=8.0, u=(0.0, 1.0))
    V, F, center, left = _road_strip_vertices(path[:, :3], half, 0.0, 0.0)
    C.mesh(V, F, 0)
    # Shoulders and walkable sidewalks follow the same terrain profile as the carriageway.
    for side in (-1.0, 1.0):
        shoulder = path[:, :3].copy()
        shoulder[:, :2] += left * side * (half + float(sidewalk) / 2.0)
        shoulder[:, 2] += 0.16
        M.ribbon(shoulder, float(sidewalk) / 2.0, 'nc_sidewalk', tile_v=4.0, u=(0.0, 1.0))
        SV, SF, _, _ = _road_strip_vertices(path[:, :3], float(sidewalk) / 2.0, side * (half + float(sidewalk) / 2.0), 0.16)
        C.mesh(SV, SF, 0)
    _curb_quads(M, path[:, :3], half, 0.16)
    # A subdued continuous centre guide is visible on long curves without looking like a freeway stripe.
    guide = path[:, :3].copy()
    guide[:, 2] += 0.025
    M.ribbon(guide, 0.07, 'nc_strip_white', tile_v=8.0, u=(0.49, 0.51), emis=0.0)
    # Drainage slots every 28 m and lamps at a human-scaled spacing.
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1))])
    for target in np.arange(14.0, cumulative[-1], 28.0):
        idx = min(len(path) - 2, int(np.searchsorted(cumulative, target, side='right') - 1))
        t = (target - cumulative[idx]) / max(cumulative[idx + 1] - cumulative[idx], 1e-6)
        p = path[idx, :3] * (1.0 - t) + path[idx + 1, :3] * t
        M.box((p[0] - 0.32, p[1] - 0.10, p[2] + 0.015), (p[0] + 0.32, p[1] + 0.10, p[2] + 0.035), 'nc_metal_dark', tile=(0.7, 0.2))
    if lamps:
        spacing = 42.0 if kind == 'coastal' else 54.0
        for target in np.arange(spacing / 2.0, cumulative[-1], spacing):
            idx = min(len(path) - 2, int(np.searchsorted(cumulative, target, side='right') - 1))
            t = (target - cumulative[idx]) / max(cumulative[idx + 1] - cumulative[idx], 1e-6)
            p = path[idx, :3] * (1.0 - t) + path[idx + 1, :3] * t
            tg = path[min(idx + 1, len(path) - 1), :2] - path[max(idx - 1, 0), :2]
            tang = tg / max(float(np.linalg.norm(tg)), 1e-8)
            leftn = np.array([-tang[1], tang[0]])
            side = 1.0 if int(target // spacing) % 2 else -1.0
            pp = p.copy()
            pp[:2] += leftn * side * (half + sidewalk + 1.0)
            M.merge(P.street_lamp(8.0, 2.1, kind == 'industrial', False), pp, rz=float(np.degrees(np.arctan2(-leftn[0], leftn[1]))))
    return M, C, dict(kind='curved_road', w=float(width + 2 * sidewalk), d=float(cumulative[-1]), h=float(np.ptp(path[:, 2]) + 9), dist=820.0, roadclass=kind)


def curved_highway(points, width=18.0, support_spacing=42.0, seed=0):
    """Elevated multi-lane viaduct chunk with real deck collision and terrain-reaching piers.

    Each point is (x,y,deck_z,ground_z) in the model-local x/y frame. Chunks are kept short
    by the planner, so neither the DFF nor its COL exceeds RenderWare's local bounds.
    """
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 4 or len(raw) < 2:
        raise ValueError('curved_highway requires (x,y,deck_z,ground_z) samples')
    path = raw[:, :3]
    half = float(width) / 2.0
    M, C = Mesh(), Col()
    M.ribbon(path, half, 'nc_road_hwy', tile_v=9.0, u=(0.0, 1.0))
    M.ribbon(path - np.array([0.0, 0.0, 2.2]), half + 0.8, 'nc_deck_under', tile_v=6.0, u=(0.0, 1.0), up=False)
    V, F, center, left = _road_strip_vertices(path, half, 0.0, 0.0)
    C.mesh(V, F, 0)
    for side in (-1.0, 1.0):
        edge = center + left * (half * side)
        Q = []
        for i in range(len(path) - 1):
            a, b = edge[i], edge[i + 1]
            za, zb = path[i, 2], path[i + 1, 2]
            Q.append([[a[0], a[1], za], [b[0], b[1], zb], [b[0], b[1], zb + 1.05], [a[0], a[1], za + 1.05]])
        if Q:
            M.quads(np.asarray(Q), 'nc_concrete', emis=0.0)
            # The road barriers are low enough for traffic flow but have continuous collision.
            for q in Q:
                C.mesh(np.asarray(q), np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64), 0)
    # White edge reflectors and a muted median guide.
    for side in (-1.0, 1.0):
        edge = path.copy()
        edge[:, :2] += left * side * (half - 0.65)
        edge[:, 2] += 0.035
        M.ribbon(edge, 0.055, 'nc_strip_white', tile_v=8.0, u=(0.0, 1.0))
    mid = path.copy(); mid[:, 2] += 0.03
    M.ribbon(mid, 0.06, 'nc_strip_white', tile_v=8.0, u=(0.49, 0.51))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for target in np.arange(float(support_spacing) * 0.5, cumulative[-1] - 2.0, float(support_spacing)):
        idx = min(len(path) - 2, int(np.searchsorted(cumulative, target, side='right') - 1))
        t = (target - cumulative[idx]) / max(cumulative[idx + 1] - cumulative[idx], 1e-6)
        p = path[idx] * (1.0 - t) + path[idx + 1] * t
        g = raw[idx, 3] * (1.0 - t) + raw[idx + 1, 3] * t
        bottom = float(g)
        if p[2] - bottom < 3.0:
            continue
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        M.box((x - 1.15, y - 1.15, bottom), (x + 1.15, y + 1.15, z - 2.1), 'nc_pillar', tile=(2.4, 4.0))
        M.box((x - 2.0, y - 2.0, z - 2.1), (x + 2.0, y + 2.0, z - 1.8), 'nc_concrete_dark', tile=(3, 3))
        C.box((x - 1.2, y - 1.2, bottom), (x + 1.2, y + 1.2, z - 1.8))
    for target in np.arange(28.0, cumulative[-1], 56.0):
        idx = min(len(path) - 2, int(np.searchsorted(cumulative, target, side='right') - 1))
        t = (target - cumulative[idx]) / max(cumulative[idx + 1] - cumulative[idx], 1e-6)
        p = path[idx] * (1.0 - t) + path[idx + 1] * t
        M.merge(P.street_lamp(9.0, 3.2, False, True), (p[0], p[1], p[2] + 1.05), rz=90.0)
    return M, C, dict(kind='curved_highway', w=float(width), d=float(cumulative[-1]), h=float(np.max(path[:, 2])), dist=1200.0)


def roundabout(radius=24.0, width=9.0, seed=0):
    """Circular two-lane junction ring with a planted, collision-safe central island."""
    M, C = Mesh(), Col()
    n = 48
    th = np.arange(n + 1) * (2.0 * np.pi / n)
    ro, ri = float(radius) + float(width) / 2.0, float(radius) - float(width) / 2.0
    outer = np.column_stack([ro * np.cos(th), ro * np.sin(th)])
    inner = np.column_stack([ri * np.cos(th), ri * np.sin(th)])
    Q = np.zeros((n, 4, 3), dtype=np.float64)
    for k in range(n):
        Q[k] = [[outer[k, 0], outer[k, 1], 0.0], [outer[k + 1, 0], outer[k + 1, 1], 0.0],
                [inner[k + 1, 0], inner[k + 1, 1], 0.0], [inner[k, 0], inner[k, 1], 0.0]]
    M.quads(Q, 'nc_road_ave')
    verts = np.column_stack([np.vstack([outer[:-1], inner[:-1]]), np.zeros(2 * n)])
    faces = []
    for k in range(n):
        j = (k + 1) % n
        faces.extend(((k, j, n + j), (k, n + j, n + k)))
    C.mesh(verts, np.asarray(faces, dtype=np.int64), 0)
    M.cyl((0.0, 0.0), ri * 0.92, 0.0, 0.25, 24, 'nc_grass', tile=(6, 6), top_mat='nc_grass')
    C.box((-ri * 0.92, -ri * 0.92, 0.0), (ri * 0.92, ri * 0.92, 0.25))
    M.cyl((0.0, 0.0), 2.0, 0.25, 2.6, 12, 'nc_concrete_dark', tile=(2, 2), top_mat='nc_sidewalk')
    M.cyl((0.0, 0.0), 1.2, 2.6, 2.9, 12, 'nc_strip_cyan', tile=(2, 2), top_mat='nc_strip_cyan', emis=0.8)
    for k in range(8):
        a = 2 * np.pi * k / 8
        x, y = np.cos(a) * (radius + 1.0), np.sin(a) * (radius + 1.0)
        M.merge(P.bollard(), (x, y, 0.0), rz=float(np.degrees(a)))
    return M, C, dict(kind='roundabout', w=float(2 * ro), d=float(2 * ro), h=3.0, dist=360.0)


def runway(L=96.0, width=64.0, variant=0):
    """Straight airport runway section; neighbouring sections share exact, collision-safe seams."""
    L, width = float(L), float(width)
    M, C = Mesh(), Col()
    M.hquad(-L / 2, -width / 2, L / 2, width / 2, 0.0, 'nc_asphalt', tile=(12.0, 12.0))
    C.box((-L / 2, -width / 2, -0.35), (L / 2, width / 2, 0.0))
    # Painted runway centreline dashes and edge shoulders.
    M.box((-L / 2, -width / 2 + 1.2, 0.012), (L / 2, -width / 2 + 1.45, 0.024), 'nc_strip_white', tile=(4, 1))
    M.box((-L / 2, width / 2 - 1.45, 0.012), (L / 2, width / 2 - 1.2, 0.024), 'nc_strip_white', tile=(4, 1))
    for x in np.arange(-L / 2 + 4.0, L / 2 - 2.0, 12.0):
        M.box((x, -0.20, 0.025), (min(x + 6.0, L / 2 - 1.0), 0.20, 0.040), 'nc_strip_white', tile=(4, 1))
    # Threshold piano-key bars on the first / last sections of the runway.
    if int(variant) in (0, 1):
        side = -1 if int(variant) == 0 else 1
        x = -L / 2 + 5.0 if side < 0 else L / 2 - 9.0
        for k in range(8):
            y = -width * 0.34 + k * width * 0.095
            M.box((x, y, 0.026), (x + 4.5, y + 0.55, 0.04), 'nc_strip_white', tile=(2, 1))
    for x in (-L / 2 + 2.0, L / 2 - 2.0):
        for y in (-width / 2 + 0.5, width / 2 - 0.5):
            M.cyl((x, y), 0.22, 0.02, 0.18, 8, 'nc_light_warm', tile=(1, 1), top_mat='nc_light_warm', emis=0.8)
    return M, C, dict(kind='runway', w=L, d=width, h=0.2, dist=1500.0)


def pier(length=220.0, width=18.0, deck=0.2, seed=0):
    """Piled harbor pier with a collision-backed walking deck, fenders and low lighting."""
    L, W, z = float(length), float(width), float(deck)
    M, C = Mesh(), Col()
    M.box((-W / 2, -L / 2, z - 0.65), (W / 2, L / 2, z), 'nc_concrete_dark', tile=(4, 4))
    M.hquad(-W / 2, -L / 2, W / 2, L / 2, z + 0.02, 'nc_sidewalk', tile=(4, 4))
    C.box((-W / 2, -L / 2, z - 0.65), (W / 2, L / 2, z + 0.04))
    for y in np.arange(-L / 2 + 12.0, L / 2 - 5.0, 28.0):
        for sx in (-1.0, 1.0):
            x = sx * (W / 2 - 1.8)
            M.cyl((x, y), 0.8, -10.0, z - 0.6, 10, 'nc_concrete_dark', tile=(2, 4), top_mat='nc_concrete_dark')
            C.box((x - 0.8, y - 0.8, -10.0), (x + 0.8, y + 0.8, z - 0.55))
            M.merge(P.bollard(), (sx * (W / 2 - 1.1), y, z + 0.03), rz=0.0)
        if int(y // 28) % 2 == 0:
            M.merge(P.street_lamp(6.5, 1.4, True, False), (0.0, y, z + 0.02), rz=90.0)
    for sx in (-1.0, 1.0):
        M.box((sx * (W / 2 - 0.28) - 0.18, -L / 2, z + 0.6), (sx * (W / 2 - 0.28) + 0.18, L / 2, z + 0.72), 'nc_steel', tile=(3, 1))
    return M, C, dict(kind='pier', w=W, d=L, h=11.0, dist=650.0)


def port_crane(height=46.0, span=54.0, depth=34.0, seed=0):
    """Ship-to-shore gantry crane with an open trolley, twin legs and walkable maintenance deck."""
    rng = np.random.default_rng(seed)
    H, W, D = float(height), float(span), float(depth)
    M, C = Mesh(), Col()
    legw = 1.7
    # four tapered portal legs leave a clear vehicle / vessel passage below the boom.
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x0, y0 = sx * (W / 2 - 4.2), sy * (D / 2 - 3.0)
            x1, y1 = sx * (W / 2 - 8.0), sy * (D / 2 - 2.0)
            M.tube(np.asarray([[x0, y0, 0.0], [x1, y1, H - 9.0]]), legw * 0.58, 6, 'nc_steel', tile=(5, 3), caps=False, smooth=False)
            M.box((x0 - 2.4, y0 - 2.4, 0.0), (x0 + 2.4, y0 + 2.4, 0.75), 'nc_concrete_dark', tile=(3, 2))
            C.box((x0 - 2.4, y0 - 2.4, 0.0), (x0 + 2.4, y0 + 2.4, 1.2))
    # runway rails and deep cross-girders.
    for sy in (-1.0, 1.0):
        y = sy * (D / 2 - 3.0)
        M.box((-W / 2, y - 1.2, H - 10.0), (W / 2, y + 1.2, H - 7.8), 'nc_metal_dark', tile=(4, 2))
        M.box((-W / 2, y - 0.22, H - 7.8), (W / 2, y + 0.22, H - 7.55), 'nc_light_amber', tile=(3, 1), emis=0.9)
    M.box((-W / 2, -D / 2, H - 8.0), (W / 2, D / 2, H - 5.0), 'nc_steel', tile=(5, 3))
    # trussed main beam and cantilevered sea-boom pointing toward local +y.
    M.box((-W / 2, -3.0, H - 5.0), (W / 2, 3.0, H), 'nc_metal_dark', tile=(4, 4))
    M.box((-W / 2, 1.0, H - 1.2), (W / 2 + 18.0, 4.0, H - 0.65), 'nc_steel', tile=(4, 2))
    for x in np.arange(-W / 2 + 4.0, W / 2, 8.0):
        M.tube(np.asarray([[x, -2.0, H - 5.2], [x + 4.0, 2.0, H - 1.0],
                          [x + 8.0, -2.0, H - 5.2]]), 0.32, 5, 'nc_steel', tile=(4, 1), caps=False, smooth=False)
    # trolley, twin hoist cables, yellow spreader and corner beacons.
    tx = float(rng.uniform(-W * 0.18, W * 0.18))
    M.box((tx - 5.0, -5.0, H - 4.9), (tx + 5.0, 5.0, H - 2.4), 'nc_concrete_dark', tile=(3, 2))
    for sx in (-1.0, 1.0):
        M.box((tx + sx * 3.6 - 0.38, -4.5, H - 2.4), (tx + sx * 3.6 + 0.38, 4.5, H - 1.0), 'nc_metal_light', tile=(1, 2))
        M.tube(np.asarray([[tx + sx * 3.6, 0.0, H - 1.0], [tx + sx * 3.6, 0.0, 14.0]]), 0.13, 5, 'nc_steel', tile=(3, 1), caps=False)
    M.box((tx - 8.0, -7.0, 13.2), (tx + 8.0, 7.0, 14.0), 'nc_light_amber', tile=(3, 2), emis=0.2)
    for x in (-W / 2 + 1.0, W / 2 - 1.0):
        P.beacon_light(M, (x, 0.0, H))
    M.light((tx, 0.0, H - 4.0), P.WARMW, 1.2, 42.0)
    for sy in (-1.0, 1.0):
        M.box((-W / 2 + 1.0, sy * (D / 2 - 3.0) - 0.18, 1.0), (W / 2 - 1.0, sy * (D / 2 - 3.0) + 0.18, 1.12), 'nc_strip_cyan', tile=(3, 1), emis=0.8)
    return M, C, dict(kind='port_crane', w=W + 18.0, d=D, h=H + 0.5, dist=900.0, landmark=True)


def airport_apron(length=240.0, width=78.0, seed=0):
    """Terminal apron with walkable concrete, aircraft stand boxes, pushback guides and blue lead-in lights."""
    L, W = float(length), float(width)
    rng = np.random.default_rng(seed)
    M, C = Mesh(), Col()
    M.hquad(-L / 2, -W / 2, L / 2, W / 2, 0.0, 'nc_concrete', tile=(8.0, 8.0))
    C.box((-L / 2, -W / 2, -0.35), (L / 2, W / 2, 0.0))
    # A double yellow lead-in, three numbered contact stands and red safety limits.
    M.box((-L / 2 + 8.0, -0.16, 0.018), (L / 2 - 8.0, 0.16, 0.03), 'nc_light_amber', tile=(6, 1), emis=0.35)
    for k, x in enumerate(np.linspace(-L * 0.32, L * 0.32, 3)):
        M.ribbon(np.asarray([[x, -W * 0.36, 0.035], [x, W * 0.22, 0.035]]), 0.055, 'nc_strip_white', tile_v=8.0)
        M.box((x - 6.0, W * 0.22, 0.018), (x + 6.0, W * 0.22 + 0.12, 0.03), 'nc_strip_white', tile=(3, 1))
        M.box((x - 7.0, -W * 0.40, 0.02), (x + 7.0, -W * 0.40 + 0.14, 0.035), 'nc_light_red', tile=(3, 1), emis=0.55)
        for side in (-1.0, 1.0):
            px, py = x + side * 10.0, W * 0.28
            M.cyl((px, py), 0.18, 0.03, 0.14, 8, 'nc_light_cyan', tile=(1, 1), top_mat='nc_light_cyan', emis=0.8)
        if k == 1:
            M.box((x - 5.0, W * 0.32, 0.04), (x + 5.0, W * 0.32 + 0.14, 0.06), 'nc_strip_white', tile=(2, 1))
    for x in (-L / 2 + 3.0, L / 2 - 3.0):
        M.box((x - 0.2, -W / 2 + 1.0, 0.02), (x + 0.2, W / 2 - 1.0, 0.04), 'nc_light_red', tile=(1, 3), emis=0.45)
    # Low service bollards mark the terminal-facing edge without becoming collision clutter.
    for x in np.arange(-L / 2 + 14.0, L / 2 - 8.0, 20.0):
        M.merge(P.bollard(), (x, W / 2 - 1.4, 0.0), rz=0.0)
    M.light((0.0, 0.0, 3.5), P.COOL, 0.45, 34.0)
    return M, C, dict(kind='airport_apron', w=L, d=W, h=4.0, dist=1000.0)
