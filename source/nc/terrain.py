# Created by: Arena.ai Agent Mode (AI) - NightCity terrain / coast data
# -----------------------------------------------------------------------------
# Height field shared by terrain tiles, roads, buildings and the validation report.
# City coordinates are metres; x east, y north, z up.  The urban basin stays at street
# datum while the outlying ridges rise naturally and the south / east / west edges fall
# through an irregular beach into the sea.
# -----------------------------------------------------------------------------
import math

import numpy as np

TILE = 384.0
WORLD = (-2304.0, -2304.0, 2304.0, 1920.0)
SEA_Z = -3.0


def coast_south(x):
    """Curved southern shoreline with a broad embayment for the working harbor."""
    x = np.asarray(x, dtype=np.float64)
    bay = 360.0 * np.exp(-((x - 520.0) / 260.0) ** 2)
    return -1180.0 + 110.0 * np.sin(x / 360.0) + 75.0 * np.sin(x / 145.0 + 0.8) + bay


def coast_east(y):
    """The eastern coast is a separate, irregular headland rather than a map edge."""
    y = np.asarray(y, dtype=np.float64)
    inlet = 240.0 * np.exp(-((y + 450.0) / 400.0) ** 2)
    return 1500.0 + 100.0 * np.sin(y / 330.0) + inlet


def coast_west(y):
    y = np.asarray(y, dtype=np.float64)
    return -1900.0 + 105.0 * np.sin(y / 410.0 + 0.4) + 45.0 * np.sin(y / 155.0)


def land_distance(x, y):
    """Positive on land, negative offshore; the min of three curved coast distances."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    south = y - coast_south(x)
    east = coast_east(y) - x
    west = x - coast_west(y)
    return np.minimum(np.minimum(south, east), west)


def _smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def height(x, y):
    """Deterministic landscape elevation in city space, in metres."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    dland = land_distance(x, y)
    # Tidal shelf and beach: a short, readable shoreline transition above the -3 m sea.
    beach = -3.0 + 3.15 * _smoothstep(0.0, 72.0, dland)
    sea_floor = -9.0 - np.minimum(np.maximum(-dland, 0.0) * 0.08, 48.0)
    shore_blend = _smoothstep(-96.0, 0.0, dland)
    shore = sea_floor * (1.0 - shore_blend) - 3.0 * shore_blend

    # A broad, very slightly undulating coastal plain; the central urban bowl is level
    # enough that the hand-built road / sidewalk cells meet without floating seams.
    urban_r = np.sqrt((x / 930.0) ** 2 + (y / 900.0) ** 2)
    basin_blend = _smoothstep(0.72, 1.22, urban_r)
    low_roll = 2.0 * np.sin(x / 410.0) * np.sin(y / 370.0) + 1.1 * np.sin((x + y) / 175.0)
    plain = -1.15 + basin_blend * (7.0 + low_roll)
    # The hand-built central street grid is an exact flat bench at near street datum;
    # relief transitions smoothly into the rolling suburbs just outside its footprint.
    city_r = np.maximum(np.abs(x) / 602.0, np.abs(y) / 692.0)
    city_flat = 1.0 - _smoothstep(1.0, 1.16, city_r)
    plain = plain * (1.0 - city_flat) - 0.10 * city_flat

    # Separate mountain masses, with asymmetric shoulders and a low-frequency ridge
    # term.  None of these are faceted pyramids; the overlapping lobes produce irregular
    # silhouettes and valleys between the ranges.
    peaks = (
        (-1630.0, 1190.0, 205.0, 360.0, 290.0),
        (-1110.0, 1590.0, 165.0, 420.0, 300.0),
        (-420.0, 1510.0, 130.0, 390.0, 320.0),
        (390.0, 1450.0, 176.0, 410.0, 300.0),
        (1110.0, 1120.0, 138.0, 360.0, 420.0),
        (-1840.0, 390.0, 88.0, 270.0, 420.0),
    )
    relief = np.zeros_like(x)
    for px, py, amp, sx, sy in peaks:
        q = ((x - px) / sx) ** 2 + ((y - py) / sy) ** 2
        relief += amp * np.exp(-0.5 * q)
    ridge = (np.sin((x + 0.35 * y) / 112.0) * np.sin(y / 79.0)
             + 0.42 * np.sin(x / 53.0 + y / 170.0))
    rugged = np.clip((relief - 18.0) / 180.0, 0.0, 1.0)
    relief += rugged * (12.0 * ridge + 6.0 * np.sin((x - y) / 35.0))
    # An alluvial valley cuts between the north-west ridge and the coastal hills.
    valley = 18.0 * np.exp(-((x + 1010.0 + 0.24 * (y - 850.0)) / 140.0) ** 2)
    land_h = np.minimum(plain + relief - valley, 220.0)

    # The beach-to-inland transition is a real slope, not a hard switch at one contour.
    # This avoids the 30-50 m coastal cliffs that a stepped height function created.
    land_blend = _smoothstep(0.0, 180.0, dland)
    interior = beach * (1.0 - land_blend) + land_h * land_blend

    # Designed flat pads sit on top of the terrain blend so runways, the cargo yard and
    # the stadium remain level at their surveyed elevations. Their fades are masked at
    # the shoreline, preventing a pad from creating a vertical coast cut.
    inland_mask = _smoothstep(0.0, 72.0, dland)
    airport_r = np.maximum(np.abs((x - 1030.0) / 650.0), np.abs((y + 500.0) / 235.0))
    airport_blend = (1.0 - _smoothstep(0.78, 1.12, airport_r)) * inland_mask
    interior = interior * (1.0 - airport_blend) + (-0.9) * airport_blend
    port_r = np.maximum(np.abs((x + 410.0) / 340.0), np.abs((y + 1030.0) / 205.0))
    port_blend = (1.0 - _smoothstep(0.72, 1.12, port_r)) * inland_mask
    interior = interior * (1.0 - port_blend) + (-0.8) * port_blend
    stadium_r = np.maximum(np.abs((x - 900.0) / 250.0), np.abs((y - 300.0) / 150.0))
    stadium_blend = (1.0 - _smoothstep(0.86, 1.16, stadium_r)) * inland_mask
    interior = interior * (1.0 - stadium_blend) + 8.0 * stadium_blend

    out = np.where(dland < 0.0, shore, interior)
    return out


def surface_material(x, y, z, slope=0.0):
    """Stable material zoning for the terrain mesh: beach, plain, rock and shaded dirt."""
    d = float(land_distance(x, y))
    if d < 75.0 and z < 4.0:
        return 'nc_sand'
    if z > 54.0 or slope > 0.62:
        return 'nc_rock'
    if z < 12.0:
        return 'nc_grass'
    return 'nc_dirt'


def terrain_tile(size=384.0, step=16.0, lod=False, cx=0.0, cy=0.0, holes=()):
    """Build a terrain tile, clipping world-space rectangular openings exactly; high detail carries walkable triangle collision."""
    from .mb import Col, Mesh

    n = int(round(size / step))
    coords = np.linspace(-size / 2.0, size / 2.0, n + 1)
    xx, yy = np.meshgrid(coords, coords, indexing='xy')
    # Offset-free sampling keeps adjacent tiles bit-identical at their shared edges.
    cx, cy = float(cx), float(cy)
    wx, wy = cx + xx, cy + yy
    zz = height(wx, wy)
    # A small deterministic, sub-metre micro-variation enriches the turf but is deliberately
    # suppressed in the city basin and at the airport / port pads.
    basin = np.sqrt((wx / 930.0) ** 2 + (wy / 900.0) ** 2) < 1.0
    pad = (np.maximum(np.abs((wx - 1030.0) / 650.0), np.abs((wy + 500.0) / 235.0)) < 0.82)
    pad |= (np.maximum(np.abs((wx + 410.0) / 340.0), np.abs((wy + 1030.0) / 205.0)) < 0.76)
    pad |= (np.maximum(np.abs((wx - 900.0) / 215.0), np.abs((wy - 300.0) / 129.0)) < 0.82)
    micro = (0.24 * np.sin(wx / 23.0 + 0.7) * np.sin(wy / 31.0 - 0.3)
             + 0.12 * np.sin((wx + wy) / 11.0))
    zz = zz + np.where(basin | pad | (land_distance(wx, wy) < 80.0), 0.0, micro)

    M, C = Mesh(), Col()
    rows = []
    by_material = {}
    # Split only at cutout boundaries, then discard sub-quads wholly inside the opening.
    # Dropping an entire 24 m terrain quad when a river rectangle touched one corner used
    # to leave broad collision holes on the banks and shorten the tile's visible bounds.
    cutouts = tuple(tuple(float(v) for v in r) for r in holes)
    tx0, tx1 = cx - size / 2.0, cx + size / 2.0
    ty0, ty1 = cy - size / 2.0, cy + size / 2.0
    tile_cutouts = tuple(r for r in cutouts if tx0 < r[2] and tx1 > r[0] and ty0 < r[3] and ty1 > r[1])

    def sampled_z(x, y):
        z = float(height(x, y))
        in_basin = math.sqrt((x / 930.0) ** 2 + (y / 900.0) ** 2) < 1.0
        in_pad = (max(abs((x - 1030.0) / 650.0), abs((y + 500.0) / 235.0)) < 0.82
                   or max(abs((x + 410.0) / 340.0), abs((y + 1030.0) / 205.0)) < 0.76
                   or max(abs((x - 900.0) / 215.0), abs((y - 300.0) / 129.0)) < 0.82)
        if not in_basin and not in_pad and float(land_distance(x, y)) >= 80.0:
            z += 0.24 * math.sin(x / 23.0 + 0.7) * math.sin(y / 31.0 - 0.3) + 0.12 * math.sin((x + y) / 11.0)
        return z

    def add_quad(p, x0, x1, y0, y1):
        gx = (p[1, 2] + p[2, 2] - p[0, 2] - p[3, 2]) / (2.0 * (x1 - x0))
        gy = (p[2, 2] + p[3, 2] - p[0, 2] - p[1, 2]) / (2.0 * (y1 - y0))
        slope = math.hypot(float(gx), float(gy))
        zc = float(p[:, 2].mean())
        mx, my = float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)
        material = surface_material(mx, my, zc, slope)
        by_material.setdefault(material, []).append(p)
        # Up-facing triangle soup is an actual walkable collision surface, rather than
        # a giant AABB that would create invisible walls through the hills.
        rows.append(p)

    for j in range(n):
        for i in range(n):
            qx0, qx1 = cx + xx[j, i], cx + xx[j, i + 1]
            qy0, qy1 = cy + yy[j, i], cy + yy[j + 1, i]
            local_holes = [r for r in tile_cutouts if qx0 < r[2] and qx1 > r[0] and qy0 < r[3] and qy1 > r[1]]
            if not local_holes:
                p = np.array([
                    [xx[j, i], yy[j, i], zz[j, i]],
                    [xx[j, i + 1], yy[j, i], zz[j, i + 1]],
                    [xx[j + 1, i + 1], yy[j + 1, i + 1], zz[j + 1, i + 1]],
                    [xx[j + 1, i], yy[j + 1, i], zz[j + 1, i]],
                ], dtype=np.float64)
                add_quad(p, qx0, qx1, qy0, qy1)
                continue

            xcuts = {qx0, qx1}
            ycuts = {qy0, qy1}
            for hx0, hy0, hx1, hy1 in local_holes:
                if qx0 < hx0 < qx1: xcuts.add(hx0)
                if qx0 < hx1 < qx1: xcuts.add(hx1)
                if qy0 < hy0 < qy1: ycuts.add(hy0)
                if qy0 < hy1 < qy1: ycuts.add(hy1)
            xs, ys = sorted(xcuts), sorted(ycuts)
            for xa, xb in zip(xs, xs[1:]):
                for ya, yb in zip(ys, ys[1:]):
                    mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
                    if any(hx0 <= mx <= hx1 and hy0 <= my <= hy1 for hx0, hy0, hx1, hy1 in local_holes):
                        continue
                    p = np.array([
                        [xa - cx, ya - cy, sampled_z(xa, ya)],
                        [xb - cx, ya - cy, sampled_z(xb, ya)],
                        [xb - cx, yb - cy, sampled_z(xb, yb)],
                        [xa - cx, yb - cy, sampled_z(xa, yb)],
                    ], dtype=np.float64)
                    add_quad(p, xa, xb, ya, yb)
    for material, quads in by_material.items():
        M.quads(np.asarray(quads), material, emis=0.0)
    verts = np.concatenate(rows, axis=0) if rows else np.zeros((0, 3), np.float64)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 4)
    tri = np.concatenate((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]), axis=0)
    if not lod:
        C.mesh(verts, tri, 0)
    zmin = float(zz.min())
    zmax = float(zz.max())
    meta = dict(kind='terrain_lod' if lod else 'terrain', w=float(size), d=float(size), h=zmax - zmin,
                dist=2600.0 if lod else 1400.0, nocol=bool(lod), lod=bool(lod), step=float(step),
                slope='natural', collision='mesh' if not lod else 'none')
    return M, C, meta
