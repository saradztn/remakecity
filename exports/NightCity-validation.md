# NightCity build and validation report

**Status:** Built, packaged, and passed the repository's automated build checks.  
**Build date:** 2026-10-05  
**Resource folder:** `resource/NightCity/`  
**Export archive:** `exports/NightCity-export.zip` (contains `NightCity/meta.xml` at its root)

## Build

Built from the current `source/` pipeline with `python source/build.py`.

- 688 custom models; 1,385 placements (within the 1,500-placement budget)
- 103 water rectangles: 12 river and 91 coastal
- 132 high-detail terrain tiles, paired with 132 no-collision terrain LODs
- 4,345 lighting / ambience sprites
- 800,595 reported vertices; 449,334 triangles; largest model: 19,175 vertices
- Archive contents: 1,401 files, 65,733,718 uncompressed bytes; ZIP: 22,941,475 bytes
- ZIP CRC check passed
- SHA-256: `6aea1e30980e56b5fd33487f84e964ad0c77b86ac09fb78823be79c0d1e94910`

## Automated validation

- `source/validate.py --librw`: **256 checks passed, 0 failed, 0 validator warnings**
  - Model/material/layout checks passed, including the duplicate-face audit (zero same-facing coincident triangles).
  - Terrain tile and LOD coverage passed; the 132 terrain LOD elements are paired and excluded from collision probes.
  - Terrain / street collision coverage, visible-ground agreement, water coverage, and the road-tunnel geometry and portal checks passed.
  - Resource budgets passed; collision export contains 5,276 boxes and 79,600 triangles.
  - The independent `aap/librw` reference loader parsed all 688 DFF files with their TXDs.
- `source/mta_lua_test_nc.py`: **207 checks passed, 0 failed**
  - Covers load/show/hide and failure cleanup, environment controls, the 300-second camera tour, tunnel passage, and both metro services.
  - Confirms each metro level runs a four-car consist using vanilla GTA:SA train models 538 and 570, calls at stations, supports boarding/riding/exiting, and cleans up on hide. Both loops complete a full lap to MARKET without a position jump (maximum 50 ms sampled step: 0.80 m lower, 0.90 m upper).
- `source/mta_lua_static.py`: passed; client and server MTA API calls are recognized.

## Metro and terrain changes

The metro is now two independent operational loops: a lower tunnel service at city z=13 and an upper express service at z=31, offset 16 m north. Both serve MARKET, UNION, and DOCKS. Use `/ncmetro` for the lower platform and `/ncmetro 2` for the upper platform; press **E** near a train to board or leave.

Terrain openings are clipped to exact river / portal boundaries rather than dropping every coarse terrain cell they touch. This preserves the river bed and surrounding walkable collision. Each high-detail tile has a linked, lower-resolution no-collision LOD.

## Known verification limit

The resource has **not** been launched in a live MTA:SA client/server in this environment. The Lua runtime/API harness is headless and cannot verify in-engine streaming, train rendering, LOD switching, or MTA's Direct3D 9 effect compiler. SlangPy's CPU front-end type-checked all four shader bodies, but its runtime reported that no shader model was supported and fell back to `sm_6_0`; this does not certify the game's `ps_2_0` / `ps_3_0` profiles or exact MTA effect syntax. These limits are not counted as passes.

## Install

Extract `NightCity-export.zip` into the MTA server's `resources/` directory so the result is `resources/NightCity/meta.xml`, then start the resource with `/start NightCity`. The default `/ncshow` command places the city above the map; `/nchide` removes it.

**No preview render was created.** The explicit `CREATE PREVIEW` condition has not been given.
