"""
Generate hero_map.png (1920×1080) with CartoDB dark_all base tiles + H3 hexagons.
"""
import math, os, csv, time
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ── Output ────────────────────────────────────────────────────────────────────
W, H       = 1920, 1080
_DIR       = os.path.dirname(os.path.abspath(__file__))
OUT_PNG    = os.path.join(_DIR, "hero_map.png")
CSV_PATH   = os.path.join(_DIR, "EastChina_one_h3_similarity.csv")
TILE_CACHE = os.path.join(_DIR, ".tile_cache")

# ── Map params ────────────────────────────────────────────────────────────────
ZOOM        = 8
LON_CENTER  = 117.5
LAT_CENTER  = 30.5
TILE_SIZE   = 256
# Render at a larger intermediate canvas then downscale — keeps tiles & hexagons
# perfectly aligned at zoom-8 while achieving a zoom-out effect.
ZOOM_SCALE  = 0.76          # <1 = zoom out; 1.0 = native zoom-8
WI = int(W / ZOOM_SCALE)    # intermediate canvas width  (~2526)
HI = int(H / ZOOM_SCALE)    # intermediate canvas height (~1421)
S_PX_PER_RAD = 256 * (2 ** ZOOM) / (2 * math.pi)  # 10430 — tile-aligned, applied to WI×HI canvas

# ── Tile source (CartoDB dark, English labels) ─────────────────────────────
TILE_SERVERS = [
    "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
]
HEADERS = {"User-Agent": "Mozilla/5.0 MoRA-hero-map-generator/1.0"}

# ── Colors ────────────────────────────────────────────────────────────────────
STOPS = [
    (-0.30, ( 13,   8, 135)),
    ( 0.00, ( 59,   4, 154)),
    ( 0.15, ( 91,   2, 163)),
    ( 0.30, (139,  10, 165)),
    ( 0.42, (185,  50, 137)),
    ( 0.54, (219,  92, 104)),
    ( 0.65, (237, 121,  83)),
    ( 0.75, (248, 149,  64)),
    ( 0.85, (253, 180,  47)),
    ( 0.95, (240, 249,  33)),
    ( 1.00, (255, 255, 255)),
]
FILL_ALPHA    = int(0.60 * 255)   # 153 — semi-transparent so base map reads through
LINE_COLOR    = (255, 255, 255, 120)  # ~47% — clearly visible grid boundary

# ─────────────────────────────────────────────────────────────────────────────
def _merc_y(lat_deg):
    lat = math.radians(lat_deg)
    return math.log(math.tan(math.pi / 4 + lat / 2))

_MY0 = _merc_y(LAT_CENTER)

def geo_to_px(lon, lat):
    x = math.radians(lon - LON_CENTER) * S_PX_PER_RAD + WI / 2
    y = -(_merc_y(lat) - _MY0) * S_PX_PER_RAD + HI / 2
    return x, y

def sim_to_rgba(sim):
    sim = max(STOPS[0][0], min(STOPS[-1][0], sim))
    for i in range(len(STOPS) - 1):
        v0, c0 = STOPS[i]
        v1, c1 = STOPS[i + 1]
        if v0 <= sim <= v1:
            t = (sim - v0) / (v1 - v0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b, FILL_ALPHA)
    return (13, 8, 135, FILL_ALPHA)

# ─────────────────────────────────────────────────────────────────────────────
# Tile helpers
# ─────────────────────────────────────────────────────────────────────────────
def lon_to_tile_x(lon, zoom):
    return (lon + 180) / 360 * (2 ** zoom)

def lat_to_tile_y(lat, zoom):
    lat_r = math.radians(lat)
    return (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * (2 ** zoom)

def fetch_tile(z, x, y, server_idx=0):
    cache_path = os.path.join(TILE_CACHE, str(z), str(x), f"{y}.png")
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGBA")
    url = TILE_SERVERS[server_idx % len(TILE_SERVERS)].format(z=z, x=x, y=y)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(resp.content)
                return Image.open(cache_path).convert("RGBA")
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    print(f"  WARN: failed tile z={z} x={x} y={y}")
    return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (20, 20, 20, 255))

# ─────────────────────────────────────────────────────────────────────────────
# Build base map canvas from tiles
# ─────────────────────────────────────────────────────────────────────────────
def build_base_map():
    tx0f = lon_to_tile_x(LON_CENTER, ZOOM)
    ty0f = lat_to_tile_y(LAT_CENTER, ZOOM)
    # pixel offset of map center within its tile grid
    cx_px = tx0f * TILE_SIZE
    cy_px = ty0f * TILE_SIZE

    # canvas top-left in tile-pixels (use intermediate WI×HI canvas)
    canvas_left_px = cx_px - WI / 2
    canvas_top_px  = cy_px - HI / 2

    # tile range needed
    tx_min = int(math.floor(canvas_left_px / TILE_SIZE))
    tx_max = int(math.ceil((canvas_left_px + WI) / TILE_SIZE))
    ty_min = int(math.floor(canvas_top_px / TILE_SIZE))
    ty_max = int(math.ceil((canvas_top_px + HI) / TILE_SIZE))

    n_tiles = (2 ** ZOOM)
    jobs = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tx_wrapped = tx % n_tiles
            jobs.append((tx, ty, tx_wrapped))

    print(f"Fetching {len(jobs)} tiles (zoom={ZOOM}, intermediate {WI}×{HI})…")
    canvas = Image.new("RGBA", (WI, HI), (20, 20, 20, 255))

    results = {}
    idx = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        future_map = {ex.submit(fetch_tile, ZOOM, tw, ty, i): (tx, ty)
                      for i, (tx, ty, tw) in enumerate(jobs)}
        for fut in as_completed(future_map):
            tx, ty = future_map[fut]
            results[(tx, ty)] = fut.result()
            idx += 1
            if idx % 10 == 0:
                print(f"  {idx}/{len(jobs)} tiles done")

    for (tx, ty), tile_img in results.items():
        paste_x = int(tx * TILE_SIZE - canvas_left_px)
        paste_y = int(ty * TILE_SIZE - canvas_top_px)
        canvas.paste(tile_img, (paste_x, paste_y))

    print("Base map assembled.")
    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# Draw H3 hexagons
# ─────────────────────────────────────────────────────────────────────────────
def draw_hexagons(base_img):
    try:
        import h3 as h3lib
    except ImportError:
        print("h3 not available — skipping hexagons")
        return base_img

    print("Loading CSV…")
    sim_map = {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim_map[row["h3"]] = float(row["similarity"])
    print(f"  {len(sim_map)} data cells loaded")

    # collect visible CSV cells only
    data_visible = []
    for h3_id, sim in sim_map.items():
        boundary = h3lib.cell_to_boundary(h3_id)
        pts = [geo_to_px(lon, lat) for lat, lon in boundary]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if max(xs) < -20 or min(xs) > WI + 20 or max(ys) < -20 or min(ys) > HI + 20:
            continue
        pts_int = [(int(round(x)), int(round(y))) for x, y in pts]
        data_visible.append((pts_int, sim_to_rgba(sim)))

    print(f"  {len(data_visible)} cells rendered")

    # pass 1: fills
    fill_layer = Image.new("RGBA", (WI, HI), (0, 0, 0, 0))
    draw_f = ImageDraw.Draw(fill_layer)
    for pts_int, color in data_visible:
        draw_f.polygon(pts_int, fill=color)

    # pass 2: outlines
    line_layer = Image.new("RGBA", (WI, HI), (0, 0, 0, 0))
    draw_l = ImageDraw.Draw(line_layer)
    for pts_int, _ in data_visible:
        draw_l.polygon(pts_int, fill=None, outline=LINE_COLOR)

    result = Image.alpha_composite(base_img, fill_layer)
    result = Image.alpha_composite(result, line_layer)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Vignettes
# ─────────────────────────────────────────────────────────────────────────────
def apply_vignettes(img):
    arr = np.array(img, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, WI)
    ys = np.linspace(0.0, 1.0, HI)

    # Left vignette: soft fade, keeps map labels faintly visible
    vx_l = [(0.00, 0.82), (0.15, 0.60), (0.30, 0.20), (0.45, 0.00)]
    left_dark = np.interp(xs, [s[0] for s in vx_l], [s[1] for s in vx_l])

    # Right vignette: only the last ~10% to hide ocean edge after coastline
    vx_r = [(0.00, 0.00), (0.88, 0.00), (0.93, 0.55), (0.97, 0.90), (1.00, 0.98)]
    right_dark = np.interp(xs, [s[0] for s in vx_r], [s[1] for s in vx_r])

    h_mult = (1.0 - left_dark) * (1.0 - right_dark)
    arr[:, :, :3] *= h_mult[np.newaxis, :, np.newaxis]

    # Top vignette
    vy_t = [(0.00, 0.92), (0.08, 0.55), (0.18, 0.10), (0.28, 0.00)]
    top_dark = np.interp(ys, [s[0] for s in vy_t], [s[1] for s in vy_t])

    # Bottom vignette
    vy_b = [(0.00, 0.00), (0.70, 0.00), (0.85, 0.42), (1.00, 0.92)]
    bot_dark = np.interp(ys, [s[0] for s in vy_b], [s[1] for s in vy_b])

    v_mult = (1.0 - top_dark) * (1.0 - bot_dark)
    arr[:, :, :3] *= v_mult[:, np.newaxis, np.newaxis]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=== Generating hero_map.png ===")
    base = build_base_map()
    with_hex = draw_hexagons(base)
    full = apply_vignettes(with_hex)
    # Downscale intermediate canvas to final output size (anti-aliased)
    final = full.resize((W, H), Image.LANCZOS)
    final.convert("RGB").save(OUT_PNG, "PNG", optimize=False, compress_level=6)
    size_mb = os.path.getsize(OUT_PNG) / 1e6
    print(f"Saved {OUT_PNG}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
