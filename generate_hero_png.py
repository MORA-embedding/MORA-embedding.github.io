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
S_PX_PER_RAD = 256 * (2 ** ZOOM) / (2 * math.pi)   # ~10430 px/rad at zoom 8

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
FILL_ALPHA = int(0.52 * 255)   # ~133 — more transparent so base map shows through

# ─────────────────────────────────────────────────────────────────────────────
def _merc_y(lat_deg):
    lat = math.radians(lat_deg)
    return math.log(math.tan(math.pi / 4 + lat / 2))

_MY0 = _merc_y(LAT_CENTER)

def geo_to_px(lon, lat):
    x = math.radians(lon - LON_CENTER) * S_PX_PER_RAD + W / 2
    y = -(_merc_y(lat) - _MY0) * S_PX_PER_RAD + H / 2
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

    # canvas top-left in tile-pixels
    canvas_left_px = cx_px - W / 2
    canvas_top_px  = cy_px - H / 2

    # tile range needed
    tx_min = int(math.floor(canvas_left_px / TILE_SIZE))
    tx_max = int(math.ceil((canvas_left_px + W) / TILE_SIZE))
    ty_min = int(math.floor(canvas_top_px / TILE_SIZE))
    ty_max = int(math.ceil((canvas_top_px + H) / TILE_SIZE))

    n_tiles = (2 ** ZOOM)
    jobs = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tx_wrapped = tx % n_tiles
            jobs.append((tx, ty, tx_wrapped))

    print(f"Fetching {len(jobs)} tiles (zoom={ZOOM})…")
    canvas = Image.new("RGBA", (W, H), (20, 20, 20, 255))

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
    cells = []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells.append((row["h3"], float(row["similarity"])))
    print(f"  {len(cells)} cells loaded")

    # draw onto a separate RGBA overlay, then composite
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    drawn = 0
    for h3_id, sim in cells:
        boundary = h3lib.cell_to_boundary(h3_id)   # list of (lat, lon)
        pts = [geo_to_px(lon, lat) for lat, lon in boundary]
        # clip: skip if all vertices outside canvas with margin
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) < -20 or min(xs) > W + 20 or max(ys) < -20 or min(ys) > H + 20:
            continue
        color = sim_to_rgba(sim)
        pts_int = [(int(round(x)), int(round(y))) for x, y in pts]
        draw.polygon(pts_int, fill=color)
        drawn += 1

    print(f"  {drawn} hexagons drawn")
    result = Image.alpha_composite(base_img, overlay)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Vignettes
# ─────────────────────────────────────────────────────────────────────────────
def apply_vignettes(img):
    arr = np.array(img, dtype=np.float32)

    # Left vignette: darkens left edge
    # stops: x=0%→α=0.985, 24%→0.9, 43%→0.52, 66%→0.18, 100%→0.04
    xs = np.linspace(0.0, 1.0, W)
    vx_stops = [(0.00, 0.985), (0.24, 0.900), (0.43, 0.520), (0.66, 0.180), (1.00, 0.040)]
    left_mult = np.interp(xs, [s[0] for s in vx_stops], [s[1] for s in vx_stops])
    # convert: multiply = 1 means no change; at x=0, mult=0.985 means slight darkening
    # Actually the vignette darkens: alpha of a black overlay = (1 - mult)
    left_dark = 1.0 - left_mult   # 0 at right, ~0.96 at left edge
    # flip so left edge is darkest
    left_dark = left_dark[::-1]
    left_dark_2d = left_dark[np.newaxis, :, np.newaxis]   # (1, W, 1)

    arr[:, :, :3] *= (1.0 - left_dark_2d)

    # Bottom vignette: 70%→0, 85%→0.42, 100%→0.92
    ys = np.linspace(0.0, 1.0, H)
    vy_stops = [(0.00, 0.0), (0.70, 0.0), (0.85, 0.42), (1.00, 0.92)]
    bot_alpha = np.interp(ys, [s[0] for s in vy_stops], [s[1] for s in vy_stops])
    bot_alpha_2d = bot_alpha[:, np.newaxis, np.newaxis]   # (H, 1, 1)

    arr[:, :, :3] *= (1.0 - bot_alpha_2d)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=== Generating hero_map.png ===")
    base = build_base_map()
    with_hex = draw_hexagons(base)
    final = apply_vignettes(with_hex)
    # convert to RGB for smaller PNG
    final_rgb = final.convert("RGB")
    final_rgb.save(OUT_PNG, "PNG", optimize=False, compress_level=6)
    size_mb = os.path.getsize(OUT_PNG) / 1e6
    print(f"Saved {OUT_PNG}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
