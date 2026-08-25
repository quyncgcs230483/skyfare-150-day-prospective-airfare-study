"""Collect Trip.com observations through Playwright and Camoufox.

Output defaults to ``data/raw/trip_com/YYYY-MM-DD.csv`` relative to repository
root. Set ``SKYFARE_TRIP_OUTPUT_DIR`` to override it.

Collect all five airlines from Trip.com through Playwright browser automation,
with Camoufox providing browser isolation.
Trip.com shows all airlines together per route+date query \u2192 1 search = all airlines.

Routes  : 20 routes x 11 windows = 220 queries/day
Schema  : shared raw-offer schema (data_source = "trip_com")

Install:
    pip install camoufox[geoip]
    python -m camoufox fetch

Run:
    python scrape_trip_all.py
"""

import asyncio
import csv
import os
import random
import sys
from datetime import datetime, timedelta

from camoufox.async_api import AsyncCamoufox

from skyfare.core.paths import DataLayout

# ======================================================
# CONFIG
# ======================================================

LAYOUT = DataLayout.resolve()
OUTPUT_DIR = LAYOUT.raw_trip_com
LOG_DIR = LAYOUT.collection_logs

USD_TO_VND = 26_309

ROUTES = [
    ("SGN", "HAN"), ("HAN", "SGN"),
    ("SGN", "PQC"), ("PQC", "SGN"),
    ("HAN", "PQC"), ("PQC", "HAN"),
    ("DAD", "PQC"), ("PQC", "DAD"),
    ("SGN", "DAD"), ("DAD", "SGN"),
    ("HAN", "DAD"), ("DAD", "HAN"),
    ("SGN", "CXR"), ("CXR", "SGN"),
    ("HAN", "CXR"), ("CXR", "HAN"),
    ("SGN", "HPH"), ("HPH", "SGN"),
    ("HAN", "VCA"), ("VCA", "HAN"),
]

BOOKING_WINDOWS = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1]
# HISTORICAL DATA ()

# \u2500\u2500 TEST MODES \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#   --test   : 1 query, 1 tab (verify collection stability)
#   --test3  : 3 queries, 3 tabs concurrently (verify parallel machinery + CSV safety)
TEST_MODE  = ("--test"  in sys.argv) or (os.environ.get("TEST_MODE")  == "1")
TEST3_MODE = ("--test3" in sys.argv) or (os.environ.get("TEST3_MODE") == "1")

# Camoufox's automatic GeoIP lookup depends on an external public-IP service.
# Keep it off by default so a transient lookup failure cannot stop the scraper.
GEOIP_ENABLED = os.environ.get("CAMOUFOX_GEOIP", "").strip().lower() in {
    "1", "true", "yes",
}

# Rate-limit and access-verification handling.
BLOCK_MARKERS    = ["whale" + "guard", "verify", "captcha", "robot check",
                    "are you a human", "unusual traffic", "access denied",
                    "security check", "blocked"]
MAX_BLOCK_RETRY  = 2            # extra reload attempts when a block is detected
SELECTOR_BUDGET  = 30           # seconds to wait for flights OR block to resolve (was 45)

# Stale-fare interstitial ("ve expired / gia da cu").
# Trip.com expires fares when a page sits too long and shows a "T\u1ea3i l\u1ea1i" reload prompt.
STALE_MARKERS    = ["t\u1ea3i l\u1ea1i", "\u0111\u00e3 c\u00f3 l\u1ed7i x\u1ea3y ra", "vui l\u00f2ng th\u1eed", "gi\u00e1 \u0111\u00e3 c\u0169",
                    "reload", "try again", "something went wrong"]
RELOAD_LABELS    = ["T\u1ea3i l\u1ea1i", "T\u1ea3i l\u1ea1i trang", "Th\u1eed l\u1ea1i", "Reload", "Try again", "Retry"]
MAX_STALE_RETRY  = 3            # reload attempts on a stale-fare interstitial

# \u2500\u2500 3-tab parallel tuning (balanced cadence \u2014 3 tabs run at once) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
N_WORKERS = 1
W_DELAY_MIN             = 2          # per-worker inter-query delay lower bound (was 4)
W_DELAY_MAX             = 6          # per-worker inter-query delay upper bound (was 9)
# (long-break-every-12 removed for speed \u2014 always use short inter-query delay)
W_CONSEC_BLOCK_LIMIT    = 5          # consecutive empty/blocked queries \u2192 extra cool-down
W_CONSEC_BLOCK_COOLDOWN = 120        # seconds cool-down for a stuck worker
W_STAGGER_STEP          = 12         # stagger worker startup by k*12s (offset per-tab warm-ups)
GOTO_TIMEOUT            = 75000      # ms \u2014 flight-page goto budget (raised for 3-tab contention)


CITY_NAMES = {
    "SGN": ("sgn", "Ho Chi Minh City"),
    "HAN": ("han", "Hanoi"),
    "DAD": ("dad", "Da Nang"),
    "PQC": ("pqc", "Phu Quoc"),
    "CXR": ("cxr", "Nha Trang"),
    "HPH": ("hph", "Hai Phong"),
    "VCA": ("vca", "Can Tho"),
}

# Ordered by priority: more specific keywords first to avoid false matches
AIRLINE_MAP = [
    {"keywords": ["Vietnam Airlines"],            "code": "VN", "name": "Vietnam Airlines"},
    {"keywords": ["Vietravel"],                   "code": "VU", "name": "Vietravel Airlines"},
    {"keywords": ["Bamboo"],                      "code": "QH", "name": "Bamboo Airways"},
    {"keywords": ["VietJet", "Vietjet", "VJAIR"], "code": "VJ", "name": "Vietjet Air"},
    {"keywords": ["Sun PhuQuoc", "SunPhuQuoc", "Sun Phu Quoc", "9G"], "code": "9G", "name": "Sun Phu Quoc Airways"},
]

FIELDNAMES = [
    "scraped_at", "session_id",
    "origin", "dest", "route",
    "days_until_departure", "flight_date",
    "airline", "airline_name",
    "flight_no", "departure_time",
    "price_usd", "price_vnd",
    "data_source", "seats_left", "is_soldout",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

today_str   = datetime.today().strftime("%Y-%m-%d")
output_file = os.path.join(OUTPUT_DIR, f"{today_str}.csv")
log_file    = os.path.join(LOG_DIR,    f"trip_all_{today_str}.log")

# ======================================================
# HELPERS
# ======================================================

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_url(origin, dest, flight_date):
    dep_code, dep_name = CITY_NAMES[origin]
    arr_code, arr_name = CITY_NAMES[dest]
    # No &airline= filter \u2192 trip.com returns all airlines for this route+date
    return (
        f"https://www.trip.com/flights/showfarefirst"
        f"?pagesource=list"
        f"&triptype=OW"
        f"&class=Y"
        f"&quantity=1&childqty=0&babyqty=0"
        f"&dcity={dep_code}"
        f"&acity={arr_code}"
        f"&ddate={flight_date}"
        f"&dcityName={dep_name.replace(' ', '%20')}"
        f"&acityName={arr_name.replace(' ', '%20')}"
        f"&locale=en-XX"
        f"&curr=VND"
    )


def save_rows(rows):
    write_header = not os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ======================================================
# SCRAPE ONE PAGE
# ======================================================

_AIRLINE_MAP_JS = str(AIRLINE_MAP).replace("'", '"')

_EXTRACT_JS = f"""
() => {{
    const results = [];
    let hasNonNonstop = false;
    const AIRLINE_MAP = {_AIRLINE_MAP_JS};
    const cards = document.querySelectorAll('.J_FlightItem');
    cards.forEach(card => {{
        try {{
            const text = card.innerText || '';
            const premiumKeywords = [
                "Premium Economy","Premium","Flexible","Flex","Business",
                "SkyBoss","Skyboss","Deluxe","Bamboo Plus","Bamboo Business",
                "Th\u01b0\u01a1ng gia","Ph\u1ed5 th\u00f4ng \u0111\u1eb7c bi\u1ec7t"
            ];
            if (premiumKeywords.some(k => text.toLowerCase().includes(k.toLowerCase()))) return;
            if (!text.includes('Nonstop') && !text.includes('Direct')) {{
                hasNonNonstop = true;
                return;
            }}

            let airlineCode = null, airlineName = null;
            for (const entry of AIRLINE_MAP) {{
                if (entry.keywords.some(k => text.includes(k))) {{
                    airlineCode = entry.code;
                    airlineName = entry.name;
                    break;
                }}
            }}
            if (!airlineCode) return;

            const timeEls = card.querySelectorAll('[class*="time_"]');
            const times = Array.from(timeEls)
                .map(el => el.innerText.trim())
                .filter(t => /^\\d{{1,2}}:\\d{{2}}$/.test(t));
            const depTime = times[0] || '';
            if (!depTime) return;

            const fnMatch = text.match(/\\b(VN|VJ|QH|VU|9G)\\s*(\\d{{1,4}})\\b/);
            const flightNo = fnMatch
                ? (fnMatch[1] + fnMatch[2])
                : (airlineCode + '-' + depTime.replace(':', ''));

            const priceMatch = text.match(/VND\\s*([\\d,]+)/);
            let priceVnd = 0;
            if (priceMatch) priceVnd = parseInt(priceMatch[1].replace(/,/g, ''));

            const isSoldOut = text.toLowerCase().includes('sold out') ? 1 : 0;
            if (priceVnd > 100000 || isSoldOut) {{
                results.push({{
                    dep_time    : depTime,
                    price_vnd   : priceVnd,
                    is_soldout  : isSoldOut,
                    airline_code: airlineCode,
                    airline_name: airlineName,
                    flight_no   : flightNo,
                }});
            }}
        }} catch(e) {{}}
    }});
    return {{ results, hasNonNonstop }};
}}
"""

_FIND_MORE_JS = """
() => {
    const candidates = document.querySelectorAll('button, a, span, div, p');
    for (const el of candidates) {
        const t = (el.innerText || '').trim().toLowerCase();
        if (t.length > 60) continue;   // skip large containers
        if (
            t === 'show more results' ||
            t === 'view more results' ||
            t === 'load more'         ||
            t.startsWith('show more') ||
            t.startsWith('view more')
        ) {
            if (el.offsetParent !== null) {
                el.scrollIntoView({block: 'center', behavior: 'instant'});
                return t;
            }
        }
    }
    return null;
}
"""


async def _scroll_fast(page, steps=6, step_px=900):
    for _ in range(steps):
        await page.evaluate(f"window.scrollBy(0, {step_px})")
        await asyncio.sleep(random.uniform(0.3, 0.5))


# Request pacing and page-readiness helpers.
async def human_pause(a, b):
    await asyncio.sleep(random.uniform(a, b))


async def human_mouse(page, n=3):
    """Move cursor while waiting for client-rendered flight results."""
    for _ in range(n):
        try:
            await page.mouse.move(
                random.randint(80, 1180), random.randint(120, 760),
                steps=random.randint(5, 18),
            )
        except Exception:
            pass
        await asyncio.sleep(random.uniform(0.2, 0.6))


async def human_scroll(page):
    """Small randomized up/down scrolls (not just programmatic scroll-down)."""
    for _ in range(random.randint(2, 4)):
        await page.evaluate(f"window.scrollBy(0, {random.randint(200, 700)})")
        await asyncio.sleep(random.uniform(0.4, 0.9))
    if random.random() < 0.6:  # occasionally scroll back up like a human re-reading
        await page.evaluate(f"window.scrollBy(0, {-random.randint(120, 400)})")
        await asyncio.sleep(random.uniform(0.3, 0.7))


async def looks_blocked(page):
    """Detect an access-verification interstitial, not a genuine no-results page."""
    try:
        title = (await page.title() or "").lower()
        body  = (await page.evaluate(
            "() => (document.body && document.body.innerText || '').slice(0, 4000)"
        ) or "").lower()
    except Exception:
        return False
    blob = title + " " + body
    return any(m in blob for m in BLOCK_MARKERS)


async def looks_stale(page):
    """Detect Trip.com's recoverable stale-fare interstitial."""
    try:
        title = (await page.title() or "").lower()
        body  = (await page.evaluate(
            "() => (document.body && document.body.innerText || '').slice(0, 4000)"
        ) or "").lower()
    except Exception:
        return False
    blob = title + " " + body
    return any(m in blob for m in STALE_MARKERS)


async def click_reload(page):
    """Find and click a reload/retry button on the stale-fare interstitial.
    Try role=button by name, then visible text, case-insensitive. Return True if clicked."""
    for label in RELOAD_LABELS:
        try:
            btn = page.get_by_role("button", name=label, exact=False)
            if await btn.count() > 0:
                await btn.first.click(timeout=5000)
                return True
        except Exception:
            pass
        try:
            txt = page.get_by_text(label, exact=False)
            if await txt.count() > 0:
                await txt.first.click(timeout=5000)
                return True
        except Exception:
            pass
    return False


async def _warmup_inner(page):
    log("  WARMUP: visiting trip.com homepage to establish a session...")
    # domcontentloaded (NOT networkidle \u2014 trip.com ads/trackers never go idle) + hard cap
    await page.goto("https://www.trip.com/", wait_until="domcontentloaded", timeout=12000)
    # Warmup only needs to set session cookies, not browse \u2014 keep it light.
    await human_pause(1, 2)
    await human_mouse(page, n=2)


async def warmup(page):
    """Best-effort bounded homepage visit to establish required session state.

    Capped at 15 seconds overall;
    any stall/error is logged and we proceed to scraping anyway."""
    try:
        await asyncio.wait_for(_warmup_inner(page), timeout=15)
        log("  WARMUP done (session cookies set).")
    except asyncio.TimeoutError:
        log("  WARMUP skipped (timeout 15s) \u2014 proceeding to flight pages anyway.")
    except Exception as e:
        log(f"  WARMUP failed (continuing anyway): {str(e)[:80]}")


async def scrape_one(page, origin, dest, flight_date, days_until, session_id, tag=""):
    url  = build_url(origin, dest, flight_date)
    rows = []

    try:
        # Techniques 2-4: domcontentloaded (don't race the JS challenge with networkidle),
        # human dwell + mouse + up/down scroll BEFORE reading flights, then poll for
        # .J_FlightItem OR an access-verification page, with bounded backoff and retry.
        loaded = False
        for attempt in range(MAX_BLOCK_RETRY + 1):
            # goto resilient: a slow parallel load can time out \u2014 retry the goto ONCE
            # (capped) before giving up on this query, so one slow load doesn't skip a route.
            goto_ok = False
            for g in range(2):
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT)
                    goto_ok = True
                    break
                except Exception as e:
                    log(f"{tag}  GOTO timeout/err {origin}->{dest} {flight_date} "
                        f"(goto try {g+1}/2): {str(e)[:60]}")
                    await asyncio.sleep(random.uniform(2, 5))
            if not goto_ok:
                log(f"{tag}  GOTO FAILED {origin}->{dest} {flight_date} \u2014 skip this query")
                return rows

            await human_pause(2, 4)          # bounded delay for client rendering
            await human_mouse(page, n=2)
            await human_scroll(page)

            current_url = page.url
            if "error" in current_url or "404" in current_url:
                log(f"{tag}  ERROR {origin}->{dest} {flight_date} \u2014 redirected to {current_url}")
                return rows

            try:
                await page.wait_for_selector(".J_FlightItem", timeout=SELECTOR_BUDGET * 1000)
                loaded = True
                break
            except Exception:
                if await looks_blocked(page):
                    backoff = random.uniform(30, 90)
                    log(f"{tag}  ACCESS VERIFICATION {origin}->{dest} {flight_date} "
                        f"(attempt {attempt+1}/{MAX_BLOCK_RETRY+1}) \u2014 backoff {backoff:.0f}s then reload")
                    await asyncio.sleep(backoff)
                    await human_mouse(page, n=3)
                    continue
                elif await looks_stale(page):
                    # Stale-fare interstitial ("gi\u00e1 \u0111\u00e3 c\u0169") \u2014 click reload and re-poll.
                    for k in range(1, MAX_STALE_RETRY + 1):
                        clicked = await click_reload(page)
                        log(f"{tag}  STALE FARE \u2192 clicked 'T\u1ea3i l\u1ea1i', retry ({k}/{MAX_STALE_RETRY}) "
                            f"{origin}->{dest} {flight_date}" + ("" if clicked else "  (no button found)"))
                        await human_pause(1, 3)
                        try:
                            await page.wait_for_selector(".J_FlightItem", timeout=15000)
                            loaded = True
                            break
                        except Exception:
                            if not await looks_stale(page):
                                break   # no longer stale but no flights \u2192 fall through
                    if loaded:
                        break
                    log(f"{tag}  STALE GIVE UP {origin}->{dest} {flight_date} \u2014 still stale after {MAX_STALE_RETRY} tries")
                    return rows
                else:
                    log(f"{tag}  NO DATA {origin}->{dest} {flight_date} \u2014 no flight elements (not a block)")
                    return rows

        if not loaded:
            log(f"{tag}  GIVE UP {origin}->{dest} {flight_date} \u2014 still blocked after {MAX_BLOCK_RETRY+1} tries")
            return rows

        # Round-based: scroll group \u2192 extract \u2192 click "show more" \u2192 repeat
        # Trip.com loads one airline group at a time; clicking replaces (not appends) content.
        seen_keys  = set()
        all_items  = []
        max_rounds = 6              # was 10 \u2014 most queries finish far sooner once hasNonNonstop hits

        for rnd in range(max_rounds):
            # Fast-scroll through the current airline group
            await _scroll_fast(page, steps=4, step_px=900)
            await asyncio.sleep(0.6)

            # Extract flights visible right now
            extracted   = await page.evaluate(_EXTRACT_JS)
            batch       = extracted["results"]
            reached_end = extracted["hasNonNonstop"]  # non-nonstop card appeared \u2192 boundary hit
            for item in batch:
                key = (item["airline_code"], item["flight_no"], item["dep_time"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_items.append(item)

            if reached_end:
                break

            # Scroll to absolute bottom and look for "show more" button
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)

            btn_text = await page.evaluate(_FIND_MORE_JS)
            if btn_text:
                # Use Playwright native click (real mouse event \u2014 JS .click() gets ignored)
                try:
                    locator = page.get_by_text(btn_text, exact=False)
                    await locator.first.scroll_into_view_if_needed()
                    await locator.first.click(timeout=5000)
                except Exception:
                    pass
                await asyncio.sleep(1.2)        # was 2.5 \u2014 post-click wait
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(0.4)
            else:
                break

        # Pin scraped_at to the session timestamp (NOT datetime.now()) so a long run that
        # crosses midnight keeps one scrape-date for every row \u2014 same as session_id.
        scraped_at = session_id.strftime("%Y-%m-%d %H:%M:%S")
        for item in all_items:
            if item["price_vnd"] and item["price_vnd"] > 15_000_000:
                continue
            rows.append({
                "scraped_at"          : scraped_at,
                "session_id"          : str(session_id),
                "origin"              : origin,
                "dest"                : dest,
                "route"               : f"{origin}-{dest}",
                "days_until_departure": days_until,
                "flight_date"         : flight_date,
                "airline"             : item["airline_code"],
                "airline_name"        : item["airline_name"],
                "flight_no"           : item["flight_no"],
                "departure_time"      : f"{flight_date} {item['dep_time']}:00",
                "price_usd"           : round(item["price_vnd"] / USD_TO_VND, 2) if item["price_vnd"] else None,
                "price_vnd"           : item["price_vnd"] or None,
                "data_source"         : "trip_com",
                "seats_left"          : None,
                "is_soldout"          : item["is_soldout"],
            })

    except Exception as e:
        log(f"{tag}  ERR {origin}->{dest} {flight_date}: {str(e)[:100]}")

    return rows


# ======================================================
# MAIN
# ======================================================

async def worker(wid, page, tasks, session_id, today, write_lock, totals, start_delay=0):
    """One tab: process its slice of tasks. Independent of the other workers \u2014 its
    backoff/cool-down never blocks them. Writes to the shared CSV under write_lock.
    start_delay staggers startup so 3 tabs don't hit the network at the same instant."""
    tag = f"[W{wid}]"
    if start_delay:
        log(f"{tag} staggered start in {start_delay}s")
        await asyncio.sleep(start_delay)
    n            = len(tasks)

    # Per-worker warm-up: each tab visits the homepage on ITS OWN page so every page
    # establishes required session state before route queries.
    await warmup(page)
    log(f"{tag} warmup done, scraping {n} tasks")
    rows_total   = 0
    skipped      = 0
    consec_empty = 0

    for i, (origin, dest, days) in enumerate(tasks, 1):
        flight_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")

        rows = await scrape_one(page, origin, dest, flight_date, days, session_id, tag=tag)

        if rows:
            # Option A: serialize CSV writes so 3 tabs never interleave/corrupt rows
            async with write_lock:
                save_rows(rows)
            rows_total  += len(rows)
            consec_empty = 0
            airlines_found = sorted({r["airline"] for r in rows})
            log(f"{tag} [{i:3d}/{n}] OK   {origin}->{dest} +{days:2d}d  {len(rows):3d} flights  {airlines_found}")
        else:
            skipped      += 1
            consec_empty += 1
            log(f"{tag} [{i:3d}/{n}] SKIP {origin}->{dest} +{days:2d}d  no results  (consec={consec_empty})")
            # If stuck on many in a row, this worker cools down \u2014 others keep running
            if consec_empty >= W_CONSEC_BLOCK_LIMIT:
                log(f"{tag} {consec_empty} consecutive empty/blocked \u2192 cool-down {W_CONSEC_BLOCK_COOLDOWN}s")
                await asyncio.sleep(W_CONSEC_BLOCK_COOLDOWN)
                consec_empty = 0

        # Short per-tab cadence (long-break-every-12 removed for speed)
        if i < n:
            await asyncio.sleep(random.uniform(W_DELAY_MIN, W_DELAY_MAX))

    totals[wid] = (rows_total, skipped)
    log(f"{tag} DONE  rows={rows_total:,}  skipped={skipped}/{n}")


async def main():
    session_id = datetime.now().replace(minute=0, second=0, microsecond=0)
    today      = datetime.today()

    # Full task list (window-major so consecutive tasks are different routes \u2192 round-robin
    # spreads routes across workers, and --test3 picks 3 different routes).
    all_tasks = [(o, d, days) for days in BOOKING_WINDOWS for (o, d) in ROUTES]

    # Round-robin distribute across N_WORKERS: worker i gets tasks i, i+3, i+6, ...
    worker_tasks = [all_tasks[i::N_WORKERS] for i in range(N_WORKERS)]

    if TEST_MODE:                       # legacy: 1 query, 1 tab
        worker_tasks = [all_tasks[:1], [], []]
        log("*** TEST_MODE *** 1 query, 1 tab")
    elif TEST3_MODE:                    # 3 queries, 3 tabs (one task each)
        worker_tasks = [wt[:1] for wt in worker_tasks]
        log("*** TEST3_MODE *** 3 queries across 3 tabs (one per tab)")

    total = sum(len(wt) for wt in worker_tasks)
    log(f"START  queries={total}  workers={N_WORKERS}  output={output_file}")
    log(f"START session={session_id} queries={total}")
    log(f"CAMOUFOX geoip={'enabled' if GEOIP_ENABLED else 'disabled'}")

    # Technique 6/7: ONE browser (humanize/locale/os fingerprint), multiple pages.
    async with AsyncCamoufox(
        headless=False,
        geoip=GEOIP_ENABLED,
        humanize=True,
        locale="en-US",
        os=["windows", "macos"],
    ) as browser:
        pages = []
        for k in range(N_WORKERS):
            if not worker_tasks[k]:
                pages.append(None)
                continue
            p = await browser.new_page()
            try:
                await p.set_viewport_size({"width": 1280, "height": 800})
            except Exception:
                pass
            pages.append(p)

        # Warm-up moved INTO worker(): each tab warms up its OWN page (per-session cookies),
        # offset by the stagger so the two homepage loads don't collide.
        # Concurrent CSV write guard (Option A)
        write_lock = asyncio.Lock()
        totals = {}

        coros = [
            worker(k, pages[k], worker_tasks[k], session_id, today, write_lock, totals,
                   start_delay=k * W_STAGGER_STEP)
            for k in range(N_WORKERS) if worker_tasks[k]
        ]
        await asyncio.gather(*coros)

    total_rows = sum(t[0] for t in totals.values())
    total_skip = sum(t[1] for t in totals.values())
    log(f"END    rows={total_rows:,}  skipped={total_skip}/{total}  (combined across {len(totals)} workers)")
    log(f"END rows={total_rows} skipped={total_skip}/{total}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("STOP interrupted by user")
    except Exception as exc:
        # Camoufox can replace KeyboardInterrupt with this close-time driver
        # error after Ctrl+C. Report a clean interrupted exit instead.
        if "Browser.close: Connection closed while reading from the driver" in str(exc):
            log("STOP browser driver closed during shutdown")
            raise SystemExit(130) from None
        raise
