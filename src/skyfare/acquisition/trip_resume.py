"""Resume missing Trip.com observations using Camoufox.

The daily CSV is read from and written to ``data/raw/trip_com`` by default.

Scrape all five airlines (VN, VJ, QH, VU, 9G) from Trip.com using Camoufox.
Trip.com shows all airlines together per route+date query \u2192 1 search = all airlines.

Routes  : 20 routes x 11 windows = 220 queries/day
Schema  : shared raw-offer schema (data_source = "trip_com")

Install:
    pip install camoufox[geoip]
    python -m camoufox fetch

Run:
    python scrape_trip_resume.py
"""

import argparse
import asyncio
import csv
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
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

BOOKING_WINDOWS = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1]   # full set (tham chi\u1ebfu)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume only missing Trip.com route/window queries for one scrape session."
    )
    parser.add_argument(
        "--date",
        default=os.environ.get("RESUME_DATE", datetime.today().strftime("%Y-%m-%d")),
        help="Scrape date whose daily CSV should be resumed (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("RESUME_SESSION_ID"),
        help=("Exact existing session timestamp, e.g. '2026-07-19 09:00:00'. "
              "Default: latest session_id found in that day's CSV."),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RESUME_WORKERS", "1")),
        help="Number of browser tabs. Default 1 for lower block risk.",
    )
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=int(os.environ.get("RESUME_RETRY_ROUNDS", "3")),
        help="Maximum passes over unresolved queries. Default 3.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=int(os.environ.get("RESUME_RETRY_WAIT", "90")),
        help="Seconds between passes, useful for transient DNS/network outages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing queries without opening a browser or writing data.",
    )
    parser.add_argument("--test", action="store_true", help="Run only first missing query.")
    parser.add_argument("--test3", action="store_true", help="Run only first 3 missing queries.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.retry_rounds < 1:
        parser.error("--retry-rounds must be >= 1")
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        parser.error("--date must use YYYY-MM-DD")
    return args


ARGS = parse_args()
SESSION_DATE_OVERRIDE = ARGS.date

# \u2500\u2500 TEST MODES \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#   --test   : 1 query, 1 tab (verify Whaleguard bypass)
#   --test3  : 3 queries, 3 tabs concurrently (verify parallel machinery + CSV safety)
TEST_MODE  = ARGS.test or (os.environ.get("TEST_MODE") == "1")
TEST3_MODE = ARGS.test3 or (os.environ.get("TEST3_MODE") == "1")

# \u2500\u2500 Anti-Whaleguard tuning \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
BLOCK_MARKERS    = ["whaleguard", "verify", "captcha", "robot check",
                    "are you a human", "unusual traffic", "access denied",
                    "security check", "blocked"]
# Stale-price / soft-error interstitial (NOT a Whaleguard block) \u2014 Trip shows this + a "T\u1ea3i l\u1ea1i" button.
STALE_MARKERS    = ["t\u1ea3i l\u1ea1i", "\u0111\u00e3 c\u00f3 l\u1ed7i x\u1ea3y ra", "vui l\u00f2ng th\u1eed", "th\u1eed l\u1ea1i",
                    "gi\u00e1 \u0111\u00e3 c\u0169", "reload", "try again", "something went wrong"]
RELOAD_LABELS    = ["T\u1ea3i l\u1ea1i", "T\u1ea3i l\u1ea1i trang", "Th\u1eed l\u1ea1i", "Reload", "Try again", "Retry"]
MAX_STALE_RETRY  = 3            # click-reload attempts when a stale-price interstitial appears
MAX_BLOCK_RETRY  = 2            # extra reload attempts when a block is detected
SELECTOR_BUDGET  = 25           # seconds to wait for flights OR block (faster than full's 30)

# \u2500\u2500 3-tab parallel tuning (balanced cadence \u2014 3 tabs run at once) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
N_WORKERS = ARGS.workers
W_DELAY_MIN             = 1          # per-worker inter-query delay lower bound (faster than full's 2)
W_DELAY_MAX             = 3          # per-worker inter-query delay upper bound (faster than full's 6)
W_LONG_PAUSE_EVERY      = 12         # per-worker longer break every N queries
W_LONG_PAUSE_RANGE      = (20, 35)   # seconds for that break
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

_base_date  = datetime.strptime(SESSION_DATE_OVERRIDE, "%Y-%m-%d")
today_str   = _base_date.strftime("%Y-%m-%d")
output_file = os.path.join(OUTPUT_DIR, f"{today_str}.csv")
log_file    = os.path.join(LOG_DIR,    f"trip_all_{today_str}.log")
missing_file = os.path.join(LOG_DIR, f"trip_resume_missing_{today_str}.csv")

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


def _parse_session(value):
    """Normalize timestamps emitted by old and current scrapers."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_progress(requested_session_id=None):
    """Return (session datetime, completed query keys) for this daily CSV.

    Query key is (origin, dest, days_until_departure). Progress is scoped to one
    session_id, so an AM run never masks missing queries in a PM run.
    """
    if not os.path.isfile(output_file):
        raise RuntimeError(
            f"Daily CSV not found: {output_file}. Check --date, or pass --session-id "
            "when resuming a session that has not written any row yet."
        )

    parsed_rows = []
    with open(output_file, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            session = _parse_session(row.get("session_id") or row.get("scraped_at"))
            if session is None or session.strftime("%Y-%m-%d") != today_str:
                continue
            try:
                days = int(float(row["days_until_departure"]))
            except (KeyError, TypeError, ValueError):
                continue
            parsed_rows.append((session, row.get("origin"), row.get("dest"), days))

    if requested_session_id:
        session_id = _parse_session(requested_session_id)
        if session_id is None:
            raise RuntimeError(
                "Invalid --session-id. Use ISO format, e.g. '2026-07-19 09:00:00'."
            )
        if session_id.strftime("%Y-%m-%d") != today_str:
            raise RuntimeError("--session-id date must match --date")
    else:
        sessions = [item[0] for item in parsed_rows]
        if not sessions:
            raise RuntimeError(
                f"No session_id found in {output_file}. Pass --session-id explicitly."
            )
        session_id = max(sessions)

    completed = {
        (origin, dest, days)
        for session, origin, dest, days in parsed_rows
        if session == session_id and origin and dest
    }
    return session_id, completed


def write_missing_report(tasks, session_id):
    """Small auditable report; overwritten after each retry pass."""
    fields = ["session_id", "origin", "dest", "days_until_departure", "flight_date"]
    with open(missing_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for origin, dest, days in tasks:
            writer.writerow({
                "session_id": str(session_id),
                "origin": origin,
                "dest": dest,
                "days_until_departure": days,
                "flight_date": (_base_date + timedelta(days=days)).strftime("%Y-%m-%d"),
            })


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
        await asyncio.sleep(random.uniform(0.15, 0.3))


# \u2500\u2500 Human-like behaviour helpers (anti-Whaleguard) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
async def human_pause(a, b):
    await asyncio.sleep(random.uniform(a, b))


async def human_mouse(page, n=3):
    """Move cursor to a few random points \u2014 Whaleguard flags sessions with no mouse activity."""
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
    """Detect a Whaleguard / verification interstitial (vs a genuine no-results page)."""
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
    """Detect the stale-price / soft-error interstitial (the 'T\u1ea3i l\u1ea1i' reload prompt).
    Distinct from a Whaleguard block \u2014 this is recoverable by clicking reload."""
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
    """Click the 'T\u1ea3i l\u1ea1i' / 'Reload' / 'Try again' button. Return True if a click landed.
    Tries role=button first, then visible text, across label variants (case-insensitive)."""
    for lab in RELOAD_LABELS:
        try:
            loc = page.get_by_role("button", name=lab, exact=False)
            if await loc.count() > 0:
                await loc.first.scroll_into_view_if_needed()
                await loc.first.click(timeout=4000)
                return True
        except Exception:
            pass
    for lab in RELOAD_LABELS:
        try:
            loc = page.get_by_text(lab, exact=False)
            if await loc.count() > 0:
                await loc.first.scroll_into_view_if_needed()
                await loc.first.click(timeout=4000)
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
    """Technique 1 (best-effort, BOUNDED \u2014 must never hang): visit homepage so Whaleguard
    sees a real browsing session before the deep showfarefirst URLs. Capped at 15s overall;
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
        # .J_FlightItem OR an explicit Whaleguard block, with backoff+reload+retry.
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

            await human_pause(1, 3)
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
                # Branch 1: real Whaleguard block \u2192 backoff + outer retry (unchanged)
                if await looks_blocked(page):
                    backoff = random.uniform(30, 90)
                    log(f"{tag}  WHALEGUARD BLOCK {origin}->{dest} {flight_date} "
                        f"(attempt {attempt+1}/{MAX_BLOCK_RETRY+1}) \u2014 backoff {backoff:.0f}s then reload")
                    await asyncio.sleep(backoff)
                    await human_mouse(page, n=3)
                    continue
                # Branch 2: stale-price / soft-error interstitial \u2192 click 'T\u1ea3i l\u1ea1i' and re-poll
                if await looks_stale(page):
                    recovered = False
                    for k in range(1, MAX_STALE_RETRY + 1):
                        clicked = await click_reload(page)
                        if clicked:
                            log(f"{tag}  STALE PRICE \u2192 clicked 'T\u1ea3i l\u1ea1i', retrying ({k}/{MAX_STALE_RETRY})")
                        else:
                            log(f"{tag}  STALE PRICE detected but no reload button found ({k}/{MAX_STALE_RETRY})")
                        await human_pause(1, 3)
                        try:
                            await page.wait_for_selector(".J_FlightItem", timeout=15000)
                            recovered = True
                            break
                        except Exception:
                            if not await looks_stale(page):
                                break   # stale cleared but flights still absent \u2192 not stale anymore
                    if recovered:
                        loaded = True
                        break
                    log(f"{tag}  STALE not cleared after {MAX_STALE_RETRY} reloads {origin}->{dest} {flight_date}")
                    return rows
                # Branch 3: genuine empty page
                log(f"{tag}  NO DATA {origin}->{dest} {flight_date} \u2014 no flight elements (not a block)")
                return rows

        if not loaded:
            log(f"{tag}  GIVE UP {origin}->{dest} {flight_date} \u2014 still blocked after {MAX_BLOCK_RETRY+1} tries")
            return rows

        # Round-based: scroll group \u2192 extract \u2192 click "show more" \u2192 repeat
        # Trip.com loads one airline group at a time; clicking replaces (not appends) content.
        seen_keys  = set()
        all_items  = []
        max_rounds = 6

        for rnd in range(max_rounds):
            # Fast-scroll through the current airline group
            await _scroll_fast(page, steps=4, step_px=900)
            await asyncio.sleep(0.4)

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
            await asyncio.sleep(0.3)

            btn_text = await page.evaluate(_FIND_MORE_JS)
            if btn_text:
                # Use Playwright native click (real mouse event \u2014 JS .click() gets ignored)
                try:
                    locator = page.get_by_text(btn_text, exact=False)
                    await locator.first.scroll_into_view_if_needed()
                    await locator.first.click(timeout=5000)
                except Exception:
                    pass
                await asyncio.sleep(1.0)
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
    # establishes a valid session cookie (fixes cold-session Whaleguard block on W1+).
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

        # Short per-tab cadence (long-break-every-12 removed for speed \u2014 match scrape_trip_all)
        if i < n:
            await asyncio.sleep(random.uniform(W_DELAY_MIN, W_DELAY_MAX))

    totals[wid] = (rows_total, skipped)
    log(f"{tag} DONE  rows={rows_total:,}  skipped={skipped}/{n}")


async def run_pass(all_tasks, session_id, today, pass_number):
    """Run one retry pass in a fresh browser, then return to progress audit."""
    worker_tasks = [all_tasks[i::N_WORKERS] for i in range(N_WORKERS)]
    if TEST_MODE:
        worker_tasks = [all_tasks[:1]] + [[] for _ in range(N_WORKERS - 1)]
        log("*** TEST_MODE *** 1 query")
    elif TEST3_MODE:
        limited = all_tasks[:min(3, len(all_tasks))]
        worker_tasks = [limited[i::N_WORKERS] for i in range(N_WORKERS)]
        log(f"*** TEST3_MODE *** {len(limited)} queries")

    total = sum(len(wt) for wt in worker_tasks)
    log(f"RESUME PASS {pass_number}/{ARGS.retry_rounds}  queries={total} workers={N_WORKERS}")

    async with AsyncCamoufox(
        headless=False,
        geoip=True,
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
    log(f"PASS {pass_number} END rows={total_rows:,} skipped={total_skip}/{total}")


async def main():
    today = _base_date
    expected_tasks = [(o, d, days) for days in BOOKING_WINDOWS for (o, d) in ROUTES]
    session_id, completed = load_progress(ARGS.session_id)
    missing = [task for task in expected_tasks if task not in completed]

    log(f"RESUME START date={today_str} session={session_id} output={output_file}")
    log(f"AUDIT expected={len(expected_tasks)} completed={len(completed)} missing={len(missing)}")
    for origin, dest, days in missing:
        log(f"  MISSING {origin}->{dest} +{days}d")
    write_missing_report(missing, session_id)

    if not missing:
        log("RESUME COMPLETE: all 220 route/window queries already have rows")
        return
    if ARGS.dry_run:
        log(f"DRY RUN: no browser opened; report={missing_file}")
        return

    for pass_number in range(1, ARGS.retry_rounds + 1):
        await run_pass(missing, session_id, today, pass_number)
        _, completed = load_progress(str(session_id))
        missing = [task for task in expected_tasks if task not in completed]
        write_missing_report(missing, session_id)
        log(f"AUDIT AFTER PASS {pass_number}: completed={len(completed)} missing={len(missing)}")
        if not missing:
            log("RESUME COMPLETE: all 220 route/window queries recovered")
            return
        if TEST_MODE or TEST3_MODE:
            log("TEST MODE COMPLETE: stopping after one pass")
            return
        if pass_number < ARGS.retry_rounds:
            log(f"NETWORK/EMPTY RETRY: waiting {ARGS.retry_wait}s before next pass")
            await asyncio.sleep(ARGS.retry_wait)

    log(f"RESUME INCOMPLETE: {len(missing)} queries remain; report={missing_file}")


if __name__ == "__main__":
    asyncio.run(main())
