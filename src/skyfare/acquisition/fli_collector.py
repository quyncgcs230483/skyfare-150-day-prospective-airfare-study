"""Collect Google Flights observations through the Fli Python library.

Output defaults to ``data/raw/fli/YYYY-MM-DD.csv`` relative to repository root.
Set ``SKYFARE_FLI_OUTPUT_DIR`` to override it.

Notes:
    - Run between 17:00 - 21:00 to avoid window 1-2 errors
    - Estimated runtime: ~45-60 minutes
    - On HTTP 429 rate limit: script auto-waits and retries
"""

import csv
import os
import random
import time
from datetime import datetime, timedelta

from fli.models import (
    Airport,
    FlightSearchFilters,
    FlightSegment,
    PassengerInfo,
    SeatType,
    TripType,
)
from fli.search import SearchFlights

from skyfare.core.paths import DataLayout

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# CONFIG
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

LAYOUT = DataLayout.resolve()
OUTPUT_DIR = LAYOUT.raw_fli
LOG_DIR = LAYOUT.collection_logs

# Reporting-only conversion retained for historical CSV compatibility.
# All analytical and serving contracts use price_vnd as the canonical value.
DISPLAY_VND_PER_USD_FIXED = 26_309

AIRLINE_NAMES = {
    "VN": "Vietnam Airlines",
    "VJ": "Vietjet Air",
    "QH": "Bamboo Airways",
    "VU": "Vietravel Airlines",
    "BL": "Pacific Airlines",
}

ROUTES = [
    ("SGN", "HAN"), ("HAN", "SGN"),
    ("SGN", "DAD"), ("DAD", "SGN"),
    ("HAN", "DAD"), ("DAD", "HAN"),
    ("SGN", "CXR"), ("CXR", "SGN"),
    ("SGN", "PQC"), ("PQC", "SGN"),
    ("HAN", "PQC"), ("PQC", "HAN"),
    ("DAD", "PQC"), ("PQC", "DAD"),
    ("HAN", "CXR"), ("CXR", "HAN"),
    ("SGN", "HPH"), ("HPH", "SGN"),
    ("HAN", "VCA"), ("VCA", "HAN"),
]

BOOKING_WINDOWS = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1]

FIELDNAMES = [
    "scraped_at", "session_id",
    "origin", "dest", "route",
    "days_until_departure", "flight_date",
    "airline", "airline_name",
    "flight_no", "departure_time",
    "price_usd", "price_vnd",
    "data_source",
]

ALLOWED_AIRLINES = {"VN","VJ","QH","VU"}

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# SETUP
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)


today_str   = datetime.today().strftime("%Y-%m-%d")
output_file = os.path.join(OUTPUT_DIR, f"{today_str}.csv")
log_file    = os.path.join(LOG_DIR,    f"fli_{today_str}.log")

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# HELPERS
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_rows(rows, write_header):
    """Append rows to CSV. Write header only on first save."""
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# MAIN
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

def run():
    session_id = datetime.now().replace(minute=0, second=0, microsecond=0)
    today      = datetime.today()
    total      = len(ROUTES) * len(BOOKING_WINDOWS)
    done = errors = total_rows = 0
    start_time = datetime.now()

    print(f"""
+--------------------------------------------------------------+
|       VIETNAM FLIGHT PRICE SCRAPER - fli pipeline            |
+--------------------------------------------------------------+
  Date    : {today_str}
  Output  : {output_file}
  Queries : {total}  ({len(ROUTES)} routes x {len(BOOKING_WINDOWS)} windows)
  Est.    : ~{total * 11 // 60} minutes
  Session : {session_id}
""")

    log(f"START session={session_id} queries={total}")

    # Always append \u2014 never delete morning data when running evening scrape
    write_header = not os.path.exists(output_file)
    if not write_header:
        log("Appending to existing file (append mode)")

    buffer = []

    for origin, dest in ROUTES:
        for days in BOOKING_WINDOWS:
            flight_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")

            for attempt in range(3):
                try:
                    filters = FlightSearchFilters(
                        trip_type=TripType.ONE_WAY,
                        passenger_info=PassengerInfo(adults=1),
                        flight_segments=[FlightSegment(
                            departure_airport=[[getattr(Airport, origin), 0]],
                            arrival_airport  =[[getattr(Airport, dest),   0]],
                            travel_date=flight_date
                        )],
                        seat_type=SeatType.ECONOMY
                    )

                    results = SearchFlights().search(filters)
                    count   = 0

                    for r in results:
                        # Skip flights with missing or zero price
                        if not r.price or r.price <= 0:
                            continue
                        for leg in r.legs:
                            code = leg.airline.name
                            if code not in ALLOWED_AIRLINES:
                                continue
                            buffer.append({
                                "scraped_at"          : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "session_id"          : str(session_id),
                                "origin"              : origin,
                                "dest"                : dest,
                                "route"               : f"{origin}-{dest}",
                                "days_until_departure": days,
                                "flight_date"         : flight_date,
                                "airline"             : code,
                                "airline_name"        : AIRLINE_NAMES.get(code, leg.airline.value),
                                "flight_no"           : leg.flight_number,
                                "departure_time"      : str(leg.departure_datetime),
                                "price_usd"           : round(r.price / DISPLAY_VND_PER_USD_FIXED, 2),
                                "price_vnd"           : round(r.price),
                                "data_source"         : "fli_library",
                            })
                            count += 1

                    done       += 1
                    total_rows += count
                    print(f"  [{done:3d}/{total}] OK  {origin}->{dest}  +{days:2d}d  ({flight_date})  {count} rows")
                    break

                except Exception as e:
                    err = str(e)
                    if "429" in err:
                        wait = 20 + attempt * 15
                        print(f"  [{done+1:3d}/{total}] 429  {origin}->{dest} +{days}d  waiting {wait}s (attempt {attempt+1}/3)")
                        time.sleep(wait)
                        if attempt == 2:
                            done   += 1
                            errors += 1
                            print(f"  [{done:3d}/{total}] SKIP {origin}->{dest} +{days}d  skipped after 3 retries")
                            log(f"SKIP {origin}->{dest} +{days}d after 3 retries")
                    else:
                        done   += 1
                        errors += 1
                        print(f"  [{done:3d}/{total}] ERR  {origin}->{dest} +{days}d  {err[:70]}")
                        log(f"ERROR {origin}->{dest} +{days}d: {err[:100]}")
                        break

            time.sleep(random.uniform(8, 14))

        # Flush buffer to CSV after each route \u2014 prevents data loss on crash
        if buffer:
            save_rows(buffer, write_header)
            write_header = False
            buffer       = []
            print(f"  >> Saved  {origin}->{dest}  ({len(BOOKING_WINDOWS)} windows)")

    duration = (datetime.now() - start_time).seconds

    print(f"""
+--------------------------------------------------------------+
|  DONE!
|  Rows     : {total_rows:,}
|  Errors   : {errors}/{total}
|  Time     : {duration // 60}m {duration % 60}s
|  File     : {output_file}
+--------------------------------------------------------------+
""")
    log(f"END rows={total_rows} errors={errors}/{total} duration={duration}s")


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# ENTRY POINT
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

if __name__ == "__main__":
    run()
