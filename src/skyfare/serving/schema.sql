BEGIN;

CREATE SCHEMA IF NOT EXISTS skyfare_live;

CREATE TABLE IF NOT EXISTS skyfare_live.schema_versions (
    version             TEXT PRIMARY KEY,
    applied_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skyfare_live.collection_batches (
    batch_id            TEXT PRIMARY KEY,
    collection_era      TEXT NOT NULL CHECK (
        collection_era IN ('FLI_LIBRARY_ERA', 'TRIP_COM_BROWSER_ERA')
    ),
    session_date        DATE NOT NULL,
    session_label       TEXT NOT NULL CHECK (session_label IN ('AM', 'PM')),
    source_file         TEXT NOT NULL,
    source_sha256       CHAR(64) NOT NULL,
    status              TEXT NOT NULL CHECK (
        status IN ('STAGING', 'VALIDATED', 'READY', 'FAILED')
    ),
    started_at          TIMESTAMP,
    completed_at        TIMESTAMP,
    promoted_at         TIMESTAMPTZ,
    standard_rows       INTEGER NOT NULL DEFAULT 0 CHECK (standard_rows >= 0),
    route_count         INTEGER NOT NULL DEFAULT 0 CHECK (route_count >= 0),
    airline_count       INTEGER NOT NULL DEFAULT 0 CHECK (airline_count >= 0),
    dud_count           INTEGER NOT NULL DEFAULT 0 CHECK (dud_count >= 0),
    route_airline_count INTEGER NOT NULL DEFAULT 0 CHECK (
        route_airline_count >= 0
    ),
    validation_report   JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_reason      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (collection_era, session_date, session_label)
);

CREATE TABLE IF NOT EXISTS skyfare_live.standard_offers (
    row_key             CHAR(16) PRIMARY KEY,
    batch_id            TEXT NOT NULL REFERENCES skyfare_live.collection_batches(
        batch_id
    ) ON DELETE CASCADE,
    scraped_at          TIMESTAMP NOT NULL,
    session_id          TIMESTAMP NOT NULL,
    session_date        DATE NOT NULL,
    session_label       TEXT NOT NULL CHECK (session_label IN ('AM', 'PM')),
    origin              TEXT NOT NULL,
    dest                TEXT NOT NULL,
    route               TEXT NOT NULL,
    days_until_departure SMALLINT NOT NULL CHECK (
        days_until_departure BETWEEN 1 AND 60
    ),
    flight_date         DATE NOT NULL,
    airline             TEXT NOT NULL,
    airline_name        TEXT,
    flight_no           TEXT,
    departure_time      TIMESTAMP NOT NULL,
    departure_hhmm      SMALLINT NOT NULL CHECK (
        departure_hhmm BETWEEN 0 AND 2359
    ),
    schedule_slot_key   TEXT NOT NULL,
    price_vnd           BIGINT NOT NULL CHECK (price_vnd > 0),
    data_source         TEXT,
    collection_era      TEXT NOT NULL CHECK (
        collection_era IN ('FLI_LIBRARY_ERA', 'TRIP_COM_BROWSER_ERA')
    ),
    source_file         TEXT NOT NULL,
    UNIQUE (batch_id, schedule_slot_key)
);

CREATE TABLE IF NOT EXISTS skyfare_live.route_airline_registry (
    route               TEXT NOT NULL,
    airline             TEXT NOT NULL,
    first_seen          DATE NOT NULL,
    last_seen           DATE NOT NULL,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (route, airline)
);

CREATE TABLE IF NOT EXISTS skyfare_live.serving_snapshots (
    snapshot_id         TEXT PRIMARY KEY,
    status              TEXT NOT NULL CHECK (
        status IN ('BUILDING', 'READY', 'FAILED', 'SUPERSEDED')
    ),
    data_cutoff         DATE NOT NULL,
    latest_batch_id     TEXT NOT NULL REFERENCES skyfare_live.collection_batches(
        batch_id
    ),
    model_version       TEXT NOT NULL,
    snapshot_path       TEXT NOT NULL,
    offers_rows         BIGINT NOT NULL DEFAULT 0,
    source_manifest_sha256 CHAR(64),
    offers_sha256       CHAR(64),
    fare_sha256         CHAR(64),
    ledger_sha256       CHAR(64),
    validation_report   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    promoted_at         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS skyfare_live.inference_audit (
    inference_id        BIGSERIAL PRIMARY KEY,
    requested_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    snapshot_id         TEXT NOT NULL REFERENCES skyfare_live.serving_snapshots(
        snapshot_id
    ),
    model_version       TEXT NOT NULL,
    booking_date        DATE NOT NULL,
    session_label       TEXT NOT NULL CHECK (session_label IN ('AM', 'PM')),
    route               TEXT NOT NULL,
    flight_date         DATE NOT NULL,
    airline_filter      TEXT,
    result_count        INTEGER NOT NULL CHECK (result_count >= 0),
    latency_ms          NUMERIC(12, 3),
    request_payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_summary    JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Migrate older stores to direct standard-only ingestion. Source CSVs remain
-- immutable and content-addressed by collection_batches.source_sha256.
DROP VIEW IF EXISTS skyfare_live.latest_ready_batch;

-- Only the admitted standard population is stored and counted in PostgreSQL.
ALTER TABLE skyfare_live.collection_batches
    DROP COLUMN IF EXISTS raw_rows,
    DROP COLUMN IF EXISTS source_rows;

ALTER TABLE skyfare_live.serving_snapshots
    DROP COLUMN IF EXISTS fare_rows;

UPDATE skyfare_live.collection_batches
SET validation_report = validation_report - 'source_rows'
WHERE validation_report ? 'source_rows';

ALTER TABLE skyfare_live.standard_offers
    DROP CONSTRAINT IF EXISTS standard_offers_row_key_fkey;

DROP TABLE IF EXISTS skyfare_live.raw_offers;

-- Legacy inventory fields were never available consistently from FLI or Trip
-- and are not part of any feature or serving contract.
ALTER TABLE skyfare_live.standard_offers
    DROP COLUMN IF EXISTS seats_left,
    DROP COLUMN IF EXISTS is_soldout;

CREATE INDEX IF NOT EXISTS idx_live_batches_ready
    ON skyfare_live.collection_batches (status, session_date, session_label);
CREATE INDEX IF NOT EXISTS idx_live_standard_batch
    ON skyfare_live.standard_offers (batch_id);
CREATE INDEX IF NOT EXISTS idx_live_standard_query
    ON skyfare_live.standard_offers (
        session_date, session_label, route, flight_date, airline
    );
CREATE INDEX IF NOT EXISTS idx_live_standard_slot
    ON skyfare_live.standard_offers (schedule_slot_key, session_date);
CREATE INDEX IF NOT EXISTS idx_live_snapshots_ready
    ON skyfare_live.serving_snapshots (status, data_cutoff DESC);

CREATE OR REPLACE VIEW skyfare_live.latest_ready_batch AS
SELECT *
FROM skyfare_live.collection_batches
WHERE status = 'READY'
ORDER BY session_date DESC, CASE session_label WHEN 'PM' THEN 1 ELSE 0 END DESC
LIMIT 1;

CREATE OR REPLACE VIEW skyfare_live.latest_ready_snapshot AS
SELECT *
FROM skyfare_live.serving_snapshots
WHERE status = 'READY'
ORDER BY data_cutoff DESC, promoted_at DESC NULLS LAST
LIMIT 1;

INSERT INTO skyfare_live.schema_versions (version)
VALUES ('SKYFARE_LIVE_POSTGRES_V1')
ON CONFLICT (version) DO NOTHING;

INSERT INTO skyfare_live.schema_versions (version)
VALUES ('SKYFARE_LIVE_POSTGRES_V2_DROP_LEGACY_INVENTORY')
ON CONFLICT (version) DO NOTHING;

INSERT INTO skyfare_live.schema_versions (version)
VALUES ('SKYFARE_POSTGRES_SOURCE_TAXONOMY_V3')
ON CONFLICT (version) DO NOTHING;

INSERT INTO skyfare_live.schema_versions (version)
VALUES ('SKYFARE_LIVE_POSTGRES_V3_DIRECT_STANDARD_INGEST')
ON CONFLICT (version) DO NOTHING;

INSERT INTO skyfare_live.schema_versions (version)
VALUES ('SKYFARE_LIVE_POSTGRES_V4_STANDARD_ROWS_ONLY')
ON CONFLICT (version) DO NOTHING;

INSERT INTO skyfare_live.schema_versions (version)
VALUES ('SKYFARE_LIVE_POSTGRES_V5_OFFERS_ROWS_ONLY')
ON CONFLICT (version) DO NOTHING;

COMMIT;
