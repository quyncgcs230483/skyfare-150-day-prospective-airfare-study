SkyFare 150-Day Prospective Airfare Study
=========================================

This repository contains reproducible source, frozen contracts, raw
observations, evaluation code, and compact evidence for a 150-calendar-day
longitudinal airfare study. Model development used the first 128 days. Two
non-overlapping, prospectively sealed 11-day blocks were then evaluated in
chronological order. A final production refit used all observations available
through 19 August 2026 without reopening candidate selection.

Scientific scope
----------------

Four linked prediction tasks are represented:

* classification of whether the next canonical booking window has a fare drop
  of at least five per cent;
* point regression of future fare relative to a strictly prior hierarchical
  anchor;
* probabilistic forecasting through seven conditional fare quantiles;
* within-query learning-to-rank evaluated by Normalised Discounted Cumulative
  Gain at rank five.

A BUY/WAIT decision policy is evaluated as downstream utility. It remains a
guarded research policy; the production default is BUY because prospective
regret confidence intervals did not establish superiority.

Repository organisation
-----------------------

``src/skyfare/acquisition`` contains Fli and Trip.com collectors.
``src/skyfare/preparation`` standardises daily files and builds temporal data
contracts. ``src/skyfare/features`` contains strictly prior feature logic.
``src/skyfare/models`` contains candidate and finalist implementations.
``src/skyfare/evaluation`` contains prospective and pooled metrics.
``src/skyfare/production`` contains immutable final-refit and inference logic.
``src/skyfare/serving`` contains PostgreSQL ingestion and snapshot operations.
``application/skyfare_inference_demo`` contains the Reflex research interface.
``experiments`` records every material research branch under scientific names.
``artifacts/evidence`` contains compact machine-readable verification evidence.

Data layout
-----------

Collectors write one CSV per collection date to ``data/raw/fli`` and
``data/raw/trip_com``. The separately acquired 9G Google Flights supplement is
preserved in ``data/raw/google_flights_manual_9g``; acquisition issue evidence
is retained in ``data/raw/collection_issues``. Standardised source tables belong in
``data/interim/standardised``. Leakage-controlled training and evaluation
frames belong in ``data/processed``. Generated outputs belong under
``artifacts``. All locations can be overridden through documented environment
variables and are resolved from repository root, never from a host-specific path.

Collection-source terminology is canonical: ``FLI_LIBRARY_ERA`` denotes Google Flights
collected through the Fli Python library; ``TRIP_COM_BROWSER_ERA`` denotes Trip.com
collected through Playwright and Camoufox. The confirmatory study ends on
19 August 2026. Observations from 20--24 August are excluded from model fitting
and prospective evaluation; they support only the post-freeze serving snapshot.
Raw collection evidence spans 21 March--24 August 2026. The first two dates are
pre-study acquisition records; 23 March--19 August define the frozen 150-day
study, and 20--24 August provide label-free inference inputs only.

Reproduction
------------

Install package and quality dependencies. Add ``gpu``, ``collection-fli`` or
``collection-trip`` extras only on hosts that execute those stages::

    python -m pip install -e ".[quality]"
    python -m pip install -e ".[gpu,collection-fli,collection-trip,figures]"

Inspect resolved paths and frozen study contract::

    python -m skyfare paths
    python -m skyfare contract

Execute individual stages through stable repository-rooted commands::

    scripts/collect_fli.sh
    scripts/collect_trip_com.sh
    scripts/prepare_development_data.sh
    scripts/run_feature_audit.sh
    scripts/run_development_selection.sh
    scripts/run_prospective_test_one.sh
    scripts/run_prospective_test_two.sh
    scripts/run_pooled_evaluation.sh
    scripts/run_final_production_refit.sh
    scripts/build_result_figures.sh

Prospective and production stages are restartable. Existing artefacts are
accepted only when their code and content hashes satisfy the relevant frozen
contract. Test 2 consumes Test 1 history at its permitted temporal boundary;
the pooled stage consumes both immutable result archives without retraining.

Post-freeze inference and application
-------------------------------------

Build a verified prediction snapshot for booking date 24 August while retaining
the production training cutoff of 19 August. This command verifies and downloads
the frozen model archive when absent, standardises all raw observations through
24 August, creates target-free features, scores all frozen models, and publishes
an atomic application snapshot::

    python -m pip install -e ".[gpu]"
    scripts/build_serving_snapshot.sh

The snapshot gate rejects post-cutoff labels, a changed model cutoff, invalid
DUD values, or an incomplete model manifest. Canonical models support DUD
``1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60``; the application interpolates only
between adjacent canonical endpoints so requests remain within DUD 1--60.

Create the Reflex environment and start the research interface::

    scripts/setup_application.sh
    application/skyfare_inference_demo/run_local.sh

The booking-date selector defaults to the latest verified raw observation,
24 August 2026. PostgreSQL storage is optional for persisted collection audit
and can be initialised after standardisation::

    python -m pip install -e ".[serving]"
    python -m skyfare.serving.manage_live_store init-schema
    python -m skyfare.serving.manage_live_store bootstrap-standard --through-date 2026-08-24

Release artefacts
-----------------

Large immutable result and model archives are published under release tag
``study-results-2026-08``. Download and verify all assets::

    python tools/fetch_release_assets.py

Download only inputs needed for pooled prospective evaluation::

    python tools/fetch_release_assets.py \
      --filename SKYFARE_TEST_1_EVALUATION_RESULTS_R2.tar.gz \
      --filename SKYFARE_TEST_2_EVALUATION_RESULTS_R1.tar.gz

Archive filenames preserve signed historical identifiers. Public source paths,
module names and documentation use scientific task names; immutable internal
identifiers are decoded in ``configs/model_identifier_registry.json``.

Run deterministic local gates::

    python -m pytest
    python tools/verify_repository.py

Large immutable result and model archives are not stored in Git history. Their
SHA-256 records and compact verification reports remain in
``artifacts/evidence`` and ``artifacts/release_manifests`` so repository history
stays reviewable.

Development EDA and source compatibility
----------------------------------------

Five report figures preserve EDA for the 128 observed development-history days.
The underlying raw span is 21 March--28 July 2026: 21--22 March provide
pre-study context, the registered development window starts on 23 March, and 8--9
May remain explicit uncollected dates. These figures are separate from the
150-day prospective-study evaluation and from the target-free 20--24 August
serving demonstration.

During the 19-day Fli--Trip.com overlap, 150,538 matched route, airline, flight
date, departure-minute, booking-window and session slots had a median
Trip.com-to-Fli minimum-fare ratio of 1.0249. Of these matches, 93.62% were within
+/-5% and 98.46% were within +/-10%. This supports early-period listed-price
comparability. It does not identify source and time effects separately, bound
tail differences, or verify later June/July display semantics.

Collection responsibility
-------------------------

Collection code is retained for reproducibility of the completed academic
study. Each run uses a finite route-by-booking-window task grid, bounded
inter-query delays, bounded navigation retries and a two-minute cool-down after
repeated empty or access-verification responses. Access-verification markers
trigger backoff and eventual termination for the affected query; the collector
does not solve CAPTCHA challenges. Published raw evidence contains structured
fare observations and checksums rather than page HTML, browser profiles,
credentials or session cookies. Reuse remains subject to applicable website
terms, robots directives, institutional ethics requirements and local law.
Historical ``price_usd`` values use a fixed 26,309 VND/USD display conversion;
``price_vnd`` is the canonical value used by every analytical and serving path.

Limitations
-----------

The data originate from two collection mechanisms whose coverage changes over
time. Source identity is therefore an explicit covariate and audit dimension.
The overlap analysis reduces concern about large early-period listed-price
incompatibility but cannot prove that later temporal movement is source
independent or causal market drift.
Prospective Test 1 and Test 2 are separate temporal blocks, not cross-validation
folds. No post-test candidate search is represented as confirmatory evidence.
