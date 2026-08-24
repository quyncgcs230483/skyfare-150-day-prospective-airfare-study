SkyFare 150-Day Prospective Airfare Study
=========================================

This repository contains the reproducible source, frozen contracts, evaluation
code, and compact evidence for a longitudinal airfare study covering 150
consecutive collection days. Model development used the first 128 days. Two
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
``experiments`` records every material research branch under scientific names.
``artifacts/evidence`` contains compact machine-readable verification evidence.

Data layout
-----------

Collectors write one CSV per collection date to ``data/raw/fli`` and
``data/raw/trip_com``. Standardised source tables belong in
``data/interim/standardised``. Leakage-controlled training and evaluation
frames belong in ``data/processed``. Generated outputs belong under
``artifacts``. All locations can be overridden through documented environment
variables and are resolved from repository root, never from ``/workspace``.

Collection-source terminology is canonical: ``FLI_LIBRARY_ERA`` denotes Google Flights
collected through the Fli Python library; ``TRIP_COM_BROWSER_ERA`` denotes Trip.com
collected through Playwright and Camoufox. ``SERPAPI_ERA`` and
``TRIP_DAILY_ERA`` are accepted only when migrating immutable historical
artifacts and are normalised before analysis.

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

Limitations
-----------

The data originate from two collection mechanisms whose coverage changes over
time. Source identity is therefore an explicit covariate and audit dimension.
Prospective Test 1 and Test 2 are separate temporal blocks, not cross-validation
folds. No post-test candidate search is represented as confirmatory evidence.
