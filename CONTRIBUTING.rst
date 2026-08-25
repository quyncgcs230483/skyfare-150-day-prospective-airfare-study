Contributing
============

Changes must preserve temporal legality, immutable prospective predictions,
and compatibility with signed release archives. Candidate selection may not
use either prospective test block. New source names must use the canonical
taxonomy in ``skyfare.core.sources``.

Before proposing a change, run::

    python -m compileall -q src tests tools
    python -m pytest
    python tools/verify_repository.py

Raw daily CSV evidence is tracked with Git LFS. Generated aggregates, model
binaries, Parquet files, and result archives do not belong in ordinary Git
history. Publish immutable large artefacts as release assets and record their
SHA-256 digest in ``configs/release_assets.json``.
