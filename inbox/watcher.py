"""
Inbox folder watcher — the Phase 2 trigger.

Polls inbox/pending/ for new email bundle directories. When a bundle appears,
it passes it to inbox/processor.py, writes a result JSON, then archives the
bundle to inbox/done/ (or inbox/failed/ on error).

Each bundle is a sub-directory containing:
    email.json          — sender metadata, customer_id
    <doc1>.pdf          — Bill of Lading, Invoice, Packing List, etc.
    <doc2>.pdf
    ...

Usage
-----
Run continuously (watches until Ctrl-C):
    python inbox/watcher.py
    python -m inbox.watcher

Process the current queue once and exit (useful for CI / testing):
    python inbox/watcher.py --once

Environment variables
---------------------
    INBOX_PENDING_DIR      default: inbox/pending
    INBOX_DONE_DIR         default: inbox/done
    INBOX_FAILED_DIR       default: inbox/failed
    INBOX_RESULTS_DIR      default: inbox/results
    WATCHER_POLL_INTERVAL  seconds between scans (default: 5)
    NOVA_DB_PATH           path to DuckDB file (default: app.duckdb)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inbox.processor import process_bundle
from storage.db import DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nova.watcher")


# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------

PENDING_DIR  = Path(os.getenv("INBOX_PENDING_DIR",  "inbox/pending"))
DONE_DIR     = Path(os.getenv("INBOX_DONE_DIR",     "inbox/done"))
FAILED_DIR   = Path(os.getenv("INBOX_FAILED_DIR",   "inbox/failed"))
RESULTS_DIR  = Path(os.getenv("INBOX_RESULTS_DIR",  "inbox/results"))
POLL_INTERVAL = int(os.getenv("WATCHER_POLL_INTERVAL", "5"))
DB_PATH      = os.getenv("NOVA_DB_PATH", DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Bundle processing (safe wrapper)
# ---------------------------------------------------------------------------


def _process_bundle_safe(bundle_path: Path) -> bool:
    """
    Process one bundle directory, catching all exceptions.

    On success:
        - Writes ``RESULTS_DIR/{bundle.name}.json``
        - Moves bundle to ``DONE_DIR/{bundle.name}``
        - Returns True

    On failure:
        - Writes ``error.json`` inside the bundle dir
        - Moves bundle to ``FAILED_DIR/{bundle.name}``
        - Returns False
    """
    bundle_name = bundle_path.name
    log.info("Processing bundle: %s", bundle_name)

    try:
        result = process_bundle(bundle_path, db_path=DB_PATH)

        # Write result JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_file = RESULTS_DIR / f"{bundle_name}.json"
        result_file.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

        # Archive bundle
        DONE_DIR.mkdir(parents=True, exist_ok=True)
        dest = DONE_DIR / bundle_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(bundle_path), str(DONE_DIR))

        outcome = result.get("decision", {}).get("outcome", "unknown")
        storage_id = result.get("storage_id", "—")
        log.info(
            "  Done: %s  |  outcome=%s  |  storage_id=%s",
            bundle_name,
            outcome,
            storage_id,
        )
        return True

    except Exception:
        error_text = traceback.format_exc()
        log.error("  Failed: %s\n%s", bundle_name, error_text)

        # Write error file into the bundle so it's inspectable
        try:
            error_payload: dict[str, Any] = {
                "bundle": bundle_name,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "traceback": error_text,
            }
            (bundle_path / "error.json").write_text(
                json.dumps(error_payload, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # Don't mask the original error

        # Move to failed/
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        dest = FAILED_DIR / bundle_name
        if dest.exists():
            shutil.rmtree(dest)
        try:
            shutil.move(str(bundle_path), str(FAILED_DIR))
        except Exception:
            log.warning("  Could not move failed bundle to %s", FAILED_DIR)

        return False


# ---------------------------------------------------------------------------
# Watcher loop
# ---------------------------------------------------------------------------


def watch(once: bool = False) -> None:
    """
    Poll PENDING_DIR for new bundle directories and process each one.

    Parameters
    ----------
    once:
        If True, process all currently pending bundles then return.
        If False (default), loop forever until interrupted.
    """
    for directory in (PENDING_DIR, DONE_DIR, FAILED_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if once:
        log.info("Running in --once mode (processing current queue then exiting)")
    else:
        log.info(
            "Watcher started — polling %s every %ds  (Ctrl-C to stop)",
            PENDING_DIR,
            POLL_INTERVAL,
        )

    while True:
        bundles = sorted(
            p for p in PENDING_DIR.iterdir() if p.is_dir()
        )

        if bundles:
            log.info("Found %d bundle(s) to process", len(bundles))
            for bundle in bundles:
                _process_bundle_safe(bundle)
        elif once:
            log.info("No bundles found in %s — nothing to do", PENDING_DIR)

        if once:
            break

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nova Phase 2 inbox watcher — polls inbox/pending/ for new "
            "SU email bundles and runs the multi-doc validation pipeline."
        )
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help=(
            "Process all bundles currently in inbox/pending/ then exit. "
            "Without this flag the watcher loops until Ctrl-C."
        ),
    )
    parser.add_argument(
        "--pending-dir",
        default=None,
        help="Override INBOX_PENDING_DIR environment variable.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override NOVA_DB_PATH environment variable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Allow CLI overrides
    global PENDING_DIR, DB_PATH
    if args.pending_dir:
        PENDING_DIR = Path(args.pending_dir)
    if args.db_path:
        DB_PATH = args.db_path

    try:
        watch(once=args.once)
    except KeyboardInterrupt:
        log.info("Watcher stopped by user.")


if __name__ == "__main__":
    main()
