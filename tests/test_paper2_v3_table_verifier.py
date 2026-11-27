"""End-to-end deterministic table & checkpoint audit test for Paper 2 v2.3."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_paper2_v3_tables import run_verification


def test_paper2_v3_checkpoints_and_table_ledger_reconcile() -> None:
    res = run_verification(verbose=False)
    assert res["status"] == "VERIFIED_OK"
    assert res["total_encounters_reconciled"] == 600154
