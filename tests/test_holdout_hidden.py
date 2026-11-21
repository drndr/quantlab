"""rnd never runs holdout periods (periods.score_exclude) unless --with-holdout."""

from __future__ import annotations

from pathlib import Path

import yaml

from quantlab.rnd import _tuning_periods

CFG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "trend_based" / "current.yaml").read_text(encoding="utf-8")
)


def test_three_block_tuning_skips_the_2026_holdout() -> None:
    assert [p[0] for p in _tuning_periods(CFG, with_holdout=False)] == ["2020-2022", "2023-2025"]
    assert [p[0] for p in _tuning_periods(CFG, with_holdout=True)] == ["2020-2022", "2023-2025", "2026"]


def test_yearly_tuning_skips_the_2026_holdout() -> None:
    cfg = {**CFG, "periods": {**CFG["periods"], "mode": "yearly"}}
    names = [p[0] for p in _tuning_periods(cfg, with_holdout=False)]
    assert "2026" not in names and names[:2] == ["2020", "2021"]
