"""rnd never runs the holdouts (2019, 2026) unless named in --with-holdout; the webapp shows them, unscored."""

from __future__ import annotations

import copy

import pytest
import yaml

from quantlab.paths import QUANTLAB
from quantlab.periods import build_periods, holdout_names
from quantlab.rnd import _tuning_periods

CFG = yaml.safe_load(
    (QUANTLAB / "configs" / "trend_based" / "current.yaml").read_text(encoding="utf-8")
)


YEARLY = {**CFG, "periods": {**CFG["periods"], "mode": "yearly"}}


def _names(cfg: dict, *with_holdout: str) -> list[str]:
    return [p[0] for p in _tuning_periods(cfg, with_holdout=list(with_holdout))]


def test_three_block_tuning_skips_both_holdouts() -> None:
    assert _names(CFG) == ["2020-2022", "2023-2025"]


def test_yearly_tuning_skips_both_holdouts() -> None:
    names = _names(YEARLY)
    assert "2026" not in names and names[:2] == ["2020", "2021"]


def test_holdouts_are_opened_one_by_one() -> None:
    assert _names(CFG, "2026") == ["2020-2022", "2023-2025", "2026"]
    assert _names(CFG, "2019") == ["2020-2022", "2023-2025", "2019"]
    assert _names(CFG, "2019", "2026") == ["2020-2022", "2023-2025", "2019", "2026"]
    with pytest.raises(ValueError, match="Unknown holdout"):
        _names(CFG, "2025")


def test_the_webapp_shows_both_holdouts_unscored() -> None:
    # What the webapp runs, in report order: tuning periods first, then the holdouts.
    assert [p[0] for p in build_periods(CFG)] == ["2020-2022", "2023-2025", "2019", "2026"]
    assert [p[0] for p in build_periods(YEARLY)][:3] == ["2019", "2020", "2021"]
    assert set(holdout_names(CFG)) == {"2019", "2026"}  # what the reports leave out of the header stats


def test_a_period_may_not_overlap_a_holdout_block() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["periods"]["three_block"]["p1"]["start"] = "2019-12-31"
    with pytest.raises(ValueError, match="overlaps the holdout 2019"):
        build_periods(cfg)
