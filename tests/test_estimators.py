"""Schätzer: Regression, 2SLS, erste Stufe, Anderson-Rubin, J-Test, Population und der binäre Wald-Schätzer; an den Handrechnungen aus conftest.py."""

import numpy as np
import pytest

import ins_constants as C
import ins_estimators as E
import ins_world as W
from kausal_generator import SplitMix64


def test_ols_against_numpy_lstsq():
    rng = SplitMix64(1)
    x1, x2 = rng.normal(80), rng.normal(80)
    y = 1.0 + 2.0 * x1 - x2 + 0.5 * rng.normal(80)
    beta, cov = E.ols(y, [x1, x2])
    d = np.column_stack([np.ones(80), x1, x2])
    ref, res, *_ = np.linalg.lstsq(d, y, rcond=None)
    assert beta == pytest.approx(ref) and cov == pytest.approx(np.linalg.inv(d.T @ d) * res[0] / (80 - 3))


def test_naive_control_and_oracle_on_exact_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([1.0, 0.0, 1.0, 0.0])
    y = 2.0 * x + z                                                                 # exakt linear: kein Rest
    assert E.naive(x, y)[0] == pytest.approx(np.polyfit(x, y, 1)[0])
    assert E.control_instrument(x, z, y)[0] == pytest.approx(2.0) and E.oracle(x, z, y)[0] == pytest.approx(2.0)


def test_tsls_by_hand(four_units):
    z, x, y = four_units
    t = E.tsls(x, [z], y)
    assert t["estimate"] == pytest.approx(2.0) and t["se"] == pytest.approx(np.sqrt(2.0)) and t["f"] == pytest.approx(1.0) and t["pi"] == pytest.approx([0.5])
    assert t["resid"] == pytest.approx([0.5, -0.5, -0.5, 0.5]) and t["zc"][:, 0] == pytest.approx([-0.5, -0.5, 0.5, 0.5])


def test_one_instrument_is_the_ratio_of_covariances():
    rng = SplitMix64(2)
    z = rng.normal(200)
    x = 0.6 * z + rng.normal(200)
    y = 1.5 * x + rng.normal(200)
    ratio = np.cov(z, y)[0, 1] / np.cov(z, x)[0, 1]
    assert E.tsls(x, [z], y)["estimate"] == pytest.approx(ratio)


def test_two_instruments_against_the_explicit_formula():
    rng = SplitMix64(3)
    z1, z2 = rng.normal(150), rng.normal(150)
    x = 0.7 * z1 + 0.4 * z2 + rng.normal(150)
    y = 0.9 * x + rng.normal(150)
    t = E.tsls(x, [z1, z2], y)
    zc = np.column_stack([z1 - z1.mean(), z2 - z2.mean()])
    xh = zc @ np.linalg.solve(zc.T @ zc, zc.T @ (x - x.mean()))
    assert t["estimate"] == pytest.approx((xh @ (y - y.mean())) / (xh @ xh)) and t["pi"] == pytest.approx(np.linalg.solve(zc.T @ zc, zc.T @ (x - x.mean())))
    e = (y - y.mean()) - t["estimate"] * (x - x.mean())
    assert t["se"] == pytest.approx(np.sqrt(e @ e / 148 / (xh @ xh))) and t["resid"] == pytest.approx(e)
    rss1 = ((x - x.mean()) - xh) @ ((x - x.mean()) - xh)
    assert t["f"] == pytest.approx((xh @ xh / 2) / (rss1 / (150 - 3)))


def test_first_stage_f_is_the_squared_t_statistic_for_one_instrument():
    rng = SplitMix64(4)
    z = rng.normal(100)
    x = 0.3 * z + rng.normal(100)
    y = x + rng.normal(100)
    beta, cov = E.ols(x, [z])
    assert E.tsls(x, [z], y)["f"] == pytest.approx((beta[1] / np.sqrt(cov[1, 1])) ** 2)


def test_sargan_by_hand():
    zc = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])            # zwei unkorrelierte Instrumente
    assert E.sargan(zc[:, 0].copy(), zc) == pytest.approx(4.0)                      # Reste = erstes Instrument: R² = 1, J = n
    assert E.sargan(np.array([1.0, -1.0, -1.0, 1.0]), zc) == pytest.approx(0.0, abs=1e-12)           # Reste orthogonal zu beiden Instrumenten
    assert E.sargan(0.5 * zc[:, 0] + 0.5 * zc[:, 1], zc) == pytest.approx(4.0)       # in der Spanne der Instrumente: R² = 1


def _t2(x, z, y, beta0):
    beta, cov = E.ols(y - beta0 * x, [z])
    return float(beta[1] ** 2 / cov[1, 1])


def test_anderson_rubin_boundaries_satisfy_the_test_equation():
    rng = SplitMix64(5)
    z = rng.normal(300)
    x = 0.6 * z + rng.normal(300)
    y = x + rng.normal(300)
    ar = E.anderson_rubin(x, z, y)
    est = E.tsls(x, [z], y)["estimate"]
    assert ar["kind"] == "interval" and ar["lo"] < est < ar["hi"] and E.ar_bounded(ar) and E.ar_contains(ar, est)
    assert _t2(x, z, y, ar["lo"]) == pytest.approx(C.CHI2_CRIT, rel=1e-6) and _t2(x, z, y, ar["hi"]) == pytest.approx(C.CHI2_CRIT, rel=1e-6)
    assert _t2(x, z, y, est) == pytest.approx(0.0, abs=1e-9) and _t2(x, z, y, 0.5 * (ar["lo"] + ar["hi"])) < C.CHI2_CRIT
    assert not E.ar_contains(ar, ar["hi"] + 0.01) and not E.ar_contains(ar, ar["lo"] - 0.01)


def test_anderson_rubin_is_unbounded_for_an_irrelevant_instrument():
    rng = SplitMix64(6)
    z = rng.normal(100)
    x = rng.normal(100)                                                             # Z bewegt X nicht
    y = x + rng.normal(100)
    x = x - (np.cov(z, x)[0, 1] / np.cov(z)) * (z - z.mean())                       # empirische Kovarianz mit Z genau 0
    ar = E.anderson_rubin(x, z, y)
    assert ar["kind"] in ("outside", "all") and not E.ar_bounded(ar) and E.ar_contains(ar, 123.0) and E.ar_contains(ar, -50.0)


def test_anderson_rubin_is_the_whole_line_for_a_hopeless_instrument():
    """Bei einer fast irrelevanten Z gibt es keine Nullstelle der Ungleichung: kein β₀ wird abgelehnt."""
    kinds = set()
    for seed in range(200):
        rng = SplitMix64(1000 + seed)
        z, x, e = rng.normal(40), rng.normal(40), rng.normal(40)
        y = x + e
        kinds.add(E.anderson_rubin(x, z, y)["kind"])
    assert "all" in kinds and "interval" in kinds


def test_anderson_rubin_does_not_depend_on_the_scale_of_the_instrument():
    """Die Menge hängt nur von der Spanne des Instruments ab: Z mit 0,001 multipliziert gibt dieselbe Art von Menge mit denselben Grenzen (kleine q₂ dürfen die Art nicht kippen)."""
    found = set()
    for seed in range(200):
        rng = SplitMix64(2000 + seed)
        z, e = rng.normal(40), rng.normal(40)
        x = 0.15 * z + rng.normal(40)
        y = x + e
        a, b = E.anderson_rubin(x, z, y), E.anderson_rubin(x, 0.001 * z, y)
        assert a["kind"] == b["kind"], seed
        if a["kind"] != "all":
            assert b["lo"] == pytest.approx(a["lo"], rel=1e-6) and b["hi"] == pytest.approx(a["hi"], rel=1e-6)
        found.add(a["kind"])
    assert found == {"interval", "outside", "all"}


def test_ar_contains_all_kinds():
    assert E.ar_contains({"kind": "all", "lo": -np.inf, "hi": np.inf}, 5.0) and not E.ar_bounded({"kind": "all", "lo": -np.inf, "hi": np.inf})
    out = {"kind": "outside", "lo": 1.0, "hi": 2.0}
    assert E.ar_contains(out, 0.5) and E.ar_contains(out, 2.5) and not E.ar_contains(out, 1.5) and not E.ar_bounded(out)
    assert E.ar_contains(out, 1.0) and E.ar_contains(out, 2.0)                                      # die Grenzen gehören zur Menge (t² = Schwelle wird nicht abgelehnt)
    inside = {"kind": "interval", "lo": 1.0, "hi": 2.0}
    assert E.ar_contains(inside, 1.5) and E.ar_contains(inside, 1.0) and E.ar_contains(inside, 2.0) and not E.ar_contains(inside, 2.5)


def test_population_estimates_by_hand(unit_world):
    p = E.population_estimates(unit_world)
    assert p["truth"] == pytest.approx(1.0) and p["naive"] == pytest.approx(4 / 3) and p["control"] == pytest.approx(1.5) and p["iv"] == pytest.approx(1.0)
    assert p["oracle"] == pytest.approx(1.0) and p["pi"] == pytest.approx(1.0)


def test_coefficients_solve_the_normal_equations():
    cov = np.array([[3.0, 1.0, 4.0], [1.0, 1.0, 1.0], [4.0, 1.0, 7.0]])
    assert E.coefficients(cov, 2, [0, 1]) == pytest.approx([1.5, -0.5])             # [[3, 1], [1, 1]] β = (4, 1)


def test_expected_f_by_hand(unit_world):
    assert E.expected_f(unit_world, 500) == pytest.approx(250.0)                    # R² = 1/3: n R² / (1 − R²) = 500 · ½
    assert E.expected_f(unit_world, 100) == pytest.approx(50.0)


def test_iv_is_unbiased_and_ols_is_not_in_a_large_sample(unit_world):
    v = W.simulate(unit_world, 200000, SplitMix64(10))[0]
    x, y, z, u = v[:, W.X], v[:, W.Y], v[:, W.Z], v[:, W.U]
    assert E.naive(x, y)[0] == pytest.approx(4 / 3, abs=0.02) and E.control_instrument(x, z, y)[0] == pytest.approx(1.5, abs=0.02)
    assert E.tsls(x, [z], y)["estimate"] == pytest.approx(1.0, abs=0.02) and E.oracle(x, u, y)[0] == pytest.approx(1.0, abs=0.02)


def test_wald_binary_by_hand(binary_sample):
    z, x, y = binary_sample
    assert E.wald_binary(z, x, y) == pytest.approx(0.5) and E.naive_binary(x, y) == pytest.approx(0.25)


def test_wald_binary_returns_none_without_a_first_stage_or_a_group():
    z = np.array([0, 0, 1, 1])
    assert E.wald_binary(z, np.array([1, 1, 1, 1]), np.array([0, 1, 0, 1])) is None             # x folgt z nicht: Nenner 0
    assert E.wald_binary(np.array([1, 1, 1, 1]), np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) is None
    assert E.wald_binary(np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) is None
    assert E.naive_binary(np.array([1, 1, 1]), np.array([0, 1, 1])) is None and E.naive_binary(np.array([0, 0, 0]), np.array([0, 1, 1])) is None


def test_wald_binary_recovers_the_late_in_a_large_sample():
    _, z, x, y = (a[0] for a in W.simulate_late(400000, SplitMix64(11)))
    assert E.wald_binary(z, x, y) == pytest.approx(0.5, abs=0.01) and E.naive_binary(x, y) == pytest.approx(W.late_population()["naive"], abs=0.006)
