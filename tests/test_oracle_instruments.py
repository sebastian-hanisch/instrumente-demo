"""Orakel: geschlossene Formeln für die Grenzwerte, die erste Stufe und die Deckung der Intervalle, unabhängig von den Schätzern gerechnet."""

import itertools

import numpy as np
import pytest

import ins_constants as C
import ins_estimators as E
import ins_world as W
from kausal_generator import SplitMix64


@pytest.mark.parametrize("a,pi,g,h", list(itertools.product((0.5, 1.0, 2.0), (0.3, 0.5, 1.0), (0.0, 0.1, 0.3), (0.0, 0.1, 0.3))))
def test_iv_limit_equals_beta_plus_the_closed_form_bias(a, pi, g, h):
    """Var Z = 1 + h², Cov(Z, X) = π (1 + h²) + a h, Cov(Z, Y) = β Cov(Z, X) + g (1 + h²) + a h: IV = β + (g (1 + h²) + a h) / Cov(Z, X)."""
    p = E.population_estimates(W.World(a=a, pi=pi, beta=1.3, g=g, h=h))
    assert p["iv"] == pytest.approx(1.3 + (g * (1 + h * h) + a * h) / (pi * (1 + h * h) + a * h))


@pytest.mark.parametrize("a,pi", list(itertools.product((0.5, 1.0, 2.0), (0.1, 0.5, 1.0))))
def test_ols_and_control_limits_for_a_valid_instrument(a, pi):
    p = E.population_estimates(W.World(a=a, pi=pi, beta=0.8))
    assert p["naive"] == pytest.approx(0.8 + a * a / (pi * pi + a * a + 1)) and p["control"] == pytest.approx(0.8 + a * a / (a * a + 1))
    assert p["iv"] == pytest.approx(0.8) and p["oracle"] == pytest.approx(0.8)


def test_control_limit_does_not_depend_on_the_instrument_strength():
    assert len({round(E.population_estimates(W.World(pi=pi))["control"], 9) for pi in (0.05, 0.3, 0.5, 1.5)}) == 1


def test_iv_error_over_pi_is_the_exclusion_bias_amplification():
    for pi in (0.1, 0.2, 0.5, 1.0):
        assert E.population_estimates(W.World(pi=pi, g=0.1))["iv"] - 1.0 == pytest.approx(0.1 / pi)


def test_expected_f_matches_the_simulated_mean_f():
    w = W.World(pi=0.3)
    v = W.simulate(w, 500, SplitMix64(20), reps=300)
    f = [E.tsls(v[r, :, W.X], [v[r, :, W.Z]], v[r, :, W.Y])["f"] for r in range(300)]
    assert np.mean(f) == pytest.approx(E.expected_f(w, 500), rel=0.12)


def test_standard_error_matches_the_empirical_spread_and_the_interval_covers():
    v = W.simulate(W.World(pi=1.0), 500, SplitMix64(21), reps=400)
    out = [E.tsls(v[r, :, W.X], [v[r, :, W.Z]], v[r, :, W.Y]) for r in range(400)]
    est, se = np.array([o["estimate"] for o in out]), np.array([o["se"] for o in out])
    assert np.mean(se) == pytest.approx(np.std(est), rel=0.12) and 0.92 <= np.mean(np.abs(est - 1.0) <= C.Z_CRIT * se) <= 0.985
    assert np.median(est) == pytest.approx(1.0, abs=0.02)


def test_the_variance_ratio_of_iv_to_the_oracle_is_the_inverse_squared_correlation():
    """Var(β̂_IV) ≈ σ² / (n Var(X) corr(Z, X)²), das Orakel: σ² / (n Var(X | U)); das Verhältnis der Streuungen ist 1 / corr(Z, X) · sqrt(Var(X | U) / Var X)."""
    w = W.World(pi=1.0)
    cov = W.population(w)[0]
    corr = cov[W.Z, W.X] / np.sqrt(cov[W.Z, W.Z] * cov[W.X, W.X])
    v = W.simulate(w, 500, SplitMix64(22), reps=400)
    iv = [E.tsls(v[r, :, W.X], [v[r, :, W.Z]], v[r, :, W.Y])["estimate"] for r in range(400)]
    orc = [E.oracle(v[r, :, W.X], v[r, :, W.U], v[r, :, W.Y])[0] for r in range(400)]
    predicted = np.sqrt(cov[W.X, W.X] / (cov[W.X, W.X] - cov[W.U, W.X] ** 2)) / corr
    assert np.std(iv) / np.std(orc) == pytest.approx(predicted, rel=0.15) and predicted > 1.9


def test_anderson_rubin_covers_even_with_a_weak_instrument():
    w = W.World(pi=0.1)
    v = W.simulate(w, 500, SplitMix64(23), reps=300)
    hits = [E.ar_contains(E.anderson_rubin(v[r, :, W.X], v[r, :, W.Z], v[r, :, W.Y]), 1.0) for r in range(300)]
    assert 0.92 <= np.mean(hits) <= 0.98


def test_the_j_test_keeps_its_size_with_valid_instruments_and_is_blind_to_equal_violations():
    n, reps = 500, 400
    def reject(**kw):
        v = W.simulate(W.World(**kw), n, SplitMix64(24), reps)
        out = [E.tsls(v[r, :, W.X], [v[r, :, W.Z], v[r, :, W.Z2]], v[r, :, W.Y]) for r in range(reps)]
        return np.mean([E.sargan(o["resid"], o["zc"]) > C.CHI2_CRIT for o in out]), np.median([o["estimate"] for o in out])
    ok, _ = reject(pi=0.7, pi2=0.7)
    one, est_one = reject(pi=0.7, pi2=0.7, g2=0.3)
    both, est_both = reject(pi=0.7, pi2=0.7, g=0.3, g2=0.3)
    assert 0.02 <= ok <= 0.09 and one >= 0.85 and 0.02 <= both <= 0.09
    assert est_one == pytest.approx(1.0 + 0.3 / (2 * 0.7), abs=0.02)                                                    # eines ungültig: der Fehler ist g₂ / (π + π₂)
    assert est_both == pytest.approx(1.0 + 0.3 / 0.7, abs=0.02)                                                         # beide gleich: Fehler g / π, der J-Test sieht ihn nicht


def test_wald_estimator_in_the_binary_example_converges_to_the_late_and_not_the_ate():
    p = W.late_population()
    _, z, x, y = (a[0] for a in W.simulate_late(400000, SplitMix64(25)))
    wald = E.wald_binary(z, x, y)
    assert wald == pytest.approx(p["late"], abs=0.01) and abs(wald - p["ate"]) > 0.2
