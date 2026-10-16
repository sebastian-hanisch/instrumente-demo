"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import ins_constants as C
import ins_evaluation as ev
import ins_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= p[key] <= spec.hi, (name, key)
            assert P.snap(p[key], spec) == pytest.approx(p[key]), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Starkes Instrument (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(p["a"], p["pi"], p["beta"], p["g"], p["h"], p["n"], p["seed"])
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_exactly_one_setting():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)
    base = C.PRESETS["Starkes Instrument (Voreinstellung)"]
    for name, p in C.PRESETS.items():
        assert len([k for k in p if p[k] != base[k]]) <= 1, name
    assert C.PRESETS["Schwaches Instrument"]["pi"] < C.PRESETS["Grenzfall (F um 10)"]["pi"] < C.DEFAULT_PI
    assert C.PRESETS["Ausschluss verletzt"]["g"] > 0 and C.PRESETS["Instrument mit Confounder verbunden"]["h"] > 0 and C.PRESETS["Starker Confounder"]["a"] > C.DEFAULT_A


def test_the_weak_presets_are_weak_at_the_default_sample_size():
    """Die Hilfetexte sagen F etwa 3 (π = 0,1) und F um 10 (π = 0,2) bei n = 500."""
    import ins_estimators as E
    from ins_world import World

    assert 2.5 < E.expected_f(World(pi=0.1), 500) < 3.5 and 9.0 < E.expected_f(World(pi=0.2), 500) < 11.5 and E.expected_f(World(), 500) > 50


def test_default_instrument_is_not_the_extreme_case():
    """π = 1 wäre ein Grundfall mit F = 250; die Voreinstellung zeigt die doppelte Streuung bei F im Bereich 60 bis 80."""
    assert C.DEFAULT_PI == 0.5 and C.PI_MIN <= 0.05 and C.PI_MAX >= 1.0


def test_snap_clamps_and_rounds_to_the_grid():
    a = P.SETTING_SPECS["a_slider"]
    assert P.snap(5.0, a) == C.A_MAX and P.snap(-1.0, a) == C.A_MIN and P.snap(0.43, a) == pytest.approx(0.4)
    pi = P.SETTING_SPECS["pi_slider"]
    assert P.snap(0.02, pi) == C.PI_MIN and P.snap(9.0, pi) == C.PI_MAX and P.snap(0.27, pi) == pytest.approx(0.25) and P.snap(0.28, pi) == pytest.approx(0.3)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == C.N_MIN and P.snap(10 ** 6, n) == C.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    assert P.snap(999999, P.SETTING_SPECS["seed_input"]) == 999999


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("pi_slider") == (C.PI_MIN, C.PI_MAX)
    assert C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP) and {s.url_param for s in P.SETTING_SPECS.values()} == {"a", "pi", "b", "g", "h", "n", "seed"}


def test_experiment_constants_are_consistent():
    assert C.DEFAULT_PI in C.EXP_PI_VALUES and C.EXP_PI_VALUES == tuple(sorted(C.EXP_PI_VALUES, reverse=True)) and 0.1 in C.EXP_PI_VALUES
    assert C.EXP_REPS >= 200 and C.EXP_OVER_REPS >= 500 and C.EXP_DIAGRAMS >= 1000 and C.EXP_NODES >= 5 and C.EXP_LATE_NS == tuple(sorted(C.EXP_LATE_NS))
    assert C.COEF_LO > 0 and C.COEF_HI > C.COEF_LO and 0 < C.RANDOM_P_LO and C.RANDOM_P_LO + C.RANDOM_P_WIDTH < 1
    assert next(iter(C.VIOLATIONS)).startswith("Grundfall") and next(iter(C.VIOLATIONS.values())) == {} and abs(sum(C.LATE_SHARES) - 1.0) < 1e-12
    assert set(C.LATE_PY) == set(C.LATE_TYPES) and len(C.LATE_SHARES) == len(C.LATE_TYPES) and 0 < C.LATE_PZ < 1


def test_the_late_example_does_not_hide_a_coincidence():
    """Zweig-Test: Der naive Vergleich liegt klar neben dem LATE (die erste Fassung mit Always-Taker 0,5 / 0,6 hatte 0,51 gegen 0,50)."""
    import ins_world as W

    p = W.late_population()
    assert abs(p["naive"] - p["late"]) > 0.1 and abs(p["ate"] - p["late"]) > 0.2
