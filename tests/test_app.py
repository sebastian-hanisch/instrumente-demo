"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import ins_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("a_slider", "pi_slider", "beta_slider", "g_slider", "h_slider", "n_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_truth():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahre Wirkung"] == "1,00" and m["Naiver Vergleich"].startswith("+1,") and m["2SLS"].startswith("+1,") and m["Kontrolle für das Instrument"].startswith("+1,")
    assert at.slider(key="a_slider").value == C.DEFAULT_A and at.slider(key="pi_slider").value == C.DEFAULT_PI and at.number_input(key="seed_input").value == C.DEFAULT_SEED
    assert any("Der 2SLS schätzt" in s.value for s in list(at.success) + list(at.warning) + list(at.info))


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        for key, state_key in (("a", "a_slider"), ("pi", "pi_slider"), ("beta", "beta_slider"), ("g", "g_slider"), ("h", "h_slider"), ("n", "n_slider")):
            assert at.slider(key=state_key).value == preset[key], (name, key)


def test_a_violated_assumption_shows_the_warning():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Ausschluss verletzt")].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Eine Annahme ist verletzt" in w.value and "direkten Pfeil" in w.value for w in at.warning)


def test_a_weak_instrument_shows_the_warning():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Schwaches Instrument")].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Das Instrument ist schwach" in w.value for w in at.warning)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["a"] = "9"
    at.query_params["pi"] = "0.02"
    at.query_params["g"] = "0.44"
    at.query_params["n"] = "50"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="a_slider").value == C.A_MAX and at.slider(key="pi_slider").value == C.PI_MIN and at.slider(key="g_slider").value == 0.4
    assert at.slider(key="n_slider").value == C.N_MIN and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="pi_slider").set_value(0.25).run(timeout=TIMEOUT)
    assert at.query_params["pi"] in ("0.25", ["0.25"])
