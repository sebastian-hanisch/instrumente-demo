"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import pytest

import ins_evaluation as ev
import ins_visualization as V


def test_diagram_chart_bars_are_the_shares_of_the_counts():
    res = ev.diagram_experiment(diagrams=300, nodes=7)
    y = list(V.build_diagrams_chart(res).data[0].y)
    assert len(y) == 5 and y[0] == 0.0 and y[1] == pytest.approx(100 * res["instrument"] / 300) and y[2] == pytest.approx(100 * res["frontdoor"] / 300)
    assert y[3] == pytest.approx(100 * res["both"] / 300) and y[4] == pytest.approx(y[1] + y[2] - y[3])


def test_estimate_figures_and_tables_build_for_every_setting():
    for kw in (dict(), dict(g=0.3), dict(h=0.3), dict(g=0.2, h=0.2), dict(pi=0.05), dict(pi=0.1, n=100), dict(a=0.0)):
        a = ev.analyse(ev.Settings(**kw))
        assert V.build_diagram(a.settings) is not None and V.build_estimates_chart(a) is not None
        assert len(V.build_estimates_table(a)) == 4 and len(V.build_first_stage_table(a)) == 3
        V.style_estimates_table(V.build_estimates_table(a))


def test_the_diagram_shows_violation_arrows_only_when_active():
    def arrows(s):
        return sum(1 for ann in V.build_diagram(s).layout.annotations if ann.showarrow)

    base = arrows(ev.Settings())
    assert arrows(ev.Settings(g=0.1)) == base + 1 and arrows(ev.Settings(h=0.1)) == base + 1 and arrows(ev.Settings(g=0.1, h=0.1)) == base + 2


def test_experiment_figures_and_tables_build():
    rows = ev.base_experiment(pi_values=(1.0, 0.1, 0.05), n=200, reps=40)
    assert V.build_base_chart(rows) is not None and V.build_weak_chart(rows) is not None and len(V.build_base_table(rows)) == 3 and len(V.build_interval_table(rows)) == 3
    viol = ev.violations_experiment()
    assert V.build_violations_chart(viol) is not None and len(V.build_violations_table(viol)) == len(viol)
    over = ev.overid_experiment(n=300, reps=30)
    assert V.build_overid_chart(over) is not None and len(V.build_overid_table(over)) == len(over)
    late = ev.late_experiment(ns=(300,), reps=30)
    assert V.build_late_chart(late) is not None and len(V.build_late_table(late)) == 3 and len(V.build_late_runs_table(late)) == 1
