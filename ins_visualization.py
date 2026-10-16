"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import ins_constants as C
import ins_world as W
from ins_evaluation import ROW_NAMES, Analysis, Settings, ar_text, de, signed

SHORT = {"Naiver Vergleich": "naiv", "Kontrolle für das Instrument": "Kontrolle Z", "2SLS (Instrument)": "2SLS", "Orakel mit U (nicht beobachtbar)": "Orakel mit U"}
COLORS = {"Naiver Vergleich": C.COLOR_NEUTRAL, "Kontrolle für das Instrument": C.COLOR_CONTROL, "2SLS (Instrument)": C.COLOR_GOOD, "Orakel mit U (nicht beobachtbar)": C.COLOR_TRUTH}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _arrow(fig, p0, p1, color, width=2, dash=None, shrink=0.11):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    fig.add_annotation(x=x1 - ux * shrink, y=y1 - uy * shrink, ax=x0 + ux * shrink, ay=y0 + uy * shrink, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                       arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def build_diagram(s: Settings) -> go.Figure:
    """Das Diagramm der Einstellung: verborgener Knoten grau, der Weg Z → X → Y grün, verletzte Annahmen als rote Pfeile (nur wenn der Regler > 0 ist)."""
    pos = {W.U: (0.5, 1.05), W.Z: (0.0, -0.9), W.X: (0.5, 0.0), W.Y: (1.0, 0.0)}
    fig = go.Figure()
    for a, b in ((W.U, W.X), (W.U, W.Y)):
        _arrow(fig, pos[a], pos[b], "#6b6b6b", 2)
    _arrow(fig, pos[W.Z], pos[W.X], C.COLOR_GOOD, 4)
    _arrow(fig, pos[W.X], pos[W.Y], C.COLOR_GOOD, 4)
    fig.add_annotation(x=0.19, y=-0.5, text=f"π = {de(s.pi, 2)}", showarrow=False, font=dict(size=11, color="#555555"), bgcolor="rgba(255,255,255,0.75)")
    fig.add_annotation(x=0.75, y=0.12, text=f"β = {de(s.beta, 1)}", showarrow=False, font=dict(size=11, color="#555555"), bgcolor="rgba(255,255,255,0.75)")
    if s.g > 0:
        _arrow(fig, pos[W.Z], pos[W.Y], C.COLOR_BAD, 3)
    if s.h > 0:
        _arrow(fig, pos[W.U], pos[W.Z], C.COLOR_BAD, 3)
    names = {W.U: "Standortqualität<br>(verborgen)", W.Z: "Anstoß<br>(Z)", W.X: "Regel B<br>(X)", W.Y: "Servicegrad<br>(Y)"}
    for group, is_hidden in (([W.Z, W.X, W.Y], False), ([W.U], True)):
        fig.add_trace(go.Scatter(x=[pos[v][0] for v in group], y=[pos[v][1] for v in group], mode="markers+text", text=[names[v] for v in group], textposition="middle center",
                                 textfont=dict(size=10, color="#1b1b1b") if is_hidden else dict(size=10),
                                 marker=dict(size=74, color=C.COLOR_HIDDEN if is_hidden else "rgba(47,107,101,0.12)", line=dict(width=1.5 if is_hidden else 4, color="#8a8a8a" if is_hidden else "#1b1b1b")),
                                 hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.3, 1.3])
    fig.update_yaxes(visible=False, range=[-1.3, 1.4])
    return _fixed(fig, 360, showlegend=False)


def build_estimates_chart(a: Analysis) -> go.Figure:
    """Schätzung mit 95-%-Intervall je Schätzer (grün: deckt die Wahrheit, rot: deckt nicht); Raute: Grenzwert bei n → ∞."""
    fig = go.Figure()
    for r in a.rows:
        color = C.COLOR_GOOD if r.covers else C.COLOR_BAD
        fig.add_trace(go.Scatter(x=[r.estimate], y=[SHORT[r.name]], mode="markers", marker=dict(color=color, size=10), showlegend=False,
                                 error_x=dict(type="data", array=[r.hi - r.estimate], arrayminus=[r.estimate - r.lo], color=color, thickness=2),
                                 hovertemplate=f"{r.name}: {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}]<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r.limit for r in a.rows], y=[SHORT[r.name] for r in a.rows], mode="markers", marker=dict(color="#1b1b1b", size=8, symbol="diamond"), name="Grenzwert (n → ∞)"))
    fig.add_vline(x=a.truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(a.truth, 2)}", annotation_position="top")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[SHORT[n] for n in ROW_NAMES])
    fig.update_xaxes(title="geschätzte Wirkung der Regel B")
    return _fixed(fig, 320)


def build_estimates_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Schätzer": r.name, "Schätzung": signed(r.estimate, 3), "Fehler": signed(r.error, 3), "Standardfehler": de(r.se, 3), "95-%-Intervall": f"{de(r.lo, 2)} bis {de(r.hi, 2)}",
                          "Deckt": "ja" if r.covers else "nein", "Grenzwert": de(r.limit, 3)} for r in a.rows])


def style_estimates_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Deckt"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_first_stage_table(a: Analysis) -> pd.DataFrame:
    f, s = a.first, a.settings
    weak = f["f"] < C.F_RULE
    return pd.DataFrame([
        {"Größe": "Erste Stufe: Wirkung des Anstoßes auf die Regel (π̂)", "Wert": f"{de(f['pi'], 3)} (Standardfehler {de(f['se_pi'], 3)})", "Grenzwert": de(f["pi_limit"], 3), "wahr": de(s.pi, 2)},
        {"Größe": "Erste-Stufe-F (Faustregel: über 10)", "Wert": de(f["f"], 1) + (" (schwach)" if weak else ""), "Grenzwert": f"etwa {de(f['expected_f'], 1)} erwartet", "wahr": "-"},
        {"Größe": "Anderson-Rubin-Menge (95 %)", "Wert": ar_text(a.ar) + (", deckt die Wahrheit" if a.ar["covers"] else ", deckt die Wahrheit nicht"), "Grenzwert": "-", "wahr": de(a.truth, 2)},
    ])


# ----------------------------------------------------------------------------- Experiment 1

def build_base_chart(rows: list) -> go.Figure:
    fig = go.Figure()
    labels = [f"π = {de(r['pi'], 2)}" for r in rows]
    for name, key in (("Naiver Vergleich", "ols"), ("2SLS (Instrument)", "iv"), ("Orakel mit U (nicht beobachtbar)", "oracle")):
        fig.add_trace(go.Bar(x=labels, y=[r[key]["median_bias"] for r in rows], name=SHORT[name], marker_color=COLORS[name],
                             error_y=dict(type="data", array=[r[key]["iqr"] / 2 for r in rows], visible=True)))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Median-Verzerrung (Balken: halber IQR)", range=[-0.8, 1.2])
    return _fixed(fig, 400)


def build_weak_chart(rows: list) -> go.Figure:
    labels = [f"π = {de(r['pi'], 2)}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["share_weak"] for r in rows], name="Läufe mit F < 10", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=labels, y=[100 * (1 - r["ar_bounded"]) for r in rows], name="Anderson-Rubin-Menge unbeschränkt", marker_color=C.COLOR_CONTROL))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Anteil der 500 Läufe (%)", range=[0, 105])
    return _fixed(fig, 340)


def build_base_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{"π": de(r["pi"], 2), "F (Mittel)": de(r["f_mean"], 1), "Anteil F < 10": f"{de(100 * r['share_weak'], 0)} %", "OLS: Median-Verzerrung": signed(r["ols"]["median_bias"], 3),
                          "2SLS: Median-Verzerrung": signed(r["iv"]["median_bias"], 3), "2SLS: IQR": de(r["iv"]["iqr"], 3), "2SLS: SD": de(r["iv"]["sd"], 2), "Orakel: SD": de(r["oracle"]["sd"], 3)}
                         for r in rows])


def build_interval_table(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        p = r["pretest"]
        out.append({"π": de(r["pi"], 2), "Wald: Deckung": f"{de(100 * r['wald_cover'], 0)} %", "Wald: Breite (Median)": de(r["wald_width"], 2), "AR: Deckung": f"{de(100 * r['ar_cover'], 0)} %",
                    "AR beschränkt": f"{de(100 * r['ar_bounded'], 0)} %", "AR: Breite (Median, beschränkt)": de(r["ar_width"], 2) if r["ar_bounded"] > 0 else "-",
                    "nur F > 10: Läufe": f"{de(100 * p['share'], 0)} %" if p else "unter 4 %", "nur F > 10: Median-Verzerrung": signed(p["median_bias"], 3) if p else "-",
                    "nur F > 10: Wald-Deckung": f"{de(100 * p['cover'], 0)} %" if p else "-"})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Experiment 2

def build_violations_chart(rows: list) -> go.Figure:
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[r["naive_error"] for r in rows], name="naiver Vergleich", marker_color=C.COLOR_NEUTRAL, orientation="h"))
    fig.add_trace(go.Bar(y=labels, x=[r["iv_error"] for r in rows], name="2SLS", marker_color=C.COLOR_GOOD, orientation="h"))
    fig.add_vline(x=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Fehler im Grenzfall (Schätzung minus Wirkung)")
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, 120 + 56 * len(rows))


def build_violations_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{"Fall": r["label"], "Wahrheit": de(r["truth"], 2), "naiv": de(r["naive"], 2), "Kontrolle Z": de(r["control"], 2), "2SLS": de(r["iv"], 2),
                          "Fehler des 2SLS": signed(r["iv_error"], 2), "Fehler naiv": signed(r["naive_error"], 2),
                          "Verhältnis der Fehler": de(r["ratio"], 2) if r["naive_error"] != 0 and r["iv_error"] != 0 else "-"} for r in rows])


# ----------------------------------------------------------------------------- Experiment 3

def build_overid_chart(rows: list) -> go.Figure:
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[100 * r["reject"] for r in rows], name="J-Test lehnt ab (%)", marker_color=C.COLOR_CONTROL, orientation="h",
                         text=[f"{de(100 * r['reject'], 0)} %" for r in rows], textposition="outside"))
    fig.add_vline(x=5.0, line=dict(color="#888888", width=1, dash="dot"), annotation_text="Fehler 1. Art 5 %", annotation_position="top")
    fig.update_xaxes(title="Anteil der 1 000 Läufe, in denen der J-Test ablehnt (%)", range=[0, 112])
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, 120 + 50 * len(rows), showlegend=False)


def build_overid_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{"Fall": r["label"], "Wahrheit": de(r["truth"], 2), "J lehnt ab": f"{de(100 * r['reject'], 0)} %", "2SLS: Median-Verzerrung": signed(r["iv"]["median_bias"], 3),
                          "2SLS: IQR": de(r["iv"]["iqr"], 3), "F (Mittel)": de(r["f_mean"], 1)} for r in rows])


# ----------------------------------------------------------------------------- Experiment 4

def build_late_chart(res: dict) -> go.Figure:
    p = res["population"]
    labels = ["durchschnittliche Wirkung (ATE)", "Wirkung bei Compliern (LATE)"] + [f"Wald-Schätzer, n = {r['n']} (Mittel)" for r in res["rows"]] + ["naiver Vergleich"]
    vals = [p["ate"], p["late"]] + [r["wald"]["mean"] for r in res["rows"]] + [p["naive"]]
    colors = [C.COLOR_NEUTRAL, C.COLOR_TRUTH] + [C.COLOR_GOOD] * len(res["rows"]) + [C.COLOR_BAD]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, text=[de(v, 2) for v in vals], textposition="outside"))
    fig.update_yaxes(title="Wirkung (Differenz der Wahrscheinlichkeiten)", range=[0, 0.85])
    fig.update_xaxes(tickangle=-12)
    return _fixed(fig, 380, showlegend=False)


def build_late_table(res: dict) -> pd.DataFrame:
    p = res["population"]
    out = [{"Typ": t, "Anteil": f"{de(100 * p['shares'][t], 0)} %", "P(Y = 1 | x = 0)": de(C.LATE_PY[t][0], 2), "P(Y = 1 | x = 1)": de(C.LATE_PY[t][1], 2), "Wirkung": de(p["effect"][t], 2)} for t in C.LATE_TYPES]
    return pd.DataFrame(out)


def build_late_runs_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"n": r["n"], "Wald: Mittel": de(r["wald"]["mean"], 3), "Wald: SD": de(r["wald"]["sd"], 3), "naiv: Mittel": de(r["naive"]["mean"], 3), "naiv: SD": de(r["naive"]["sd"], 3),
                          "Läufe": r["used"]} for r in res["rows"]])


# ----------------------------------------------------------------------------- Experiment 5

def build_diagrams_chart(res: dict) -> go.Figure:
    labels = ["gültige Backdoor-Menge", "gültiges Instrument", "gültige Frontdoor-Menge", "Instrument und Frontdoor", "mindestens eines von beiden"]
    vals = [res["backdoor"], res["instrument"], res["frontdoor"], res["both"], res["either"]]
    fig = go.Figure(go.Bar(x=labels, y=[100 * v / res["diagrams"] for v in vals], marker_color=[C.COLOR_BAD, C.COLOR_GOOD, C.COLOR_CONTROL, C.COLOR_NEUTRAL, C.COLOR_TRUTH],
                           text=[de(100 * v / res["diagrams"], 1) + " %" for v in vals], textposition="outside"))
    fig.update_yaxes(title="Anteil der Diagramme (%)", range=[0, 24])
    fig.update_xaxes(tickangle=-12)
    return _fixed(fig, 380, showlegend=False)
