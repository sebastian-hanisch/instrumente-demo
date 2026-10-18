"""Instrumente: die Wirkung über einen Anstoß von außen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Linie Kausale Diagramme und Adjustierung. Der Confounder ist verborgen, einen Mediator gibt es nicht; ein Anstoß Z, der nur über x auf y wirkt, liefert über
das Verhältnis Cov(Z, Y) / Cov(Z, X) die Wirkung. Kern: ins_estimators.py, ins_criterion.py; Diagramme: dsep_graph.py und bdr_backdoor.py (wortgleich aus den Vorgängern).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ins_constants as C
from ins_evaluation import (
    Settings,
    analyse,
    base_experiment,
    de,
    diagram_experiment,
    late_experiment,
    overid_experiment,
    signed,
    verdict_base,
    verdict_diagrams,
    verdict_instance,
    verdict_late,
    verdict_overid,
    verdict_violations,
    violations_experiment,
)
from ins_pdf_export import build_pdf
from ins_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from ins_visualization import (
    build_base_chart,
    build_base_table,
    build_diagram,
    build_diagrams_chart,
    build_estimates_chart,
    build_estimates_table,
    build_first_stage_table,
    build_interval_table,
    build_late_chart,
    build_late_runs_table,
    build_late_table,
    build_overid_chart,
    build_overid_table,
    build_violations_chart,
    build_violations_table,
    build_weak_chart,
    style_estimates_table,
)

st.set_page_config(page_title="Instrumente – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _base():
    return base_experiment()


@st.cache_data(show_spinner=False)
def _violations():
    return violations_experiment()


@st.cache_data(show_spinner=SPINNER)
def _overid():
    return overid_experiment()


@st.cache_data(show_spinner=SPINNER)
def _late():
    return late_experiment()


@st.cache_data(show_spinner=SPINNER)
def _diagrams():
    return diagram_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎯 Instrumente: die Wirkung über einen Anstoß von außen")
st.markdown(
    """
**Wenn der Confounder fehlt und kein Mediator da ist, hilft ein Anstoß, der nur über x wirkt.** Dies ist das neunte Stück der Linie **Kausale Diagramme und Adjustierung**. Das dritte Stück hat
gezeigt, dass es bei einer verborgenen gemeinsamen Ursache keine gültige Menge zum Festhalten gibt, das achte, dass die Vordertür nur selten offen ist; das vierte, dass derselbe Anstoß als
**Kontrolle** schadet. Hier wird er richtig benutzt: Er verschiebt die Regelwahl, ohne den Servicegrad selbst zu bewegen. Was er im Servicegrad bewegt, kommt also über die Regel, und das
**Verhältnis** beider Verschiebungen ist die Wirkung der Regel, ohne dass man den verborgenen Confounder kennt.

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Instrument", expanded=True):
    st.markdown(
        """
1. **Die Struktur.** Die Standortqualität U (verborgen) bewegt die Regelwahl X und den Servicegrad Y. Der Anstoß Z bewegt nur X; ein Pfeil Z → Y oder eine Verbindung von U zu Z darf es nicht geben.
2. **Erste Stufe.** Wie stark bewegt Z die Regel? Das ist die Steigung π̂ von X auf Z, und die **F-Statistik** dieser Regression sagt, ob das Instrument stark genug ist (Faustregel: über 10).
3. **Reduzierte Form.** Wie stark bewegt Z den Servicegrad? Das ist die Steigung von Y auf Z; sie enthält nur, was über X läuft.
4. **Das Verhältnis** der reduzierten Form zur ersten Stufe ist die Wirkung β̂ (der Wald-Schätzer; bei mehreren Instrumenten der 2SLS). Der Confounder fällt heraus, weil Z mit ihm nichts zu tun hat.
5. **Nicht testbar.** Ob Z wirklich nur über X wirkt und unabhängig von U ist, lässt sich aus den Daten nicht prüfen; die Experimente zeigen, was bei Verletzung passiert.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Struktur (Annahmen erfüllt)**")
    a_ = st.slider("Stärke des Confounders", *bounds("a_slider"), step=C.A_STEP, key="a_slider", help="Wie stark die verborgene Standortqualität U die Regelwahl und den Servicegrad bewegt (beide gleich stark).")
    pi_ = st.slider("Stärke des Instruments π (Anstoß → Regel)", *bounds("pi_slider"), step=C.PI_STEP, key="pi_slider",
                    help="Wirkung des Anstoßes auf die Regelwahl. Je kleiner, desto schwächer das Instrument; unter etwa 0,2 (bei n = 500) fällt F unter 10.")
    beta_ = st.slider("Wirkung der Regel β (die Wahrheit)", *bounds("beta_slider"), step=C.BETA_STEP, key="beta_slider", help="Wirkung der Regel auf den Servicegrad. Das ist die Größe, die geschätzt wird.")
    st.markdown("**Verletzte Annahmen**")
    g_ = st.slider("Anstoß wirkt direkt auf den Servicegrad", *bounds("g_slider"), step=C.G_STEP, key="g_slider", help="Ein Pfeil Z → Y verletzt die Ausschlussrestriktion: Der Fehler ist g / π.")
    h_ = st.slider("Standortqualität wirkt auf den Anstoß", *bounds("h_slider"), step=C.H_STEP, key="h_slider", help="Eine Verbindung U → Z verletzt die Unabhängigkeit: Z ist selbst vom Confounder betroffen.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Stichprobengröße der simulierten Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

values = {"a_slider": a_, "pi_slider": pi_, "beta_slider": beta_, "g_slider": g_, "h_slider": h_}
sync_query_params({**{k: round(float(v), 2) for k, v in values.items()}, "n_slider": int(n), "seed_input": int(seed)})

settings = Settings(*(round(float(v), 2) for v in values.values()), int(n), int(seed))
a = _analysis(settings)
by = {r.name: r for r in a.rows}
iv = by["2SLS (Instrument)"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahre Wirkung", de(a.truth, 2), delta="aus dem Modell, nicht aus den Daten", delta_color="off")
c2.metric("Naiver Vergleich", signed(by["Naiver Vergleich"].estimate, 3), delta=f"Fehler {signed(by['Naiver Vergleich'].error, 2)}", delta_color="off",
          help="Koeffizient von x in der Regression von y auf x, ohne Adjustierung.")
c3.metric("2SLS", signed(iv.estimate, 3), delta=f"Fehler {signed(iv.error, 2)}, Intervall {de(iv.lo, 2)} bis {de(iv.hi, 2)}", delta_color="off",
          help="Verhältnis Cov(Z, Y) / Cov(Z, X); Intervall mit dem Standardfehler des 2SLS.")
c4.metric("Kontrolle für das Instrument", signed(by["Kontrolle für das Instrument"].estimate, 3), delta=f"Fehler {signed(by['Kontrolle für das Instrument'].error, 2)}", delta_color="off",
          help="Die schlechte Kontrolle aus Stück 4: Festhalten von Z verstärkt die Verzerrung durch U.")

left, right = st.columns([1, 1.2])
with left:
    st.plotly_chart(build_diagram(settings), width="stretch", key="diagram_chart")
    st.caption("Grün: der Weg über die Regel. Grau: verborgene Knoten und der Confounder-Pfad. Rot: verletzte Annahmen (nur wenn der Regler über 0 liegt).")
with right:
    st.plotly_chart(build_estimates_chart(a), width="stretch", key="estimates_chart")
    st.caption("Punkt mit Intervall: grün, wenn es die Wahrheit deckt, rot, wenn nicht. Raute: Grenzwert bei unendlich vielen Beobachtungen. Das Orakel kennt U und ist in der Praxis nicht möglich.")
verdict(verdict_instance(a))
st.dataframe(style_estimates_table(build_estimates_table(a)), hide_index=True, width="stretch")
st.markdown("**Die erste Stufe und die Anderson-Rubin-Menge**")
st.dataframe(build_first_stage_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="instrumente.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie stark muss das Instrument sein?")
res1 = _base()
st.caption(f"Alle Annahmen erfüllt, n = {C.EXP_N}, {C.EXP_REPS} feste Wiederholungen je Stärke π des Instruments; die Wahrheit ist β = 1. Weil der 2SLS bei einem Instrument theoretisch "
           "keine Momente hat (bei schwachem Instrument zeigt es sich in der Stichprobe), zählen Median und Quartile, nicht Mittel und Standardabweichung.")
st.plotly_chart(build_base_chart(res1), width="stretch", key="base_chart")
verdict(verdict_base(res1))
st.dataframe(build_base_table(res1), hide_index=True, width="stretch")
st.plotly_chart(build_weak_chart(res1), width="stretch", key="weak_chart")
st.dataframe(build_interval_table(res1), hide_index=True, width="stretch")
st.caption("Die Wald-Deckung bleibt bei schwachem Instrument hoch, bei gleichzeitig sehr breit werdendem Intervall: Die Deckung allein sagt nichts. Wer bei schwachem bis mittlerem Instrument nur Läufe mit F über 10 behält, verzerrt "
           "die Schätzung selbst (die letzten drei Spalten).")

st.subheader("🔬 Experiment 2: Wenn eine Annahme verletzt ist")
res2 = _violations()
st.caption("Exakte Grenzwerte (unendlich viele Beobachtungen), keine Stichprobe. Die Wahrheit ist die Wirkung β = 1; die Verletzung ändert sie nicht.")
st.plotly_chart(build_violations_chart(res2), width="stretch", key="violations_chart")
verdict(verdict_violations(res2))
st.dataframe(build_violations_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Zwei Instrumente und der J-Test")
res3 = _overid()
st.caption(f"Zwei Anstöße Z und Z₂ (je π = 0,7, falls nicht anders), n = {C.EXP_OVER_N}, {C.EXP_OVER_REPS} feste Wiederholungen. Der J-Test (Sargan) fragt, ob die Reste des 2SLS noch mit den "
           "Instrumenten zusammenhängen; mit einem Instrument ist er nicht möglich.")
st.plotly_chart(build_overid_chart(res3), width="stretch", key="overid_chart")
verdict(verdict_overid(res3))
st.dataframe(build_overid_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Was misst der Wald-Schätzer bei unterschiedlichen Wirkungen?")
res4 = _late()
st.caption(f"Binäre Ermutigung Z (Münzwurf), x und y sind 0/1. Drei Sorten von Standorten: Complier folgen dem Anstoß, Always-Taker führen die Regel immer ein, Never-Taker nie (keine Defier). "
           f"{C.EXP_LATE_REPS} feste Wiederholungen je n.")
st.dataframe(build_late_table(res4), hide_index=True, width="stretch")
st.plotly_chart(build_late_chart(res4), width="stretch", key="late_chart")
verdict(verdict_late(res4))
st.dataframe(build_late_runs_table(res4), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: Wie oft gibt es ein Instrument?")
res5 = _diagrams()
st.caption(f"{res5['diagrams']} Zufallsdiagramme mit {res5['nodes']} Knoten, in denen ein verborgener Knoten die Regel und das Ergebnis gemeinsam verursacht und es mindestens einen kausalen Pfad "
           "gibt; alle anderen Knoten sind beobachtet. Das Kriterium: z hängt mit x zusammen und ist in dem Diagramm ohne alle Pfeile in x von y getrennt.")
st.plotly_chart(build_diagrams_chart(res5), width="stretch", key="diagrams_chart")
verdict(verdict_diagrams(res5))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Z wirkt nur über x auf y (Ausschlussrestriktion)** | Der Fehler ist g / π: Ein kleiner direkter Pfeil wird bei schwachem Instrument vervielfacht (Experiment 2). | Sensitivität und Falsifikation |
| **Z ist unabhängig von der verborgenen Ursache** | Der 2SLS ist verzerrt; die Daten zeigen es nicht, die erste Stufe sieht die Verbindung sogar mit: Bei π = 0,2 und U → Z (0,3) steigt das erwartete F von 10 auf 64 (Experiment 2). | Sensitivität und Falsifikation |
| **Z bewegt x genug (Relevanz)** | Bei F unter etwa 10 streut die Schätzung stark, das Intervall wird sehr breit oder unbeschränkt, die Vorauswahl nach F verzerrt selbst (Experiment 1). | nicht gebaut |
| **Mehrere Instrumente lassen sich prüfen** | Der J-Test ist blind, wenn alle Instrumente dieselbe Verzerrung g / π erzeugen (zum Beispiel gleiche Verletzung bei gleicher Stärke, Experiment 3). | Sensitivität und Falsifikation |
| **Die Wirkung ist für alle gleich** | Bei unterschiedlichen Wirkungen misst der Wald-Schätzer nur die Wirkung bei den Compliern, nicht die durchschnittliche (Experiment 4). | nicht gebaut |
| **Die Struktur ist bekannt** | Keine der Annahmen lässt sich aus den Daten prüfen; ein falsches Diagramm gibt eine sichere, aber falsche Zahl. | Wie falsch darf das Diagramm sein? |
| **Es gibt überhaupt ein Instrument** | Im Zufallsmodell von Experiment 5 (sieben Knoten, Kantenwahrscheinlichkeit 0,25 bis 0,5, verborgene Ursache) hat etwa jedes zehnte Diagramm ein Instrument; der Anteil hängt stark von Größe und Dichte der Diagramme ab. | Wann geht es gar nicht? |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Instrument-Kriterium** (Pearl). $Z$ ist ein Instrument für $(X, Y)$, wenn (i) $Z$ nicht von $X$ d-getrennt ist (Relevanz) und (ii) $Z$ in dem Diagramm ohne alle Pfeile in $X$ von $Y$
d-getrennt ist (keine direkte Wirkung, keine gemeinsame Ursache mit $Y$).

**Wald-Schätzer und 2SLS.** Bei einem Instrument ist $\hat\beta = \widehat{\mathrm{Cov}}(Z, Y)\,/\,\widehat{\mathrm{Cov}}(Z, X)$. Bei mehreren Instrumenten: $\hat\beta = (\hat X'Y)/(\hat X'X)$ mit
$\hat X$ aus der ersten Stufe $X \sim Z_1 + Z_2$. Der Standardfehler benutzt die Reste $Y - \hat\beta X$, nicht $Y - \hat\beta \hat X$.

**Erste Stufe.** $F = t^2$ der Steigung von $X$ auf $Z$ (bei einem Instrument); erwartet etwa $n R^2/(1 - R^2)$ mit $R^2 = \mathrm{Korr}(Z, X)^2$.

**Anderson-Rubin-Menge.** $\{\beta_0 : |t(\beta_0)| \le 1{,}96\}$, wobei $t(\beta_0)$ die $t$-Statistik der Regression von $Y - \beta_0 X$ auf $Z$ ist; eine quadratische Ungleichung in $\beta_0$,
die Menge ist ein Intervall, die Außenseite eines Intervalls (unbeschränkt) oder die ganze Gerade. Ihre Deckung hängt nicht von der Stärke des Instruments ab, ihre Form schon.

**Verzerrung ohne die Annahmen.** Mit einem direkten Pfeil $g$ von $Z$ auf $Y$ ist der Grenzwert $\beta + g/\pi$; mit einer Verbindung $h$ von $U$ zu $Z$ ist er verzerrt, solange $Z$ mit dem Fehlerterm von $Y$ zusammenhängt (hier: $h$ und die Stärke des Confounders beide ungleich 0).

**J-Test (Sargan).** $J = n R^2$ der Regression der 2SLS-Reste auf die Instrumente, unter der Nullhypothese gültiger Instrumente $\chi^2$ mit (Zahl der Instrumente − 1) Freiheitsgraden.

**LATE.** Mit zufällig zugeteiltem binärem $Z$ (unabhängig, nur über $X$ wirkend, relevant), binärem $X$ und ohne Defier misst $(E[Y|Z=1] - E[Y|Z=0])/(E[X|Z=1] - E[X|Z=0])$ die mittlere Wirkung bei den Compliern.

**Literatur.** J. Angrist, G. Imbens, D. Rubin, *Identification of causal effects using instrumental variables*, JASA 91, 1996; D. Staiger, J. Stock, *Instrumental variables regression with weak
instruments*, Econometrica 65, 1997; J. Pearl, *Causality*, 2. Auflage, 2009; T. Anderson, H. Rubin, *Estimation of the parameters of a single equation in a complete system of stochastic
equations*, Annals of Mathematical Statistics 20, 1949 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
