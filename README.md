# Instrumente: die Wirkung über einen Anstoß von außen – Streamlit-Demo

Neuntes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [frontdoor-demo](https://github.com/sebastian-hanisch/frontdoor-demo): Dort lief die Wirkung über einen beobachteten Mediator. Hier gibt es keinen, aber einen Anstoß von außen, der nur über die Regel wirkt, und das Verhältnis seiner Wirkungen auf Regel und Ergebnis liefert die Wirkung der Regel, obwohl der Confounder verborgen bleibt.

## Kernfrage

Ein Instrument Z bewegt x, wirkt aber nur über x auf y und ist unabhängig von der verborgenen Ursache. Dann ist Cov(Z, Y) / Cov(Z, X) die Wirkung (Wald-Schätzer, bei mehreren Instrumenten der 2SLS). Was kostet das an Genauigkeit, wann bricht es zusammen, und was zeigt der J-Test wirklich?

## Modell

- **Vehikel:** ein Standort wählt die neue Regel B (X) abhängig von der verborgenen Standortqualität U; ein Anstoß Z verschiebt die Regelwahl, wirkt aber nicht selbst auf den Servicegrad Y. Linear-gaußsches Strukturmodell mit Einheitsrauschen in jedem Knoten: U → X und U → Y (Stärke a), Z → X (π), X → Y (β). Die Wahrheit ist **β** (Grundfall 1,0, π = 0,5).
- **Zwei Verletzungen** als Regler: ein direkter Pfeil Z → Y (g, Ausschlussrestriktion) und U → Z (h, Unabhängigkeit). Die Wahrheit ändert sich dadurch nicht.
- **Zweites Instrument** (Experiment 3), **binäre Ermutigung** mit Compliern, Always-Takern und Never-Takern (Experiment 4) und **Zufallsdiagramme** (Experiment 5).
- **Daten:** `SplitMix64`-Ströme (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).

## Methodik

- **Schätzer** (`ins_estimators.py`): naiver Vergleich; Kontrolle für das Instrument (Y ~ X + Z, die schlechte Kontrolle aus Stück 4); **2SLS** (bei einem Instrument der Wald-Quotient), erste Stufe mit **F-Statistik**; das Orakel Y ~ X + U (U ist nicht beobachtbar, nur Maßstab).
- **Intervalle:** Wald (Schätzung ± 1,96 Standardfehler) und die **Anderson-Rubin-Menge** (geschlossene Form bei einem Instrument: ein Intervall, die Außenseite eines Intervalls oder die ganze Gerade).
- **J-Test** (Sargan) bei zwei Instrumenten: n R² der Reste auf die Instrumente.
- **Kriterium** (`ins_criterion.py`): Z ist relevant (nicht d-getrennt von x) und in dem Diagramm ohne Pfeile in x von y d-getrennt; gerechnet über `dsep_graph.py` und `bdr_backdoor.py` (wortgleich aus den Vorgängern). Die Frontdoor-Prüfung steht für den Vergleich in Experiment 5 wortgleich aus Stück 8 mit drin.
- **Experimente** laufen über feste Zufallsströme; die Verletzungen (Experiment 2) sind exakte Grenzwerte aus der Population, keine Stichproben. Bei schwachem Instrument zählen **Median, Quartile und Deckung**, nie das Mittel des 2SLS: Seine Momente existieren dort nicht.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wahrheit β = 1,00, Verzerrung = Median minus Wahrheit.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (n = 500, π = 0,5, Seed 35) | Naiv **1,43**; Kontrolle für Z **1,50**; **2SLS 1,04** (Standardfehler 0,113, deckt); das Orakel mit U 1,02. Erste Stufe: π̂ 0,550 (Standardfehler 0,062), F **79,1**; Anderson-Rubin-Menge 0,79 bis 1,25. | `test_default_sample` |
| Trifft das Instrument, und was kostet es? (π = 1,0 / 0,5, n = 500, 500 Wiederholungen) | Median-Verzerrung 0,000 / −0,001; Streuung **0,065 / 0,132** gegen das Orakel 0,032 / 0,040: **das Doppelte, bei π = 0,5 das 3,3fache**. Der OLS steht bei +0,334 / +0,445. Wald- und AR-Deckung 95 bis 96 %. F im Mittel 253 / 64. | `test_experiment_1_strong_instrument` |
| Wie fällt es, wenn das Instrument schwächer wird? (π = 0,3 / 0,2 / 0,1 / 0,05) | F im Mittel **22,5 / 11,1 / 3,6 / 1,6**; Anteil der Läufe mit F < 10: 7 / **53** / **94** / 100 %. Median-Verzerrung −0,024 / +0,022 / +0,022 / **+0,314** (zum OLS hin), IQR 0,29 / 0,48 / **0,92** / **1,32**. Die Standardabweichung (0,40 bei 0,2, **4,9** bei 0,1, über 100 bei 0,05) ist wertlos. | `test_experiment_1_weakening_instrument` |
| Und die Intervalle? | Wald-Deckung bleibt bei 95 bis 98 %, **weil das Intervall breiter wird**: Breite (Median) **0,50 / 0,86 / 1,25 / 2,74 / 5,32** für π = 0,5 bis 0,05. Die Anderson-Rubin-Menge deckt zu 93 bis 96 %, ist aber bei π = 0,2 / 0,1 / 0,05 nur in **89 / 37 / 14 %** der Läufe beschränkt. | `test_experiment_1_intervals` |
| Hilft die Vorauswahl „F > 10“? | Nein: Wer nur Läufe mit F über 10 behält, verzerrt selbst. π = 0,2: 47 % der Läufe, Median-Verzerrung **+0,132**, Deckung 89 %; π = 0,1: 6 % der Läufe, **+0,349**, Deckung **72 %**. | `test_experiment_1_pretest_on_f` |
| Was kostet jede Verletzung? (exakte Grenzwerte) | Ein direkter Pfeil g = 0,1 gibt den Fehler **g / π**: +0,10 (π = 1,0), +0,20 (0,5), +0,33 (0,3), **+1,00** (0,1) — bei π = 0,1 doppelt so groß wie der OLS-Fehler (+0,50). Eine Verbindung U → Z (0,1) gibt +0,09 (π = 1,0) und +0,25 (π = 0,3); sie macht die erste Stufe stärker, nicht schwächer (erwartetes F bei π = 0,2 und U → Z 0,3: **64** statt 10). Ein starker Confounder (a = 2) ändert den 2SLS nicht (0,00), der naive Vergleich geht auf +0,76. | `test_experiment_2_violations` |
| Erkennt der J-Test ungültige Instrumente? (zwei Instrumente, 1 000 Wiederholungen) | Beide gültig: lehnt in **5 %** ab. Eines ungültig (0,3): **95 %**, Median-Verzerrung +0,216. Beide **gleich** ungültig (je 0,3): nur **6 %** (blind), Verzerrung **+0,429**. Eines schwach ungültig (0,15): 40 %, +0,112. Beide gültig, aber schwach (π = 0,15, F 6,8): 5 %, IQR 0,36. | `test_experiment_3_overidentification` |
| Was misst der Wald-Schätzer bei unterschiedlichen Wirkungen? (binäre Ermutigung) | Wirkung bei den Compliern (LATE) **0,50**, durchschnittliche Wirkung (ATE) **0,26**, naiv **0,67**. Wald: 0,498 (n = 500, Streuung 0,069) und 0,503 (n = 2 000, 0,034): er trifft das LATE, nicht das ATE. | `test_experiment_4_late` |
| Wie oft gibt es ein Instrument? (3 000 Zufallsdiagramme, sieben Knoten, verborgene Ursache von x und y) | Nie eine gültige Backdoor-Menge (**0**), aber in **304** (10,1 %) ein gültiges Instrument und in 265 (8,8 %) eine Frontdoor-Menge, in nur **15** beides; **554** (18,5 %) mindestens eines von beiden. Gültige Instrumente je Diagramm: 0 in 2 696, 1 in 225, 2 in 58, 3 oder mehr in 21. In allen 410 Paaren aus Instrument und linearer Population stimmt der Quotient mit der Wirkung. | `test_experiment_5_diagrams` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Z wirkt nur über x auf y (Ausschlussrestriktion)** | Der Fehler ist g / π: Ein kleiner direkter Pfeil wird bei schwachem Instrument vervielfacht (Experiment 2). | Sensitivität und Falsifikation |
| **Z ist unabhängig von der verborgenen Ursache** | Der 2SLS ist verzerrt; die Daten zeigen es nicht, die erste Stufe sieht die Verbindung sogar mit: Bei π = 0,2 und U → Z (0,3) steigt das erwartete F von 10 auf 64 (Experiment 2). | Sensitivität und Falsifikation |
| **Z bewegt x genug (Relevanz)** | Bei F unter etwa 10 streut die Schätzung stark, das Intervall wird sehr breit oder unbeschränkt, die Vorauswahl nach F verzerrt selbst (Experiment 1). | nicht gebaut |
| **Mehrere Instrumente lassen sich prüfen** | Der J-Test erkennt nur, wenn die Instrumente verschieden ungültig sind; sind alle gleich ungültig, ist er blind (Experiment 3). | Sensitivität und Falsifikation |
| **Die Wirkung ist für alle gleich** | Bei unterschiedlichen Wirkungen misst der Wald-Schätzer nur die Wirkung bei den Compliern, nicht die durchschnittliche (Experiment 4). | nicht gebaut |
| **Die Struktur ist bekannt** | Keine der Annahmen lässt sich aus den Daten prüfen; ein falsches Diagramm gibt eine sichere, aber falsche Zahl. | Wie falsch darf das Diagramm sein? |
| **Es gibt überhaupt ein Instrument** | In nur etwa jedem zehnten Zufallsdiagramm mit verborgener Ursache existiert eines (Experiment 5). | Wann geht es gar nicht? |

Die Zahlen gelten für dieses erfundene Vehikel (linear-gaußsch, ein verborgener Confounder, Einheitsrauschen, gleichmäßige Streuung, bis zu zwei Instrumente); die Instrumentformel selbst gilt allgemein, ihre linearen Zahlen hier nicht. „Wahrheit“ ist das simulierte Modell, nicht ein Befund über echte Standorte. Die Quellen im 📐-Expander sind aus dem Gedächtnis, nicht geprüft; die „F > 10“-Regel ist eine Faustregel, ihre Verzerrung durch die Vorauswahl ist hier **gemessen**, nicht zitiert.

## Bezug zu den Nachbarn im Portfolio

- [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo): Dort verstärkt das Festhalten eines Instruments die Verzerrung. Hier ist dasselbe Z der Schätzer; die Zeile „Kontrolle für das Instrument“ (Grenzwert 1,5 bei jedem π) ist die Gegenprobe.
- [frontdoor-demo](https://github.com/sebastian-hanisch/frontdoor-demo): die andere Ersatztür bei verborgener Ursache; beide zusammen helfen in 18,5 % der Diagramme, die Überschneidung ist klein.
- [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo) und [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo): Adjustierung über beobachtete Mengen; hier findet die Backdoor-Suche in den Zufallsdiagrammen nie eine Menge.
- [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo): der verborgene Confounder und sein naiver Vergleich; hier wird er umgangen, nicht gemessen.
- **Abgrenzung zur Wirkungsmessung durch Experimente:** Ein Anstoß ist ein unvollständig befolgtes Experiment; Zuteilung durch den Test, Nicht-Befolgung und Complier stehen dort. Hier ist die Zuteilung nicht kontrolliert, nur der Anstoß ist von außen gegeben. Keine Typ-Schätzung, keine Monotonie-Tests, keine Zerlegung nach Teilgruppen (heterogene Wirkungen sind nicht gebaut).

## Tests

Mini-Instanzen von Hand (a = π = β = 1: Varianzen 1 und 3, naiv 4/3, Kontrolle 1,5, 2SLS 1; vier Einheiten mit β̂ = 2, Standardfehler √2, F = 1; acht binäre Einheiten mit Wald 0,5), das Kriterium an Hand-Diagrammen und gegen ein zweites Verfahren (networkx, 1 200 Paare aus Knoten und Diagramm in 300 Zufallsdiagrammen), Orakel für die geschlossenen Formeln der Verletzungen, die erwartete erste Stufe, die Deckung von Wald und Anderson-Rubin, die Größe des J-Tests, Auswertung und Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py`, `bdr_backdoor.py` | d-Separation und Backdoor/Adjustierungskriterium (wortgleich aus Stück 2 und 3) |
| `ins_world.py` | Vehikel: Strukturmodell, Wahrheit, Population, Stichprobe, binäre Ermutigung |
| `ins_estimators.py` | naiv, Kontrolle, Orakel, 2SLS, erste Stufe, Anderson-Rubin, J-Test, binärer Wald |
| `ins_criterion.py` | Instrument-Kriterium und Frontdoor-Prüfung |
| `ins_evaluation.py` | Einstellungen, Analyse, fünf Experimente, Verdikte |
| `ins_visualization.py`, `ins_pdf_export.py` | Figuren, Tabellen, PDF |
| `ins_constants.py`, `ins_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Mini-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **LIML, Fuller, bedingte Likelihood-Quotienten, Stock-Yogo-Tabellen:** Die Verzerrung bei schwachem Instrument wird gezeigt, nicht behandelt; die Tabellen stehen nur namentlich im Expander (aus dem Gedächtnis).
- **Bedingte Instrumente, mehrere Behandlungen, nichtlineare erste Stufe:** das Kriterium rechnet unbedingte Instrumente.
- **Heterogene Wirkungen, Monotonie-Tests, Typ-Schätzung, marginale Wirkungen:** eigenes Thema; Experiment 4 zeigt nur, dass der Wald-Schätzer das LATE misst.
- **Robuste Standardfehler, Cluster:** gleichmäßige Streuung im Vehikel.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
