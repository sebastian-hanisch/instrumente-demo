"""Konstanten: Regler-Grenzen, feste Experimente, Farben und Presets."""

# Rechenaufwand am Maximum: n = 5 000 und fünf Knoten, zwei Regressionen: wenige Millisekunden.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 500
SEED_MAX, DEFAULT_SEED = 999999, 35
Z_CRIT = 1.959964
CHI2_CRIT = 3.841459                                         # 95-%-Punkt der χ²-Verteilung mit einem Freiheitsgrad (Anderson-Rubin, Sargan)
F_RULE = 10.0                                               # Faustregel für die erste Stufe

# Stärken (Einheitsrauschen in jedem Knoten)
A_MIN, A_MAX, A_STEP, DEFAULT_A = 0.0, 2.5, 0.1, 1.0       # Stärke des verborgenen Confounders (U -> X und U -> Y gleich stark)
PI_MIN, PI_MAX, PI_STEP, DEFAULT_PI = 0.05, 1.5, 0.05, 0.5  # Stärke des Instruments Z -> X
BETA_MIN, BETA_MAX, BETA_STEP, DEFAULT_BETA = 0.1, 1.5, 0.1, 1.0   # Wirkung X -> Y (die Wahrheit)
G_MIN, G_MAX, G_STEP = 0.0, 0.6, 0.1                       # Verletzung: Z -> Y (Ausschlussrestriktion)
H_MIN, H_MAX, H_STEP = 0.0, 0.6, 0.1                       # Verletzung: U -> Z (Unabhängigkeit)

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
EXP_N, EXP_REPS = 500, 500
EXP_PI_VALUES = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)            # Experiment 1: Stärke des Instruments
EXP_OVER_N, EXP_OVER_REPS = 500, 1000                      # Experiment 3: zwei Instrumente und der J-Test
EXP_LATE_NS, EXP_LATE_REPS = (500, 2000), 500              # Experiment 4: binäre Ermutigung
EXP_DIAGRAMS, EXP_NODES = 3000, 7                          # Experiment 5: Zufallsdiagramme mit verborgener gemeinsamer Ursache
RANDOM_P_LO, RANDOM_P_WIDTH = 0.25, 0.25                   # Kantenwahrscheinlichkeit 0,25 bis 0,5
COEF_LO, COEF_HI = 0.4, 1.2

# Experiment 2: Verletzungen (Name -> Parameter der Welt; Rest Voreinstellung)
VIOLATIONS = {
    "Grundfall (π = 0,5)": {},
    "starker Confounder (U wirkt mit 2,0)": dict(a=2.0),
    "Z wirkt direkt auf y (0,1), π = 1,0": dict(g=0.1, pi=1.0),
    "Z wirkt direkt auf y (0,1), π = 0,5": dict(g=0.1),
    "Z wirkt direkt auf y (0,1), π = 0,3": dict(g=0.1, pi=0.3),
    "Z wirkt direkt auf y (0,1), π = 0,1": dict(g=0.1, pi=0.1),
    "U wirkt auf Z (0,1), π = 1,0": dict(h=0.1, pi=1.0),
    "U wirkt auf Z (0,1), π = 0,3": dict(h=0.1, pi=0.3),
    "U wirkt auf Z (0,3), π = 1,0": dict(h=0.3, pi=1.0),
}

# Experiment 3: zwei Instrumente (je π = 0,7, falls nicht anders); Fälle (Name -> Parameter)
OVER_CASES = {
    "beide gültig": dict(pi=0.7, pi2=0.7),
    "eines ungültig (Z₂ → Y 0,3)": dict(pi=0.7, pi2=0.7, g2=0.3),
    "beide gleich ungültig (je 0,3)": dict(pi=0.7, pi2=0.7, g=0.3, g2=0.3),
    "eines schwach ungültig (Z₂ → Y 0,15)": dict(pi=0.7, pi2=0.7, g2=0.15),
    "beide gültig, aber schwach (π = 0,15)": dict(pi=0.15, pi2=0.15),
}

# Experiment 4: binäre Ermutigung Z ~ B(0,5); Typen Complier / Always-Taker / Never-Taker (keine Defier)
LATE_PZ = 0.5
LATE_TYPES = ("Complier", "Always-Taker", "Never-Taker")
LATE_SHARES = (0.5, 0.2, 0.3)
LATE_PY = {"Complier": (0.2, 0.7), "Always-Taker": (0.9, 0.95), "Never-Taker": (0.1, 0.1)}      # P(Y = 1 | Typ, X = 0), P(Y = 1 | Typ, X = 1)

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_HIDDEN = "#cfcfcf"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(a=DEFAULT_A, pi=DEFAULT_PI, beta=DEFAULT_BETA, g=0.0, h=0.0, n=DEFAULT_N, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Starkes Instrument (Voreinstellung)": preset(),
    "Schwaches Instrument": preset(pi=0.1),
    "Grenzfall (F um 10)": preset(pi=0.2),
    "Ausschluss verletzt": preset(g=0.1),
    "Instrument mit Confounder verbunden": preset(h=0.3),
    "Starker Confounder": preset(a=2.0),
}
PRESET_HELP = {
    "Starkes Instrument (Voreinstellung)": "Alle Annahmen gelten: Z bewegt x, wirkt nur über x auf y und ist unabhängig von U. Der 2SLS trifft, streut aber etwa dreimal so stark wie das unmögliche Orakel.",
    "Schwaches Instrument": "Z bewegt x kaum (π = 0,1, F etwa 3): Die Schätzung streut stark, das Intervall wird sehr breit, die Anderson-Rubin-Menge oft unbeschränkt.",
    "Grenzfall (F um 10)": "Z bewegt x gerade so (π = 0,2, F etwa 10): Die Faustregel F > 10 ist knapp erfüllt, die Streuung noch deutlich zu groß.",
    "Ausschluss verletzt": "Z wirkt auch direkt auf y: Die Ausschlussrestriktion ist verletzt, der Fehler ist g / π. Bei schwachem Instrument wird er größer.",
    "Instrument mit Confounder verbunden": "Die Standortqualität U beeinflusst auch Z: Die Unabhängigkeit ist verletzt, der 2SLS ist verzerrt.",
    "Starker Confounder": "Eine stärkere verborgene Ursache verzerrt den naiven Vergleich stark, den 2SLS nicht: Die Stärke des Confounders ist egal, solange Z gültig ist.",
}
