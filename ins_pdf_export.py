"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from ins_evaluation import Analysis, ar_text, de, signed, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "€": "EUR", "≥": ">=", "≤": "<=", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "π̂": "pi^", "π": "pi", "β": "beta",
            "∞": "inf", "₂": "2"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Instrumente: die Wirkung über einen Anstoß von außen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Stärke des Confounders {de(s.a, 1)}, Stärke des Instruments π = {de(s.pi, 2)}, Wirkung β = {de(s.beta, 1)}; Verletzungen: Z -> Y {de(s.g, 1)}, U -> Z {de(s.h, 1)}; "
                            f"n = {s.n}, Seed {s.seed}. Alles simuliert. Wahre Wirkung {de(a.truth, 2)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        pdf.multi_cell(0, 4.6, _s(f"{r.name}: {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}], Fehler {signed(r.error, 3)}, {'deckt' if r.covers else 'deckt nicht'}; Grenzwert {de(r.limit, 3)}."),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    f = a.first
    pdf.multi_cell(0, 4.6, _s(f"Erste Stufe: pi^ = {de(f['pi'], 3)} (Standardfehler {de(f['se_pi'], 3)}), F = {de(f['f'], 1)}; Anderson-Rubin-Menge: {ar_text(a.ar)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: instrumente-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
