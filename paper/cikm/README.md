# CIKM 2026 submissions

This folder contains two versions of the paper targeting CIKM 2026.

## Deadlines
- Abstract (regular track only): **May 16, 2026 AoE** (today — likely missed unless registered)
- Full regular paper: **May 23, 2026 AoE** (7 days)
- Short paper: **May 25, 2026 AoE** (9 days; no separate abstract deadline)
- Acceptance: Aug 7, 2026
- Camera-ready: Aug 20, 2026

## Files
- `cikm_short.tex` — **4-page short paper, complete draft**.
  New framing: harmonic = count-only IVW (Aitken 1934); plug-in delta-method
  variance replaces bootstrap; ARE = 4.5x on full KDD; Cochran-Q at 253 sigma
  rejects PBM. Ready for compilation; minor polish + figure pulls remaining.
- `cikm_full.tex` — **10-page regular paper, copied from `paper/main.tex`**.
  Needs integration of the new plug-in-variance section and Cochran-Q
  result, and removal/relegation of the bootstrap-based KDD analysis.
- `abstract.txt` — standalone abstract for EasyChair registration.

## Recommendation
Submit short paper. The new framing fits 4 pages cleanly and the contribution
is "principled framing + rigorous validation," which the short-paper track
explicitly invites.

## Required CIKM machinery
- ACM `sigconf` template (already in use)
- Double-blind: anonymise self-citations (use third-person)
- GenAI Usage Disclosure section (already present in short version)
- EasyChair: nominate >=1 author as reviewer (manual step in submission portal)
