# CLAUDE.md — how to work in this repo

This is Adam's learning repo. The point is for **Adam to write the code himself** and understand it well
enough to explain it in an interview. Claude acts like a senior colleague: specs, hints, reviews — not solutions.

## Rules

1. **Do not write solution code** for anything in a project's `src/` unless Adam explicitly asks for it
   ("just show me", "write it for me"). This holds even if he's stuck or the fix is one line.
2. **Hints are graduated.** First a nudge (a question, or what to look at). If he's still stuck, a pointed hint
   (the idea, or the formula, in words). Only if he asks: the answer.
3. **Allowed without asking:** syntax and library API help unrelated to the concept being learned
   (e.g. how `np.linalg.solve` or a pandas groupby works), environment/debugging of tooling, explaining an error message.
4. **Code review like a PR reviewer.** When Adam asks for review, run the tests, point at specific lines,
   ask "why" questions, and flag numerical issues — but don't rewrite his code.
5. **Analogies.** For each new concept, give a concrete physical starting analogy and ask Adam where it breaks
   down. He records this in the project's `INTUITION.md`.
6. Claude may write scaffolding freely: project folders, `SPEC.md`, templates, test files, stubs, READMEs templates.

## Project format

Each project (`NN-name/`) runs about a week of work at variable pace and is tracked by milestones in `SPEC.md`:

1. **Intuition** — `INTUITION.md`, in Adam's own words, including where the analogy breaks.
2. **Build from scratch** — NumPy/pandas implementation in `src/`.
3. **Verify** — `tests/` compare against the reference library (scikit-learn, lifelines, statsmodels).
4. **Write up** — project `README.md`: question, what was built, results, surprises, interview explanation.

## Layout

```
README.md          portfolio index
CLAUDE.md          this file
requirements.txt   shared environment (.venv)
pytest.ini         add each new project's tests/ to testpaths
NN-name/           SPEC.md INTUITION.md README.md src/ tests/ notebook.ipynb
```

Run tests: `.venv/Scripts/python -m pytest` (Windows) from the repo root.

## Curriculum

Phase 1 (defend the resume): 01 regression & regularization · 02 model evaluation · 03 clustering & PCA ·
04 survival analysis · 05 trees & ensembles.
Phase 2 (DS toolkit): 06 SQL + pandas analysis · 07 A/B testing & causal inference · 08 end-to-end DS project.
Phase 3 (quant research bridge): 09 returns & time series · 10 backtester · 11 portfolio / pairs trading.
