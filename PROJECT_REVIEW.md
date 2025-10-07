# Dynamic Stats Project Review

## Overview
Dynamic Stats is a Streamlit-based web application that combines data-upload utilities, exploratory data analysis, interactive visualization through PyGWalker, and a collection of classical statistical tests. The project delivers an impressive amount of functionality in a single file but would benefit from structural refactoring, clearer separation of concerns, and additional guardrails around statistical workflows.

## Strengths
- **Engaging UI** – The custom CSS and layout provide a polished landing experience for users entering the app.【F:app.py†L20-L79】
- **Robust data ingestion** – The `read_file` helper gracefully supports CSV and multiple Excel formats while surfacing user-friendly error messages.【F:app.py†L82-L125】
- **Thoughtful guidance** – The interface communicates next steps via success and info boxes, reducing friction for first-time users.【F:app.py†L168-L238】
- **Wide statistical coverage** – The app exposes a variety of parametric and non-parametric tests, including effect-size calculations and optional Tukey post-hoc comparisons.【F:app.py†L332-L533】

## Key Issues & Suggested Modifications
### 1. Monolithic `app.py`
All functionality—styling, file I/O, exploratory widgets, and inference logic—lives inside `app.py`. This makes maintenance hard and obscures testability.【F:app.py†L13-L533】

**Recommendations**
- Split the file into modules (e.g., `layout.py`, `data_io.py`, `analysis.py`, `stats_tests.py`) and import them in `app.py`.
- Wrap sidebar setup, main tabs, and statistical test rendering inside dedicated functions to improve readability and reuse.

### 2. Tight coupling between UI and computation
Statistical routines compute results directly inside Streamlit callbacks, which complicates validation and automated testing.【F:app.py†L372-L533】

**Recommendations**
- Move statistical calculations to pure functions that accept pandas Series/DataFrames and return typed result objects.
- Introduce unit tests around those pure functions to verify numerical correctness independent of the UI layer.

### 3. Limited data validation before analyses
Currently, only basic NaN filtering is applied. Assumption checks (normality, equal variances) are optional and incomplete, leading to risk of misinterpretation.【F:app.py†L418-L467】

**Recommendations**
- Add dedicated validators (e.g., Shapiro-Wilk, Levene) with explicit messaging about when test assumptions are violated.
- Incorporate effect-size interpretations (small/medium/large) to contextualize results.

### 4. PyGWalker workflow fragility
Large datasets trigger sampling, but column-type coercion and high-cardinality warnings are only logged to the UI; the app can still fail when PyGWalker receives unsupported dtypes.【F:app.py†L266-L322】

**Recommendations**
- Sanitize column names and enforce consistent data types prior to invoking PyGWalker.
- Provide a fallback lightweight visualization module even when the checkbox is selected, ensuring the page doesn’t stall on exceptions.

### 5. Performance & caching considerations
Repeatedly sampling or cleaning data within Streamlit callbacks can become costly for large datasets.【F:app.py†L284-L322】

**Recommendations**
- Cache intermediate transformations (e.g., cleaned sample, correlation matrices) keyed by the uploaded file hash and user selections.
- Offer user controls for sample size and column limits rather than hard-coded thresholds.

### 6. Documentation gaps
The README explains usage but omits architectural guidance, contribution standards, and deployment steps beyond local execution.【F:README.md†L1-L117】

**Recommendations**
- Document the intended module layout, coding style, and testing strategy once the codebase is reorganized.
- Provide Streamlit Cloud / Docker deployment instructions and a troubleshooting section for common errors (PyGWalker install, Excel engine issues).

## Additional Enhancement Ideas
- **Accessibility**: Provide light/dark theme toggles and ensure color palettes meet contrast guidelines.
- **Internationalization**: Externalize UI strings to support Portuguese/English toggles.
- **Data provenance**: Surface metadata about uploaded files (source, upload time) and allow users to download intermediate outputs.
- **Automated quality checks**: Integrate a `pre-commit` configuration for formatting (e.g., `black`, `ruff`) and static analysis.

## Next Steps
1. Refactor the codebase into modular components with clear responsibilities.
2. Add automated tests for data loading utilities and statistical functions.
3. Expand documentation to cover architecture, deployment, and contribution workflows.
4. Iterate on PyGWalker integration to make high-volume datasets more resilient.

Addressing these areas will make Dynamic Stats easier to maintain, safer to extend, and more reliable for analysts relying on its statistical guidance.
