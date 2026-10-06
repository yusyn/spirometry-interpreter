# Official GLI coefficient & spline tables

This directory is intentionally empty in the PoC repository.

## Required layout

```
data/
├── gli_global/
│   ├── coefficients.json
│   ├── m_spline.csv
│   └── s_spline.csv
└── gli_2012_other/
    ├── coefficients.json
    ├── m_spline.csv
    └── s_spline.csv
```

## Licence (verified 2026-10-06 from ERS official FAQ)

Equations published open access under **CC-BY-NC**
(https://creativecommons.org/licenses/by-nc/2.0/).

- Usage requires appropriate acknowledgement.
- Commercial use is **not** permitted without contacting ERS
  (permissions@ersnet.org).
- GLI software packages are restricted to research, education, training
  and validation of implementation — not for patient treatment.

Source: https://www.ersnet.org/science-and-research/ongoing-clinical-research-collaborations/the-global-lung-function-initiative/

## How to obtain the data

1. ERS GLI hub / Tools / Resources pages and paper supplements:
   - Quanjer et al. 2012 online data supplement (look-up tables)
   - Bowerman et al. 2023 supplementary Table E3 / look-up tables
2. Convert official tables into the JSON/CSV schema described in
   `docs/reference_equations.md`.
3. Place files under the directories above.
4. Re-run the test suite; numerical acceptance tests will then be eligible
   to run (still require independent expected values).

**Do not** use third-party mirrors or reverse-engineered calculator output
as the coefficient source for this module.

This PoC never embeds the numeric values; it only documents the expected
interface and validates structure when files are supplied.
