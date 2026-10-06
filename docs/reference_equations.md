# Spirometry reference equations – design notes (PoC)

**Disclaimer**  
This module produces a *comparison against published physiological reference
values*.  It does **not** constitute a medical diagnosis, treatment
recommendation, or clinical decision-support output.

---

## 1. Supported methods

| Identifier (`ReferenceMethod`) | Description | Primary citation |
|--------------------------------|-------------|------------------|
| `gli_global` | GLI Global race-neutral equations (equal weighting of the four original ethnic groups) | Bowerman C et al. Am J Respir Crit Care Med. 2023;207:768-774. DOI: 10.1164/rccm.202205-0963OC |
| `gli_2012_other` | GLI-2012 composite “Other / Mixed” equation (average of the four ethnic-specific adjustments) | Quanjer PH et al. Eur Respir J. 2012;40:1324-1343. DOI: 10.1183/09031936.00080312 |

The two methods are implemented on **independent code paths**.  Selecting one
never silently falls back to the other.

---

## 2. Required inputs and units

| Parameter | Unit | Domain (documented) | Notes |
|-----------|------|---------------------|-------|
| `sex` | – | `male` / `female` | Biological sex required by the original equations |
| `age_years` | years (float) | [3, 95] | Fractional ages accepted |
| `height_cm` | centimetres | ≈ [100, 220] | Soft bounds; extreme values produce a warning once tables are present |
| measured FEV1 / FVC | litres | ≥ 0 | Optional |
| measured FEV1/FVC | dimensionless ratio | (0, 1.5] | Optional |

The caller **must** supply the reference method explicitly; the engine never
applies a default.

---

## 3. Output contract

For every requested index the engine returns an `IndexResult` containing:

* `measured` – the value supplied by the caller (or `None`)
* `predicted` – median (M) from the LMS model (or `None` if unsupported)
* `z_score` – LMS z-score (or `None`)
* `lln` – lower limit of normal, conventionally the value at z = −1.645 (or `None`)
* `unit` – `"L"` for volumes, `"ratio"` for FEV1/FVC
* `reference_id` – the method identifier (`gli_global` / `gli_2012_other`)
* `reference_version` – version string of the coefficient tables used
* `warnings` – any domain-edge notes
* `unsupported_reason` – populated only when the index is not covered

If an index is not supported by the selected equation set, `predicted`,
`z_score` and `lln` are set to `None` and a clear reason is recorded.  No
approximate substitute is fabricated.

---

## 4. LMS mathematics (identical for both methods)

The GLI equations use the LMS (Lambda-Mu-Sigma) method of Cole & Green.

$$
z = \begin{cases}
\dfrac{\bigl(\frac{y}{M}\bigr)^{L}-1}{L\cdot S} & L \neq 0 \\[1em]
\dfrac{\ln(y/M)}{S} & L = 0
\end{cases}
$$

The lower limit of normal is obtained by inverting the same transform at
$z = -1.64485$ (approximately the 5th percentile of a standard normal).

When $L = 0$ the implementation uses the limiting logarithmic form; no
division-by-zero occurs.

---

## 5. Coefficient / spline data schema

Official look-up tables are **not** redistributed with this repository.
They must be obtained from the GLI Network (https://www.lungfunction.org)
under the CC-BY-NC licence and placed under `src/reference/data/`.

### Expected file layout

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

### `coefficients.json` (illustrative schema)

```json
{
  "method": "gli_global",
  "version": "2022-bowerman-v1",
  "source": "Bowerman et al. AJRCCM 2023; supplementary Table E3",
  "licence": "CC-BY-NC",
  "accessed": "YYYY-MM-DD",
  "parameters": {
    "FEV1": {
      "male":   {"a": ..., "b_height": ..., "c_age": ..., "L": ...},
      "female": {"a": ..., "b_height": ..., "c_age": ..., "L": ...}
    },
    "FVC": { ... },
    "FEV1/FVC": { ... }
  }
}
```

### `m_spline.csv` / `s_spline.csv`

```
age,spline
3.0,0.0123
3.5,0.0145
...
95.0,-0.0032
```

Linear interpolation between tabulated ages is required (as described in the
original publications).

---

## 6. Population limitations

* The original GLI-2012 derivation used data from four broad groups
  (Caucasian, African-American, North-East Asian, South-East Asian).  
  Many world regions (including Iran and the wider Middle East) are
  underrepresented.
* Application of either equation set to an Iranian population is **not**
  by itself a local validation.  Local healthy-reference studies remain
  necessary for clinical adoption.
* GLI Global removes race/ethnicity as an input variable; it does not claim
  that the resulting predicted values are optimal for every geographic
  population.

---

## 7. Licence & attribution

* Equations: CC-BY-NC (GLI Network).  
  Appropriate acknowledgement of the original publications is required.
* This software module: independent PoC; does not claim endorsement by the
  GLI Network or any respiratory society.

---

## 8. Acceptance barrier for numerical results

Until the official tables are present under `src/reference/data/`, any call
that would require numerical evaluation raises `DataNotAvailableError`.
Contract, validation and pure-LMS tests pass; numerical reference tests are
skipped / marked as blocked.  No fabricated coefficients are used.
