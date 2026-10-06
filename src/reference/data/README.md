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

## How to obtain the data

1. Visit the official Global Lung Function Initiative site:  
   https://www.lungfunction.org
2. Download the look-up tables / software that accompany:
   - **GLI Global (race-neutral)** – Bowerman et al., Am J Respir Crit Care Med 2023  
     (supplementary Table E3 / look-up tables)
   - **GLI-2012** – Quanjer et al., Eur Respir J 2012  
     (Other/Mixed composite equations and associated splines)
3. Convert the tables into the simple CSV / JSON schema described in  
   `docs/reference_equations.md`.
4. Place the files under the directories above.
5. Re-run the test suite; numerical acceptance tests will then execute.

## Licence

The equations and tables are published under a **CC-BY-NC** licence.  
Commercial redistribution of the coefficient tables themselves is not permitted  
without explicit permission from the GLI Network.  This PoC never embeds the  
numeric values; it only documents the expected interface.
