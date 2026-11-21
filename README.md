# Data Engineering & Numerical Analysis Portfolio

**Data Engineering & Numerical Analysis Portfolio.** Python ETL components for chunked CSV ingestion with retries, data-quality reporting and SQLite loading, plus from-scratch OLS with HC3 robust covariance and VIF diagnostics, and secant-method root-finding with convergence analysis. Demonstrations use synthetic data (50,000 ETL records; 200,000 environmental-risk records); they are not employer or capstone measurements.

## ETL pipeline

[`pipeline.py`](./pipeline.py) reads a CSV in chunks (`CSVExtractor`) or generates rows for a demo (`SyntheticExtractor`). The default run builds 50,000 synthetic transaction rows in chunks of 10,000. Each chunk is transformed — duplicate rows dropped, strings stripped and lowercased, rows missing `transaction_id` or `amount` dropped, negative amounts clipped, and log-amount and timestamp fields added — then appended to SQLite. Transform and load are retried up to 3 times. The script prints a data-quality report (row counts, and any column whose null share is above 10%) and writes `pipeline_output.db` (table `transactions`).

## Environmental risk scores

[`ecoli_risk_analysis.py`](./ecoli_risk_analysis.py) builds 200,000 synthetic sample rows and scores zip-code zones. These rows are generated; they are not field measurements. Features include log E. coli, log turbidity, an indicator for values above 235 CFU/100 mL, a rainfall interaction, and season. Zone scores are a weighted sum of min-max scaled components: mean log E. coli (0.45), exceedance rate (0.30), mean log turbidity (0.15), and mean rainfall interaction (0.10). Zones with fewer than 10 samples are dropped. A zone is flagged when its composite z-score is above 1.5. The script prints a table and writes `risk_scores.csv`.

## OLS with HC3 and VIF

[`ols_regression.py`](./ols_regression.py) fits ordinary least squares with NumPy (`pinv` of `X'X`), HC3 heteroskedasticity-robust covariance, t statistics, and two-sided p-values. It also computes a variance inflation factor for each regressor. The default run uses 2,000 synthetic housing rows (`price` on square footage, bedrooms, age, distance, and a school rating, plus an intercept) and prints the coefficient table. It does not use statsmodels.

## Secant method

[`secant_convergence.py`](./secant_convergence.py) searches for roots of `54x^6 + 45x^5 - 102x^4 - 69x^3 + 35x^2 + 16x - 4` from five start pairs, with tolerance `1e-10` and at most 100 iterations. For each start it prints the iterates, absolute error against the final iterate, the linear rate `e_i / e_{i-1}`, and the rate `e_i / (e_{i-1} e_{i-2})`. It saves `convergence.png` (error decay and linear rates).

## How to run

Verified with Python 3.12.3 after installing pandas 3.1.1, NumPy 2.5.5, SciPy 1.18.1, and Matplotlib 3.11.2. Each command below completed and wrote the output named here. Matplotlib's default backend in that run was Agg, so the plot was saved without a display.

```bash
python3 -m pip install pandas numpy scipy matplotlib
python3 pipeline.py
python3 ecoli_risk_analysis.py
python3 ols_regression.py
python3 secant_convergence.py
```

`pipeline.py` appends to `pipeline_output.db`. Delete that file before running it again if you want a fresh table. `ecoli_risk_analysis.py` overwrites `risk_scores.csv`. `ols_regression.py` only prints. `secant_convergence.py` overwrites `convergence.png`.

## Author

Ahad Hashmani · [linkedin.com/in/ahadhashmani](https://linkedin.com/in/ahadhashmani) · ahadhashmani@utexas.edu
