# Property Sales Data — Cleaning & Modeling Practice

This is a **practice project** using fake (synthetically generated) property
listing data, styled after the kind of sales data a real estate platform
like PropUsers might work with. It is for practicing data cleaning and
basic machine learning — **not real PropUsers data or results.**

## Files

| File | What it is |
|---|---|
| `make_data.py` | Generates the fake, deliberately messy dataset. |
| `practice_property_data.csv` | The raw messy data (94 rows) — duplicates, missing values, inconsistent text, mixed date formats, text-formatted prices, planted outliers. |
| `clean_and_model.py` | Cleans the raw data and runs 3 practice models (anomaly detection, price regression, sold/not-sold classification). |
| `cleaned_property_data.csv` | Output of the cleaning step — ready-to-use, tidy data. |
| `train_model.py` | Trains and saves a price-prediction model using cross-validation. |
| `price_model.joblib` | The final trained model, saved to disk. |

## How to run

Install requirements once:
```bash
pip install pandas numpy scikit-learn joblib
```

Then run in order (each step depends on the file before it):
```bash
python make_data.py          # (optional) regenerate the raw messy CSV
python clean_and_model.py    # clean data + run the 3 practice models
python train_model.py        # train & save the final price model
```

## What `clean_and_model.py` does

**Cleaning:**
- Removes duplicate rows
- Standardizes text (city, locality, property type)
- Fixes inconsistent date formats
- Converts prices written as text (`"1.25 Cr"`, `"Rs. 85 Lac"`) into plain numbers (lakhs)
- Fills missing values (area, price, agent, locality) using sensible group averages
- Adds derived columns: `price_per_sqft`, `month`

**Models (for practice only):**
1. **Anomaly detection** (Isolation Forest) — flags listings that look like data-entry errors (e.g. an unrealistic area or price).
2. **Price prediction** (Linear Regression vs. Random Forest) — predicts a listing's price from its city, locality, type, size, and BHK.
3. **Sold-or-not prediction** (Random Forest Classifier) — predicts whether a listing sells, based on location, price, lead source, and days on market.

It also prints simple business insights (e.g. average price per sq ft by locality, sold rate by lead source).

## `train_model.py` — the final model

Trains a price-prediction model on the *cleaned* data using 5-fold
cross-validation to pick the better of two models (Ridge Regression vs.
Random Forest), evaluates it on a held-out test set, then re-trains on
all the data and saves it as `price_model.joblib`. You can reload it with:

```python
import joblib
model = joblib.load('price_model.joblib')
model.predict(new_data)   # new_data must have the same columns as training
```

## Important note

All data here is synthetic — generated for practice, not scraped or real.
**Do not present these numbers, accuracy scores, or insights as actual
PropUsers results.** Use this project to practice and explain your
*process* (cleaning, anomaly detection, modeling), and keep any real
internship claims backed by your actual work there.
