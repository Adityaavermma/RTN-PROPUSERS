import re, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

df = pd.read_csv('practice_property_data.csv')
print('Raw shape:', df.shape); print(df.isna().sum(), '\n')

# ---------- CLEAN ----------
df = df.drop_duplicates()
for c in ['city', 'locality', 'property_type']:
    df[c] = df[c].str.strip().str.title().fillna('Unknown')
df['agent_id'] = df['agent_id'].fillna('Unknown')
df['listed_date'] = pd.to_datetime(df['listed_date'], format='mixed', dayfirst=True)
df['month'] = df['listed_date'].dt.month

def to_lakh(v):
    if pd.isna(v): return np.nan
    s = str(v).lower().replace('rs.', '').strip()
    n = float(re.findall(r'[\d.]+', s)[0])
    return n * 100 if 'cr' in s else n          # 1 Cr = 100 Lakh
df['price_lakh'] = df['price_lakh'].apply(to_lakh)
df['area_sqft'] = pd.to_numeric(df['area_sqft'], errors='coerce')
df['area_sqft'] = df['area_sqft'].fillna(df.groupby('property_type')['area_sqft'].transform('median'))
df['price_lakh'] = df['price_lakh'].fillna(df.groupby(['locality', 'property_type'])['price_lakh'].transform('median'))
df['price_lakh'] = df['price_lakh'].fillna(df['price_lakh'].median())
df['bhk'] = df['bhk'].fillna(0)
df['price_per_sqft'] = df['price_lakh'] * 1e5 / df['area_sqft']

# ---------- MODEL A: ANOMALY DETECTION ----------
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(df[['price_per_sqft', 'area_sqft']]) == -1
print('Suspicious listings:\n', df.loc[df['anomaly'], ['listing_id', 'locality', 'area_sqft', 'price_lakh', 'price_per_sqft']].round(0), '\n')

clean = df[(df['days_on_market'] >= 0) & (~df['anomaly'])].copy()
clean.to_csv('cleaned_property_data.csv', index=False)
print('Clean shape:', clean.shape, '\n')

# ---------- MODEL B: REGRESSION (predict price) ----------
X = pd.get_dummies(clean[['city', 'locality', 'property_type', 'bhk', 'area_sqft']], drop_first=True)
y = clean['price_lakh']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
for n, m in [('Linear Regression', LinearRegression()), ('Random Forest', RandomForestRegressor(200, random_state=42))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    print(f'{n}: R2={r2_score(yte, p):.2f}, MAE={mean_absolute_error(yte, p):.1f} lakh')

# ---------- MODEL C: CLASSIFICATION (will it sell?) ----------
Xc = pd.get_dummies(clean[['locality', 'property_type', 'lead_source', 'days_on_market', 'price_per_sqft']], drop_first=True)
yc = (clean['status'] == 'Sold').astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=42)
clf = RandomForestClassifier(200, random_state=42).fit(Xtr, ytr)
print(f'Sold-or-not accuracy: {accuracy_score(yte, clf.predict(Xte)):.2f}')
print('Top drivers:\n', pd.Series(clf.feature_importances_, Xc.columns).nlargest(4).round(3))

# ---------- INSIGHTS ----------
print('\nAvg price/sqft by locality:\n', clean.groupby('locality')['price_per_sqft'].mean().sort_values(ascending=False).round(0))
print('\nSold rate by lead source:\n', clean.groupby('lead_source')['status'].apply(lambda s: (s == 'Sold').mean()).round(2))
