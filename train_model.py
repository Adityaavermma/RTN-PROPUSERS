import joblib, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('cleaned_property_data.csv')      # run clean_and_model.py first
cat, num = ['city', 'locality', 'property_type'], ['bhk', 'area_sqft']
X, y = df[cat + num], df['price_lakh']
pre = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), cat)], remainder='passthrough')

models = {'Ridge': make_pipeline(pre, Ridge(alpha=1.0)),
          'RandomForest': make_pipeline(pre, RandomForestRegressor(300, random_state=42))}

# 5-fold cross-validation (more reliable than one split on small data)
for n, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring='r2')
    print(f'{n}: CV R2 = {s.mean():.2f} (+/- {s.std():.2f})')

# train the best one on train split, check on test, then save
best = max(models, key=lambda n: cross_val_score(models[n], X, y, cv=5, scoring='r2').mean())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = models[best].fit(Xtr, ytr)
p = model.predict(Xte)
print(f'\nBest: {best} | Test R2={r2_score(yte, p):.2f}, MAE={mean_absolute_error(yte, p):.1f} lakh')

model.fit(X, y)                                    # final fit on all data
joblib.dump(model, 'price_model.joblib')
print('Saved: price_model.joblib')

# example prediction
new = pd.DataFrame([{'city': 'Noida', 'locality': 'Sector 137', 'property_type': 'Apartment', 'bhk': 3, 'area_sqft': 1500}])
print(f"\nPredicted price for 3 BHK, 1500 sqft, Sector 137: {model.predict(new)[0]:.0f} lakh")
