"""
Traffic Demand Prediction - UPGRADED Solution v2
=================================================
Uses: XGBoost, LightGBM, Extra Trees, Ridge + cross-validation stacking

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


try:
    import xgboost as xgb
    HAS_XGB = True
    print("✅ XGBoost available")
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not found — skipping (run: pip install xgboost)")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM available")
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not found — skipping (run: pip install lightgbm)")


print("\n📂 Loading data...")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print(f"  Train: {train.shape}  |  Test: {test.shape}")

print("\n🔧 Engineering features...")

def engineer_features(df):
    df = df.copy()

    # ── Timestamp ──
    if 'timestamp' in df.columns:
        try:
            ts = pd.to_numeric(df['timestamp'], errors='raise')
            dt = pd.to_datetime(ts, unit='s', errors='coerce')
        except Exception:
            dt = pd.to_datetime(df['timestamp'], errors='coerce')

        df['hour']      = dt.dt.hour
        df['minute']    = dt.dt.minute
        df['dayofweek'] = dt.dt.dayofweek
        df['month']     = dt.dt.month
        df['quarter']   = dt.dt.quarter
        df['is_weekend']= (df['dayofweek'] >= 5).astype(int)

        # Cyclical encoding (avoids 23→0 discontinuity)
        df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
        df['min_sin']   = np.sin(2*np.pi*df['minute']/60)
        df['min_cos']   = np.cos(2*np.pi*df['minute']/60)
        df['dow_sin']   = np.sin(2*np.pi*df['dayofweek']/7)
        df['dow_cos']   = np.cos(2*np.pi*df['dayofweek']/7)
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)

        # Time-of-day buckets
        df['time_bucket'] = pd.cut(df['hour'],
            bins=[-1,5,9,12,14,17,20,24],
            labels=[0,1,2,3,4,5,6]).astype(int)

        # Peak flags
        df['is_morning_peak'] = df['hour'].between(7,  9).astype(int)
        df['is_evening_peak'] = df['hour'].between(17, 19).astype(int)
        df['is_night']        = (~df['hour'].between(6, 22)).astype(int)
        df['is_lunch']        = df['hour'].between(12, 14).astype(int)

    # ── Geohash ──
    if 'geohash' in df.columns:
        gh = df['geohash'].astype(str)
        df['geo_len']     = gh.str.len()
        df['geo_prefix2'] = gh.str[:2]
        df['geo_prefix3'] = gh.str[:3]
        df['geo_prefix4'] = gh.str[:4]
        df['geo_prefix5'] = gh.str[:5]

    # ── Day column ──
    if 'day' in df.columns:
        df['day_mod7']  = df['day'] % 7
        df['day_mod30'] = df['day'] % 30
        df['day_sin']   = np.sin(2*np.pi*df['day']/7)
        df['day_cos']   = np.cos(2*np.pi*df['day']/7)

    # ── Interaction features ──
    if 'hour' in df.columns and 'dayofweek' in df.columns:
        df['hour_x_dow']     = df['hour'] * df['dayofweek']
        df['is_rush_weekday']= (df['is_morning_peak'] | df['is_evening_peak']) & (~df['is_weekend'].astype(bool))
        df['is_rush_weekday']= df['is_rush_weekday'].astype(int)

    if 'NumberofLanes' in df.columns and 'hour' in df.columns:
        df['lanes_x_hour'] = df['NumberofLanes'] * df['hour']

    return df

train = engineer_features(train)
test  = engineer_features(test)

# ── 3. ENCODE CATEGORICALS ────────────────────────────────────────────────────
print("🏷️  Encoding categoricals...")

cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks',
            'Temperature', 'Weather',
            'geo_prefix2', 'geo_prefix3', 'geo_prefix4', 'geo_prefix5']
cat_cols = [c for c in cat_cols if c in train.columns]

le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── Target-encode geohash (mean demand per geo area) ──
if 'geohash' in train.columns and 'demand' in train.columns:
    geo_mean = train.groupby('geohash')['demand'].mean().to_dict()
    train['geo_mean_demand'] = train['geohash'].map(geo_mean).fillna(train['demand'].mean())
    test['geo_mean_demand']  = test['geohash'].map(geo_mean).fillna(train['demand'].mean())

    geo_std = train.groupby('geohash')['demand'].std().to_dict()
    train['geo_std_demand'] = train['geohash'].map(geo_std).fillna(0)
    test['geo_std_demand']  = test['geohash'].map(geo_std).fillna(0)

# ── Hour-level mean demand ──
if 'hour' in train.columns and 'demand' in train.columns:
    hour_mean = train.groupby('hour')['demand'].mean().to_dict()
    train['hour_mean_demand'] = train['hour'].map(hour_mean)
    test['hour_mean_demand']  = test['hour'].map(hour_mean)

# ── 4. PREPARE MATRICES ───────────────────────────────────────────────────────
TARGET    = 'demand'
DROP_COLS = ['Index', 'index', TARGET, 'timestamp']

feature_cols = [c for c in train.columns
                if c not in DROP_COLS and c in test.columns]

print(f"  Total features: {len(feature_cols)}")

X      = train[feature_cols].fillna(-999).values
y      = train[TARGET].values
X_test = test[feature_cols].fillna(-999).values

# ── 5. CROSS-VALIDATED STACKING ───────────────────────────────────────────────
print("\n🔁 Running 5-Fold Cross-Validation + Stacking...")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage for OOF (out-of-fold) and test predictions per model
models_cfg = []

# GBM (tuned)
models_cfg.append(('GBM', GradientBoostingRegressor(
    n_estimators=600, learning_rate=0.04, max_depth=5,
    subsample=0.8, min_samples_leaf=8,
    max_features=0.8, random_state=42)))

# Extra Trees
models_cfg.append(('ExtraTrees', ExtraTreesRegressor(
    n_estimators=400, max_depth=12, min_samples_leaf=4,
    max_features=0.7, n_jobs=-1, random_state=42)))

# Random Forest (tuned)
models_cfg.append(('RF', RandomForestRegressor(
    n_estimators=400, max_depth=14, min_samples_leaf=3,
    max_features=0.7, n_jobs=-1, random_state=42)))

if HAS_XGB:
    models_cfg.append(('XGB', xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0)))

if HAS_LGB:
    models_cfg.append(('LGB', lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1)))

oof_preds  = np.zeros((len(X), len(models_cfg)))
test_preds = np.zeros((len(X_test), len(models_cfg)))

for m_idx, (name, model) in enumerate(models_cfg):
    print(f"\n  ▶ {name}")
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros((len(X_test), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        test_fold_preds[:, fold] = model.predict(X_test)

        fold_r2 = max(0, r2_score(y_val, oof[val_idx]))
        print(f"    Fold {fold+1} val R²: {fold_r2:.4f}")

    oof_preds[:, m_idx]  = oof
    test_preds[:, m_idx] = test_fold_preds.mean(axis=1)

    cv_r2 = max(0, r2_score(y, oof))
    print(f"    → {name} CV R²: {cv_r2:.4f}  (score: {100*cv_r2:.2f})")

# ── 6. META-LEARNER (Ridge stacking) ─────────────────────────────────────────
print("\n🧩 Stacking with Ridge meta-learner...")

scaler = RobustScaler()
oof_scaled  = scaler.fit_transform(oof_preds)
test_scaled = scaler.transform(test_preds)

meta = Ridge(alpha=1.0)
meta.fit(oof_scaled, y)

oof_meta   = meta.predict(oof_scaled)
final_pred = meta.predict(test_scaled)
final_pred = np.clip(final_pred, 0, None)

stack_r2 = max(0, r2_score(y, oof_meta))
print(f"  Stacked CV R²: {stack_r2:.4f}  →  Score: {100*stack_r2:.2f}/100")
print(f"  Meta-learner weights: {dict(zip([n for n,_ in models_cfg], meta.coef_.round(3)))}")

# ── 7. SAVE SUBMISSION ────────────────────────────────────────────────────────
print("\n💾 Saving submission.csv...")
index_col = 'Index' if 'Index' in test.columns else 'index'
submission = pd.DataFrame({
    'Index' : test[index_col].values,
    'demand': final_pred
})
submission.to_csv("submission.csv", index=False)
print(f"  Saved {len(submission)} rows → submission.csv")
print(f"\n✅ Done! Estimated score: {100*stack_r2:.2f}/100")
