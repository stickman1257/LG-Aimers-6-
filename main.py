import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import optuna
import warnings

# 경고 필터링 설정
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

optuna.logging.set_verbosity(optuna.logging.WARNING)

train = pd.read_csv('./data/train.csv').drop(columns=['ID'])
test = pd.read_csv('./data/test.csv').drop(columns=['ID'])
sample_submission = pd.read_csv('./data/sample_submission.csv')

y = train['임신 성공 여부']
train = train.drop('임신 성공 여부', axis=1)

drop_col = [
    "난자 해동 경과일",
    "PGD 시술 여부", 
    "PGS 시술 여부", 
    "착상 전 유전 검사 사용 여부",
    "임신 시도 또는 마지막 임신 경과 연수",
    "배아 해동 경과일"
]
train = train.drop(drop_col, axis=1)
test = test.drop(drop_col, axis=1)

categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "불임 원인 - 정자 면역학적 요인",
    "단일 배아 이식 여부",
    "대리모 여부", 
    "난자 출처", 
    "정자 출처", 
    "기증 배아 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    # "PGD 시술 여부", 
    # "PGS 시술 여부",     
    # "착상 전 유전 검사 사용 여부",
    "불임 원인 - 여성 요인",          
    "난자 채취 경과일"
]

numeric_columns = [
    # "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    # "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    # "배아 해동 경과일"
]

for col in train.columns:
    if train[col].isnull().sum() > 0:
        train[f"{col}_missing"] = train[col].isnull().astype(int)
        test[f"{col}_missing"] = test[col].isnull().astype(int)


train[categorical_columns] = train[categorical_columns].fillna("Missing")
test[categorical_columns] = test[categorical_columns].fillna("Missing")

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[categorical_columns] = ordinal_encoder.fit_transform(train[categorical_columns].astype(str))
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns].astype(str))

train[numeric_columns] = train[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(train[numeric_columns].median())
test[numeric_columns] = test[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(train[numeric_columns].median())


kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    cv_scores = []
    for train_idx, val_idx in kf.split(train, y):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params, use_label_encoder=False)
        model.fit(X_train, y_train)
        
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        preds = calibrated.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(cv_scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=1000)
best_xgb = study_xgb.best_params

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
    }
    
    cv_scores = []
    for train_idx, val_idx in kf.split(train, y):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params, verbose=0, loss_function='Focal:focal_alpha=0.5;focal_gamma=2.0')
        model.fit(X_train, y_train)
        
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        preds = calibrated.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(cv_scores)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=1000)
best_cat = study_cat.best_params

def objective_lgbm(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    cv_scores = []
    for train_idx, val_idx in kf.split(train, y):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        preds = calibrated.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(cv_scores)

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=1000)
best_lgbm = study_lgbm.best_params

base_models = [
    ("xgb", XGBClassifier(**best_xgb, use_label_encoder=False)),
    ("catboost", CatBoostClassifier(**best_cat, verbose=0, loss_function='Focal:focal_alpha=0.5;focal_gamma=2.0')),
    ("lgbm", LGBMClassifier(**best_lgbm))
]

meta_train = np.zeros((train.shape[0], len(base_models)))
meta_test = np.zeros((test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"\n▶ Generating Meta Features: {name.upper()}")
    test_preds = np.zeros((test.shape[0], kf.n_splits))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train, y)):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        
        meta_train[val_idx, i] = calibrated.predict_proba(X_val)[:, 1]
        test_preds[:, fold] = calibrated.predict_proba(test)[:, 1]
    
    meta_test[:, i] = test_preds.mean(axis=1)

def objective_meta(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
        'random_strength': trial.suggest_float('random_strength', 0.5, 2),
    }
    
    cv_scores = []
    for train_idx, val_idx in kf.split(meta_train, y):
        X_train, X_val = meta_train[train_idx], meta_train[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params, verbose=0, loss_function='Focal:focal_alpha=0.5;focal_gamma=2.0')
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
        preds = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(cv_scores)

study_meta = optuna.create_study(direction='maximize')
study_meta.optimize(objective_meta, n_trials=1000)
best_meta = study_meta.best_params

final_preds = np.zeros(test.shape[0])
for fold, (train_idx, val_idx) in enumerate(kf.split(meta_train, y)):
    print(f"Fold {fold+1}: Training Final Meta Model")
    
    meta_model = CatBoostClassifier(
        **best_meta,
        verbose=0,
        loss_function='Focal:focal_alpha=0.5;focal_gamma=2.0'
    )
    
    meta_model.fit(
        meta_train[train_idx], y.iloc[train_idx],
        eval_set=(meta_train[val_idx], y.iloc[val_idx]),
        use_best_model=True
    )
    
    final_preds += meta_model.predict_proba(meta_test)[:, 1] / kf.n_splits

sample_submission['probability'] = final_preds
sample_submission.to_csv('./submissions/optimized_stacking_optuna.csv', index=False)
print("\nFinal Training AUC:", roc_auc_score(y, meta_train.mean(axis=1)))

import json

best_params = {
    "XGBoost": study_xgb.best_params,
    "CatBoost": study_cat.best_params,
    "LightGBM": study_lgbm.best_params,
    "MetaModel": study_meta.best_params
}

with open('best_185_drop_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print("Best parameters saved to best_params.json")
