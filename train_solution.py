from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from build_features import build_features
from metric import precision_at_recall

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
RANDOM_SEEDS = [17, 42, 2026]
CAT_FEATURES = [
    'ua_mode', 'ua_family_mode', 'ua_os_mode',
    'platform_mode', 'seller_mode', 'category_mode'
]
DROP_UNUSED_CATS = ['location_mode', 'query_mode', 'platform_raw_mode']


def make_model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=550,
        depth=5,
        learning_rate=0.042,
        loss_function='Logloss',
        class_weights=[1.0, 1.5],
        l2_leaf_reg=5,
        random_strength=0.7,
        max_ctr_complexity=1,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def main() -> None:
    train = pd.read_csv(
        DATA_DIR / 'train.csv',
        parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts'],
    )
    test = pd.read_csv(
        DATA_DIR / 'test.csv',
        parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts'],
    )
    events = pd.read_csv(DATA_DIR / 'events.csv.gz', parse_dates=['event_ts'])

    # build_features first filters events to
    # window_start_ts <= event_ts < window_end_ts, preventing look-ahead leakage.
    X_train = build_features(events, train)
    X_test = build_features(events, test)

    assert X_train['cookie_id'].tolist() == train['cookie_id'].tolist()
    assert X_test['cookie_id'].tolist() == test['cookie_id'].tolist()

    feature_cols = [
        c for c in X_train.columns
        if c != 'cookie_id' and c not in DROP_UNUSED_CATS
    ]
    y = train['target'].to_numpy()

    # Honest time validation: future days are validation.
    valid_mask = train['window_start_ts'].ge('2026-04-17').to_numpy()
    valid_predictions = []
    for seed in RANDOM_SEEDS:
        model = make_model(seed)
        model.fit(
            X_train.loc[~valid_mask, feature_cols],
            y[~valid_mask],
            cat_features=CAT_FEATURES,
        )
        valid_predictions.append(
            model.predict_proba(X_train.loc[valid_mask, feature_cols])[:, 1]
        )

    valid_score = np.mean(valid_predictions, axis=0)
    print(
        'Time validation P@R>=0.70:',
        round(precision_at_recall(y[valid_mask], valid_score), 6),
    )

    # Retrain on the complete labelled period and predict the later test period.
    test_predictions = []
    for seed in RANDOM_SEEDS:
        model = make_model(seed)
        model.fit(X_train[feature_cols], y, cat_features=CAT_FEATURES)
        test_predictions.append(model.predict_proba(X_test[feature_cols])[:, 1])

    # Round only at the final export step. This makes repeated CPU runs byte-stable
    # despite harmless ~1e-16 floating-point differences, without changing ranking.
    score = np.mean(test_predictions, axis=0).round(12)
    submission = pd.DataFrame({
        'cookie_id': test['cookie_id'],
        'score': score,
    })

    assert list(submission.columns) == ['cookie_id', 'score']
    assert len(submission) == len(test)
    assert submission['cookie_id'].is_unique
    assert submission['score'].between(0, 1).all()
    assert submission['score'].notna().all()

    submission.to_csv(ROOT / 'submission.csv', index=False)
    print('Saved:', ROOT / 'submission.csv')


if __name__ == '__main__':
    main()
