import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

y = np.log1p(train['price'])

drop_cols = ['id', 'date', 'price']
features = [col for col in train.columns if col not in drop_cols]

X = train[features].fillna(-1)
X_test = test[features].fillna(-1)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='l1',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

val_preds = model.predict(X_val)
val_preds_exp = np.expm1(val_preds)
val_true_exp = np.expm1(y_val)

mae = mean_absolute_error(val_true_exp, val_preds_exp)
print(f'MAE: {mae:.2f} | Score: {1 / (1 + mae):.5f}')

test_preds_log = model.predict(X_test)
test_preds = np.expm1(test_preds_log)

submission = sample_submission.copy()
submission['price'] = test_preds
submission.to_csv('submission.csv', index=False)
print("submission.csv сохранён!")
