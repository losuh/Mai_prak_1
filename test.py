import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

train = pd.read_csv('../train.csv', sep=';')
train_labels = pd.read_csv('train_labels.csv', sep=';')
referer_vectors = pd.read_csv('../referer_vectors.csv', sep=';')
geo_info = pd.read_csv('../geo_info.csv', sep=';')

for df in [train, train_labels, referer_vectors, geo_info]:
    df.columns = df.columns.str.strip()

train = train.sort_values('request_ts').reset_index(drop=True)
split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx]
val_part = train.iloc[split_idx:]

train_part = train_part.merge(train_labels, on='user_id', how='left')
val_part = val_part.merge(train_labels, on='user_id', how='left')

train_part = train_part.merge(geo_info, on='geo_id', how='left')
train_part = train_part.merge(referer_vectors, on='referer', how='left')

val_part = val_part.merge(geo_info, on='geo_id', how='left')
val_part = val_part.merge(referer_vectors, on='referer', how='left')


def aggregate_user_features(df):
    agg_funcs = {
        'geo_id': ['nunique', 'count'],
        'referer': ['nunique'],
        'user_agent': ['nunique'],
    }
    for i in range(10):
        agg_funcs[f'component{i}'] = ['mean', 'std']
    agg_df = df.groupby('user_id').agg(agg_funcs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    return agg_df.reset_index()

train_features = aggregate_user_features(train_part)
val_features = aggregate_user_features(val_part)

train_features = train_features.merge(train_labels, on='user_id', how='left')
val_features = val_features.merge(train_labels, on='user_id', how='left')

train_features = train_features.dropna(subset=['target'])
val_features = val_features.dropna(subset=['target'])

X_train = train_features.drop(columns=['user_id', 'target']).fillna(0)
y_train = train_features['target']
X_val = val_features.drop(columns=['user_id', 'target']).fillna(0)
y_val = val_features['target']

model = lgb.LGBMClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, val_preds)
auc = roc_auc_score(y_val, val_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC:  {auc:.4f}")


