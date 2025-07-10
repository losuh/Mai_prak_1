import pandas as pd
import lightgbm as lgb

train = pd.read_csv('../train.csv', sep=';')
train_labels = pd.read_csv('train_labels.csv', sep=';')
test = pd.read_csv('../test.csv', sep=';')
test_users = pd.read_csv('../test_users.csv', sep=';')
referer_vectors = pd.read_csv('../referer_vectors.csv', sep=';')
geo_info = pd.read_csv('../geo_info.csv', sep=';')

for df in [train, train_labels, test, test_users, referer_vectors, geo_info]:
    df.columns = df.columns.str.strip()

train = train.merge(train_labels, on='user_id', how='left')

train = train.merge(geo_info, on='geo_id', how='left')
test = test.merge(geo_info, on='geo_id', how='left')

train = train.merge(referer_vectors, on='referer', how='left')
test = test.merge(referer_vectors, on='referer', how='left')

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

train_features = aggregate_user_features(train)
test_features = aggregate_user_features(test)

train_features = train_features.merge(train_labels, on='user_id', how='left')

train_features = train_features.dropna(subset=['target'])

X = train_features.drop(columns=['user_id', 'target'])
y = train_features['target']
X = X.fillna(0)

X_test = test_features.drop(columns=['user_id'])
X_test = X_test.fillna(0)

model = lgb.LGBMClassifier(n_estimators=500, random_state=42)
model.fit(X, y)

preds = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'user_id': test_features['user_id'],
    'target': (preds > 0.5).astype(int)
})

submission.to_csv('submission.csv', index=False)
print("Файл сохранён")


