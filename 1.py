import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["text"] = train["url"].astype(str) + " " + train["title"].astype(str)
test["text"] = test["url"].astype(str) + " " + test["title"].astype(str)

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])
y = train["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print("F1 score:", f1_score(y_val, y_pred))

test_preds = clf.predict(X_test)

submission = pd.DataFrame({
    "ID": test["ID"],
    "label": test_preds
})

submission.to_csv("submission.csv", index=False)
