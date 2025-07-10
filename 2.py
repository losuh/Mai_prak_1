import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

train = sparse.load_npz("train.npz")

n_components = 20
n_clusters = 10

svd = TruncatedSVD(n_components=n_components, random_state=42)
scaler = StandardScaler(with_mean=False)
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
pipeline = make_pipeline(svd, scaler, kmeans)

cluster_labels = pipeline.fit_predict(train)

submission = pd.DataFrame({
    'ID': np.arange(train.shape[0]),
    'TARGET': cluster_labels
})

submission.to_csv('submission.csv', index=False)