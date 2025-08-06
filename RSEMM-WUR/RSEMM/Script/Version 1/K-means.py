
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# 1. Load the CSV
df = pd.read_csv("Repos (12 Factors for K-means).csv")  # replace with your actual file name

# 2. Set RepoID as index (optional)
df.set_index("RepoID", inplace=True)

# 3. Select only the 12 features
features = [
    'Testing', 'Security', 'Research Support', 'Project Activity / Maintenance',
    'Popularity / Engagement', 'Licensing & Compliance', 'Documentation',
    'Dependency Analysis', 'Collaboration', 'Code Quality', 'AI Usage', 'Agile Process'
]
X = df[features]

# 4. Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Elbow method to find optimal k
sse = []
K = range(1, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.plot(K, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.show()

# 6. Fit KMeans with k=4 (your 4 RS categories)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 7. Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2')
plt.title('K-Means Clusters of Research Software Repositories')
plt.show()
