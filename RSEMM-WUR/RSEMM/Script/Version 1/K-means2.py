import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load your dataset ===
# Make sure this file is in your working directory
df = pd.read_csv("Repos (13 Factors for K-means).csv")
df.set_index("RepoID", inplace=True)

# === 2. Use 11 SE-focused features (exclude AI Model Usage, GenAI Used) ===
se_features = [
    'Testing', 'Security', 'Research Support', 'Project Activity / Maintenance',
    'Popularity / Engagement', 'Licensing & Compliance', 'Documentation',
    'Dependency Analysis', 'Collaboration', 'Code Quality', 'Agile Process'
]

# === 3. Standardize the feature matrix ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[se_features])

# === 4. Perform KMeans clustering (k=2) ===
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
df['Initial Cluster'] = kmeans.fit_predict(X_scaled)

# === 5. Determine which cluster is RSE vs Exploratory based on mean profiles ===
cluster_profiles = df.groupby('Initial Cluster')[se_features].mean()
rse_cluster = cluster_profiles.mean(axis=1).idxmax()
exploratory_cluster = cluster_profiles.mean(axis=1).idxmin()

# === 6. Assign initial semantic labels ===
def initial_label(row):
    return "RSE" if row['Initial Cluster'] == rse_cluster else "Exploratory Coding"

df['Initial Label'] = df.apply(initial_label, axis=1)

# === 7. Refine labels using GenAI Used ===
def final_label(row):
    if row['GenAI Used'] > 0:
        return "AI(4)RSE" if row['Initial Label'] == "RSE" else "AI-Assisted Exploratory Coding"
    else:
        return row['Initial Label']

df['Final Cluster Label'] = df.apply(final_label, axis=1)

# === 8. PCA for visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# === 9. Plot the final 4 clusters ===
plt.figure(figsize=(10, 7))
sns.set(style='whitegrid', font_scale=1.2)
sns.scatterplot(
    data=df,
    x='PCA1',
    y='PCA2',
    hue='Final Cluster Label',
    palette='Set2',
    s=80,
    edgecolor='black'
)
plt.title("Final Clustering of Research Software Repositories", fontsize=16)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# === 10. Save the plot ===
plt.savefig("final_rse_clusters_plot.png", dpi=300)
plt.show()

# === 11. Save cluster-labeled data and profiles (optional) ===
df[['Initial Cluster', 'Initial Label', 'Final Cluster Label']].to_csv("rse_cluster_assignments.csv")
cluster_profiles.to_csv("rse_cluster_profiles.csv")

print("✅ Clustering complete. Plot and data saved.")


# Assuming you have 'pca' and 'se_features' already defined
import pandas as pd

pca_components = pd.DataFrame(
    data=pca.components_,
    columns=se_features,
    index=['PCA1', 'PCA2']
)

print(pca_components.T.sort_values(by='PCA1', ascending=False))  # To see which features contribute most
