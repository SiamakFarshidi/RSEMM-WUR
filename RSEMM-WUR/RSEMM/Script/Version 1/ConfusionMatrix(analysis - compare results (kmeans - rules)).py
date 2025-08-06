import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("k-means-rules-results.csv")  # Replace with your actual file path

# -------------------------------
# 1. Map numeric cluster to string labels
# -------------------------------

# Just assign the column directly
df["ClusterLabel"] = df["Cluster (k-means)"].astype(str).str.strip()


# -------------------------------
# 2. Clean & standardize labels
# -------------------------------
df.columns = df.columns.str.strip()

# Strip whitespace and ensure string types
df["RS Category (4)"] = df["RS Category (4)"].astype(str).str.strip()
df["RS Category (3)"] = df["RS Category (3)"].astype(str).str.strip()
df["ClusterLabel"] = df["ClusterLabel"].astype(str).str.strip()

# Drop rows with missing labels
df = df.dropna(subset=["RS Category (4)", "RS Category (3)", "ClusterLabel"])

# -------------------------------
# 3. Compare 4-category labels
# -------------------------------
labels_4 = ["Exploratory Coding", "AI-Assisted Exploratory Coding", "RSE", "AI(4)RSE"]

print("Unique values in RS Category (4):", df["RS Category (4)"].unique())
print("Unique values in ClusterLabel:", df["ClusterLabel"].unique())

cm_4 = confusion_matrix(df["RS Category (4)"], df["ClusterLabel"], labels=labels_4)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_4, annot=True, fmt='d', xticklabels=labels_4, yticklabels=labels_4, cmap='Blues')
plt.title("Confusion Matrix: Rule-Based (4) vs K-Means")
plt.xlabel("K-Means Label")
plt.ylabel("Rule-Based Label (4)")
plt.tight_layout()
plt.show()

print("\n=== Classification Report (4-Category) ===")
print(classification_report(df["RS Category (4)"], df["ClusterLabel"], labels=labels_4))

kappa_4 = cohen_kappa_score(df["RS Category (4)"], df["ClusterLabel"], labels=labels_4)
print(f"Cohen's Kappa (4-Category): {kappa_4:.3f}")

# -------------------------------
# 4. Compare 3-category labels
# -------------------------------
def simplify(label):
    if label in ["Exploratory Coding", "AI-Assisted Exploratory Coding"]:
        return "Exploratory"
    elif label == "RSE":
        return "RSE"
    elif label == "AI(4)RSE":
        return "AI(4)RSE"
    return label

df["RuleLabel_3"] = df["RS Category (3)"].map(simplify)
df["ClusterLabel_3"] = df["ClusterLabel"].map(simplify)

labels_3 = ["Exploratory", "RSE", "AI(4)RSE"]

cm_3 = confusion_matrix(df["RuleLabel_3"], df["ClusterLabel_3"], labels=labels_3)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_3, annot=True, fmt='d', xticklabels=labels_3, yticklabels=labels_3, cmap='Greens')
plt.title("Confusion Matrix: Rule-Based (3) vs K-Means")
plt.xlabel("K-Means Label")
plt.ylabel("Rule-Based Label (3)")
plt.tight_layout()
plt.show()

print("\n=== Classification Report (3-Category) ===")
print(classification_report(df["RuleLabel_3"], df["ClusterLabel_3"], labels=labels_3))

kappa_3 = cohen_kappa_score(df["RuleLabel_3"], df["ClusterLabel_3"], labels=labels_3)
print(f"Cohen's Kappa (3-Category): {kappa_3:.3f}")
