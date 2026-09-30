"""
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification

Classifies iris flowers (setosa, versicolor, virginica) from their sepal and
petal measurements, compares several classifiers and evaluates the best one
on a held-out test set.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Load data --------------------------------------------------------------
df = pd.read_csv(BASE / "data" / "Iris.csv")
print("Shape:", df.shape)
print(df.head(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
print("Class balance:\n", df["species"].value_counts(), "\n")
print(df.describe(), "\n")

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 2. Exploratory data analysis ---------------------------------------------
sns.pairplot(df, hue="species", corner=True)
plt.savefig(OUT / "pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, features):
    sns.boxplot(data=df, x="species", y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig(OUT / "boxplots.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig(OUT / "correlation.png", dpi=120)
plt.close()

# 3. Train / test split -----------------------------------------------------
X, y = df[features], df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Compare models with 5-fold cross-validation ----------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
rows = []
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    rows.append({"Model": name, "CV accuracy": cv.mean(), "CV std": cv.std(),
                 "Test accuracy": test_acc})
results = pd.DataFrame(rows).sort_values(["CV accuracy", "Test accuracy"], ascending=False)
print("Model comparison:\n", results.to_string(index=False), "\n")
results.to_csv(OUT / "model_comparison.csv", index=False)

# 5. Evaluate best model ------------------------------------------------------
best_name = results.iloc[0]["Model"]
best = models[best_name]
y_pred = best.predict(X_test)
print(f"Best model: {best_name}")
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=best.classes_),
                       display_labels=best.classes_).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=120)
plt.close()

# Feature importance from the random forest
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
imp.plot.barh(figsize=(6, 4), title="Feature importance (Random Forest)")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=120)
plt.close()

# 6. Predict a new flower -----------------------------------------------------
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.9, 3.1, 5.4, 2.1]],
                      columns=features)
for vals, pred in zip(sample.values.tolist(), best.predict(sample)):
    print(f"Measurements {vals} -> predicted species: {pred}")
