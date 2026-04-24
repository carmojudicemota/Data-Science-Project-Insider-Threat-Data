from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    KFold,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Pasta onde vão ficar os gráficos
OUTPUT_DIR = "figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# FUNÇÕES AUXILIARES

# guarda gráficos
def save_and_show(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# datas
def preprocess_log(df, date_col="date"):
    df = df.copy()
    df = df.drop_duplicates(subset="id")  # tirar linhas repetidas
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])  # se a data for inválida, eliminamos
    return df

# correlação alta
def print_high_corr_pairs(df, threshold=0.90, title=""):
    if title:
        print(f"\n=== {title} ===")
    corr_abs = df.corr().abs()
    pairs = []
    for col1 in corr_abs.columns:
        for col2 in corr_abs.columns:
            if col1 < col2 and corr_abs.loc[col1, col2] > threshold:
                pairs.append((col1, col2, corr_abs.loc[col1, col2]))
    if pairs:
        print(f"Pares com correlação absoluta > {threshold:.2f}:")
        for c1, c2, val in pairs:
            print(f"  {c1} - {c2}: {val:.3f}")
    else:
        print(f"Não foram encontrados pares com correlação absoluta > {threshold:.2f}.")

# inercia, silhueta
def clustering_cv_scores(X_scaled, k_range, random_state=42):
    inertias = []
    silhouette_means = []
    silhouette_stds = []
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for k in k_range:
        fold_scores = []
        for train_idx, _ in kf.split(X_scaled):
            X_fold_train = X_scaled[train_idx]
            kmeans_fold = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels_fold = kmeans_fold.fit_predict(X_fold_train)
            if len(np.unique(labels_fold)) > 1:
                fold_scores.append(silhouette_score(X_fold_train, labels_fold))
        kmeans_full = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_full = kmeans_full.fit_predict(X_scaled)
        inertias.append(kmeans_full.inertia_)
        silhouette_means.append(np.mean(fold_scores))
        silhouette_stds.append(np.std(fold_scores))
    return inertias, silhouette_means, silhouette_stds


# PREPROCESSAMENTO DOS DADOS
logon_raw = pd.read_csv("logon.csv")
device_raw = pd.read_csv("device.csv")
email_raw = pd.read_csv("email.csv")
file_raw = pd.read_csv("file.csv")

"""
# estrutura do dataset
print(logon_raw.head())
print(device_raw.head())
print(email_raw.head())
print(file_raw.head())

print(logon_raw.info())
print(device_raw.info())
print(logon_raw["user"].value_counts().head())

"""

logon_df = preprocess_log(logon_raw)
device_df = preprocess_log(device_raw)
email_df = preprocess_log(email_raw)
file_df = preprocess_log(file_raw)


"""
print(activity_counts.head())
"""

# FEATURE ENGINEERING
# logon
pc_features = logon_df.groupby(["user", "activity"]).size().unstack(fill_value=0)
pc_features.columns = [f"pc_{col.lower()}" for col in pc_features.columns]
for col in ["pc_logon", "pc_logoff"]:
    if col not in pc_features.columns:
        pc_features[col] = 0

# conversão datas para a parte da night e do weekend
logon_df["hour"] = logon_df["date"].dt.hour
logon_df["day_of_week"] = logon_df["date"].dt.dayofweek
logon_df["is_weekend"] = logon_df["day_of_week"].isin([5, 6]).astype(int)

night_logons = logon_df[
    (logon_df["activity"] == "Logon") &
    (logon_df["hour"].between(0, 6))
    ]
pc_features["pc_logins_night"] = night_logons.groupby("user").size()

weekend_logons = logon_df[
    (logon_df["activity"] == "Logon") &
    (logon_df["is_weekend"] == 1)
    ]
pc_features["pc_logins_weekend"] = weekend_logons.groupby("user").size()

pc_features["pc_logins_night"] = pc_features["pc_logins_night"].fillna(0)
pc_features["pc_logins_weekend"] = pc_features["pc_logins_weekend"].fillna(0)
# nos racios usamos sempre +1 para evitar divisão por zero
pc_features["night_logon_ratio"] = pc_features["pc_logins_night"] / (pc_features["pc_logon"] + 1)
pc_features["weekend_logon_ratio"] = pc_features["pc_logins_weekend"] / (pc_features["pc_logon"] + 1)

# device
device_features = device_df.groupby(["user", "activity"]).size().unstack(fill_value=0)
device_features.columns = [f"device_{col.lower()}" for col in device_features.columns]

for col in ["device_connect", "device_disconnect"]:
    if col not in device_features.columns:
        device_features[col] = 0

# email attachments
if "attachments" in email_df.columns:
    email_df["attachments"] = email_df["attachments"].fillna(0)

email_features = email_df.groupby("user").agg(num_emails=("id", "count"))
if "size" not in file_df.columns:
    file_df["size"] = file_df["content"].apply(lambda x: len(str(x)))
file_df["size"] = file_df["size"].fillna(0)
file_features = file_df.groupby("user").agg(
    total_file_size=("size", "sum"),
    num_files_downloaded=("id", "count")
)
# dataset
features = pc_features.join(device_features, how="left").fillna(0)
features = features.join(email_features, how="left").fillna(0)
features = features.join(file_features, how="left").fillna(0)

"""
print_high_corr_pairs(
    features.select_dtypes(include=["number"]),
    threshold=0.90,
    title="features com redundância"
)
"""

# novas features para adicionar, sem redundância
# vamos só manter racios para interpretação se for necessário
features["device_total_activity"] = features["device_connect"] + features["device_disconnect"]
features["avg_file_size"] = features["total_file_size"] / (features["num_files_downloaded"] + 1)
features["device_activity_per_logon"] = features["device_total_activity"] / (features["pc_logon"] + 1)
features["files_per_logon"] = features["num_files_downloaded"] / (features["pc_logon"] + 1)
features["files_per_device_activity"] = features["num_files_downloaded"] / (features["device_total_activity"] + 1)
features["emails_per_logon"] = features["num_emails"] / (features["pc_logon"] + 1)
# apagar as colunas redundantes
cols_to_drop = ["device_connect", "device_disconnect", "total_file_size"]
features = features.drop(columns=[col for col in cols_to_drop if col in features.columns])

features = features.fillna(0)
"""
print("\nFeatures finais")
print(features.head())
print("\nNúmero de utilizadores:", features.shape[0])
print("Número de features:", features.shape[1])
print(features.dtypes)
"""

# ESTATÍSTICAS DESCRITIVAS DO DATASET GERADO

users = features.index
numeric_cols = features.select_dtypes(include=["number"])

desc_stats = numeric_cols.describe().round(2)
desc_stats.to_csv("estatisticas_descritivas.csv")

corr_matrix = numeric_cols.corr()

#gráfico do novo heatmap (sem as correlacionadas)
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação das Features (após redução de redundância)")
save_and_show("correlacao_features.png")

# Histogramas
num_features = min(len(numeric_cols.columns), 12)
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols.columns[:num_features]):
    axes[i].hist(numeric_cols[col], bins=30, edgecolor="black", alpha=0.7)
    axes[i].set_title(f"Distribuição de {col}", fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("Frequência", fontsize=8)
    axes[i].grid(True, alpha=0.3)

for j in range(num_features, len(axes)):
    axes[j].axis("off")

plt.suptitle("Distribuições das Features", fontsize=14)
save_and_show("distribuicoes_features.png")

# Boxplots
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols.columns[:num_features]):
    bp = axes[i].boxplot(numeric_cols[col], vert=True, patch_artist=True)
    axes[i].set_title(f"Boxplot de {col}", fontsize=10)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].grid(True, alpha=0.3)

    for box in bp['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)

for j in range(num_features, len(axes)):
    axes[j].axis("off")

plt.suptitle("Boxplots das Features", fontsize=14)
save_and_show("boxplots_features.png")

# CLUSTERING
X = numeric_cols.copy()
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

K_range = list(range(2, 11)) # mais do que 10 já é uma interpretação muito difícil de explicar

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

inertias_std, silhouette_means_std, silhouette_stds_std = clustering_cv_scores(
    X_train_scaled, K_range, random_state=42
)

for i, k in enumerate(K_range):
    print(
        f"[StandardScaler] k={k}: Inércia={inertias_std[i]:.0f}, "
        f"Silhueta média={silhouette_means_std[i]:.3f} "
        f"(+/- {silhouette_stds_std[i]:.3f})"
    )
best_k = K_range[np.argmax(silhouette_means_std)]
best_sil_std = max(silhouette_means_std)

"""
print(f"\nMelhor k SS: {best_k}")
print(f"Melhor silhueta média SS: {best_sil_std:.3f}")
"""

# gráfico do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias_std, marker="o", linewidth=2, markersize=8)
plt.title("Elbow Method - StandardScaler", fontsize=12)
plt.xlabel("Número de clusters", fontsize=10)
plt.ylabel("Inércia", fontsize=10)
plt.grid(True, alpha=0.3)
save_and_show("elbow_train_standard.png")

# gráfico da silhueta
plt.figure(figsize=(10, 5))
plt.errorbar(K_range, silhouette_means_std, yerr=silhouette_stds_std,
             marker="o", capsize=5, linewidth=2, markersize=8)
plt.title("Silhouette Score vs K - StandardScaler", fontsize=12)
plt.xlabel("Número de clusters", fontsize=10)
plt.ylabel("Silhouette score", fontsize=10)
plt.grid(True, alpha=0.3)
save_and_show("silhouette_train_standard.png")

# k means
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_train = kmeans.fit_predict(X_train_scaled)
cluster_test = kmeans.predict(X_test_scaled)

X_train_df_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_train_df_scaled["cluster"] = cluster_train

X_test_df_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
X_test_df_scaled["cluster"] = cluster_test

y_test = cluster_test


#  des-escalonizar e resumir
X_train_original = X_train.copy()
X_train_original["cluster"] = cluster_train
cluster_summary_original = X_train_original.groupby("cluster").mean().round(2)
cluster_summary_original["contagem"] = X_train_original.groupby("cluster").size()
"""
print("\n Resumo Clusters")
print(cluster_summary_original)
"""
cluster_summary_original.to_csv("sumario_clusters_escala_original.csv")

# RobustScaler (porque há muitos outliers, fazer a comparação com o SS)
# Depois acabamos por escolher o SS mas importante para explicar pq essa decisão
scaler_rob = RobustScaler()
X_train_rob = scaler_rob.fit_transform(X_train)
X_test_rob = scaler_rob.transform(X_test)

inertias_rob, silhouette_means_rob, silhouette_stds_rob = clustering_cv_scores(
    X_train_rob, K_range, random_state=42
)

for i, k in enumerate(K_range):
    print(
        f"[RobustScaler] k={k}: Inércia={inertias_rob[i]:.0f}, "
        f"Silhueta média={silhouette_means_rob[i]:.3f} "
        f"(+/- {silhouette_stds_rob[i]:.3f})"
    )

best_k_rob = K_range[np.argmax(silhouette_means_rob)]
best_sil_rob = max(silhouette_means_rob)

"""
print(f"\nMelhor k RS: {best_k_rob}")
print(f"Melhor silhueta média RS: {best_sil_rob:.3f}")
"""

#gráficos do RS - mesmos que o ss
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias_rob, marker="o", linewidth=2, markersize=8, color='green')
plt.title("Elbow Method - RobustScaler", fontsize=12)
plt.xlabel("Número de clusters", fontsize=10)
plt.ylabel("Inércia", fontsize=10)
plt.grid(True, alpha=0.3)
save_and_show("elbow_train_robust.png")
plt.figure(figsize=(10, 5))
plt.errorbar(K_range, silhouette_means_rob, yerr=silhouette_stds_rob,
             marker="o", capsize=5, linewidth=2, markersize=8, color='green')
plt.title("Silhouette Score vs K - RobustScaler", fontsize=12)
plt.xlabel("Número de clusters", fontsize=10)
plt.ylabel("Silhouette score", fontsize=10)
plt.grid(True, alpha=0.3)
save_and_show("silhouette_train_robust.png")

# PCA com RS
kmeans_rob = KMeans(n_clusters=best_k_rob, random_state=42, n_init=10)
cluster_train_rob = kmeans_rob.fit_predict(X_train_rob)
cluster_test_rob = kmeans_rob.predict(X_test_rob)

pca_rob = PCA(n_components=2)
X_train_pca_rob = pca_rob.fit_transform(X_train_rob)
X_test_pca_rob = pca_rob.transform(X_test_rob)

"""
print(f"Variância explicada PCA-2 (RS): {pca_rob.explained_variance_ratio_.sum():.2%}")
print(f"Variância por componente: {pca_rob.explained_variance_ratio_}")
"""

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_train_pca_rob[:, 0], X_train_pca_rob[:, 1],
                       c=cluster_train_rob, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter1, label="Cluster")
plt.xlabel(f"PC1 ({pca_rob.explained_variance_ratio_[0]:.1%})", fontsize=10)
plt.ylabel(f"PC2 ({pca_rob.explained_variance_ratio_[1]:.1%})", fontsize=10)
plt.title("PCA 2D (treino) com clusters - RobustScaler", fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_test_pca_rob[:, 0], X_test_pca_rob[:, 1],
                       c=cluster_test_rob, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter2, label="Cluster")
plt.xlabel(f"PC1 ({pca_rob.explained_variance_ratio_[0]:.1%})", fontsize=10)
plt.ylabel(f"PC2 ({pca_rob.explained_variance_ratio_[1]:.1%})", fontsize=10)
plt.title("PCA 2D (teste) com clusters previstos - RobustScaler", fontsize=12)
plt.grid(True, alpha=0.3)

plt.suptitle("Análise de Clusters com PCA - RobustScaler", fontsize=14)
plt.tight_layout()
save_and_show("pca_clusters_robustscaler.png")

# PCA com SS
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

"""
print(f"\nVariância explicada PCA-2 (SS): {pca.explained_variance_ratio_.sum():.2%}")
print(f"Variância por componente: {pca.explained_variance_ratio_}")
"""

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                      c=cluster_train, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
plt.title("PCA 2D (treino) com clusters - StandardScaler", fontsize=12)
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                      c=cluster_test, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.xlabel("PC1", fontsize=10)
plt.ylabel("PC2", fontsize=10)
plt.title("PCA 2D (teste) com clusters previstos - StandardScaler", fontsize=12)
plt.grid(True, alpha=0.3)

plt.suptitle("Análise de Clusters com PCA - StandardScaler", fontsize=14)
plt.tight_layout()
save_and_show("pca_clusters_standard.png")

# SUPERVISIONADA - explicado no trabalho a razão de usar os pseudo-labels do K-means, sem groundtruth (que não há)
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced"),
    "SVM": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced")
}

#Só nos dados de treino, a partir da divisão já feita para evitar dataleakage
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, cluster_train)

# RF
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid_rf,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)
grid_rf.fit(X_train_res, y_train_res)
"""
print("Melhores parâmetros Random Forest:", grid_rf.best_params_)
"""
models["Random Forest"] = grid_rf.best_estimator_

# SVM
param_grid_svm = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto", 0.1, 1]
}
grid_svm = GridSearchCV(
    SVC(kernel="rbf", random_state=42, class_weight="balanced"),
    param_grid_svm,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)
grid_svm.fit(X_train_res, y_train_res)
"""
print("Melhores parâmetros SVM:", grid_svm.best_params_)
"""
models["SVM"] = grid_svm.best_estimator_

# DT
param_grid_dt = {
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    param_grid_dt,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)
grid_dt.fit(X_train_res, y_train_res)
"""
print("Melhores parâmetros Decision Tree:", grid_dt.best_params_)
"""
models["Decision Tree"] = grid_dt.best_estimator_


# Avaliar os modelos
# passa mais por uma avaliação de qual deles reproduz melhor resultados K-means, não que é mais generalizavel
predictions = {}
metrics_list = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fitted_models = {}

for name, clf in models.items():
    #SMOTE dentro de cada fold tmb para evitar dataleakage
    pipe = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", clf)
    ])
    cv_scores = cross_val_score(
        pipe,
        X_train_scaled,
        cluster_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )
    print(f"\n{name} - CV F1-score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, cluster_train)
    clf.fit(X_train_res, y_train_res)
    fitted_models[name] = clf
    y_pred = clf.predict(X_test_scaled)
    predictions[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    metrics_list.append({
        "Modelo": name,
        "Accuracy": round(acc, 3),
        "Precision (macro)": round(prec, 3),
        "Recall (macro)": round(rec, 3),
        "F1-score (macro)": round(f1, 3),
        "CV F1-score": round(cv_scores.mean(), 3)
    })

comparison_df = pd.DataFrame(metrics_list)
"""
print("Comparação dos modelos:")
print(comparison_df)
print(comparison_df)
"""
comparison_df.to_csv("comparacao_modelos.csv", index=False)

# gráfico de barras do F1 macro (pq dados pouco balanced)
plt.figure(figsize=(10, 6))
bars = plt.bar(comparison_df["Modelo"], comparison_df["F1-score (macro)"],
               color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title("F1 macro (teste holdout) - clusters obtidos com StandardScaler", fontsize=14)
plt.ylabel("F1 (macro)", fontsize=12)
plt.xlabel("", fontsize=12)
plt.ylim(0, 1.05)
plt.xticks(rotation=0, fontsize=11)

for bar, value in zip(bars, comparison_df["F1-score (macro)"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.grid(True, alpha=0.3, axis='y')
save_and_show("f1_macro_models.png")

# matrizes de confusão para todos os modelos
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
models_to_plot = ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"]

for ax, name in zip(axes.ravel(), models_to_plot):
    model = fitted_models[name]
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=[f'Cluster {i}' for i in range(len(np.unique(y_test)))],
                yticklabels=[f'Cluster {i}' for i in range(len(np.unique(y_test)))])
    ax.set_title(f'{name}', fontsize=12)
    ax.set_xlabel('Previsto', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)

plt.suptitle('Matrizes de Confusão Normalizadas por Modelo', fontsize=14)
plt.tight_layout()
save_and_show("confusion_matrices_all_models.png")

# Feature importance RF
rf_clf = fitted_models["Random Forest"]
feat_imp = pd.Series(rf_clf.feature_importances_, index=X_train.columns)

"""
print("\n Top 10 features RF")
print(feat_imp.sort_values(ascending=False).head(10))
"""

plt.figure(figsize=(10, 6))
feat_imp_sorted = feat_imp.sort_values(ascending=True)
top_features = feat_imp_sorted.tail(10)
bars = plt.barh(range(len(top_features)), top_features.values, color='steelblue')
plt.title("Top 10 Features mais Importantes (Random Forest)", fontsize=14)
plt.xlabel("Importância", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.yticks(range(len(top_features)), top_features.index, fontsize=10)

for i, (bar, value) in enumerate(zip(bars, top_features.values)):
    plt.text(value + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{value:.3f}', ha='left', va='center', fontsize=9)

plt.grid(True, alpha=0.3, axis='x')
save_and_show("feature_importance_rf.png")

"""
for model_name in ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"]:
    model = fitted_models[model_name]
    y_pred = model.predict(X_test_scaled)
    print(f"\n{model_name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
"""

# Árvore de Decisão Gráficos
dt_clf = fitted_models["Decision Tree"]

plt.figure(figsize=(30, 15))
plot_tree(
    dt_clf,
    feature_names=X_train.columns,
    class_names=[str(i) for i in range(best_k)],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Árvore de Decisão Completa", fontsize=14)
save_and_show("arvore_decisao.png")

plt.figure(figsize=(30, 15))
plot_tree(
    dt_clf,
    feature_names=X_train.columns,
    class_names=[str(i) for i in range(best_k)],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3,
    precision=2,
    proportion=True
)
plt.title("Árvore de Decisão Simplificada (max_depth=3)", fontsize=14)
save_and_show("arvore_decisao_maxdepth3.png")

# Ver overfitting - não é generalizavel de qlq forma : ajustamento interno
for name, clf in fitted_models.items():
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, cluster_train)
    train_score = clf.score(X_train_res, y_train_res)
    test_score = clf.score(X_test_scaled, y_test)
    gap = train_score - test_score
    print(f"{name}: Treino={train_score:.3f}, Teste={test_score:.3f}, Gap={gap:.3f}")
    if gap > 0.10:
        print(f"  Possível sobreajustamento em {name}")
    elif gap < -0.05:
        print(f"  Possível subajustamento em {name}")
    else:
        print(f"  Sem sobre/subajustamento em {name}")




print(f"\n The end.........")


