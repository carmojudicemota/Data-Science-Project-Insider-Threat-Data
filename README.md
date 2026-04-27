# User Behavior Segmentation for Insider-Risk Exploration

This work uses activity-based features built from user logon, device, email, and file datasets (CERT r4.2) to:

1. Engineer interpretable behavioral features
2. Segment users with K-Means clustering without using ground truth labels
3. Compare scaling strategies (StandardScaler vs RobustScaler)
4. Validate cluster reproducibility and study cluster structure with supervised models
5. Identify behavioral patterns that deserve human attention

**Key constraint:** Ground truth labels were deliberately ignored to replicate real-world scenarios where companies have no incident history. The goal is not to build the best insider classifier, but to answer: *Can we discover latent behavioral structure when no labels exist?*

---

## Project goal

The objective is to identify meaningful user behavior patterns from system activity logs and examine whether those patterns can be consistently separated and reproduced, without relying on pre-existing labels.

The analysis focuses on questions such as:

- Are there distinct behavioral groups among users?
- Which features matter most for separating those groups?
- Can supervised models reliably recover the discovered clusters?
- What models perform best and why?
- Is the segmentation stable across different data splits?

---

## Workflow summary

### Phase 1: Clustering (Unsupervised)

**Data preparation:**
- Integrated 4 logs: Logon, Devices, Email, Files (CERT dataset)
- Aggregated events to user level → 1000 users × 14 numeric features
- Removed duplicates, imputed zeros to preserve sample size

**Feature engineering (literature-inspired):**
- Intensity indicators: total logins, total logoffs, emails sent/received
- Timing features: night activity, weekend activity
- Normalized ratios: files per logon, emails per logon
- Device activity: total device connections/disconnections

**Clustering approach:**
- K-Means tested with k = 2 to 10 (k=2 is minimum for binary separation; k>10 is operationally impractical)
- Evaluation metrics: Elbow method (inertia) + Silhouette score (primary metric)
- Cross-validation with 5 folds to assess stability (mean silhouette ± standard deviation)
- PCA visualization (54.3% variance explained in 2D)

### Phase 2: Classification (Supervised Validation)

**Purpose:** Not redundancy: internal validity check + better interpretability + pratical use

**Method:**
- Pseudo-labels from K-Means → train classifiers to reproduce segmentation
- Train/test split: 80/20 holdout
- SMOTE applied only to training set to handle class imbalance (83% Cluster 0, 17% Cluster 1)
- Hyperparameter optimization with GridSearchCV (3-fold CV within training)

**Models compared:**
- Logistic Regression (linear boundary assumption)
- SVM with RBF kernel (flexible, non-linear)
- Random Forest (robust to outliers, captures interactions)
- Decision Tree (maximum interpretability)

---

## Main results

### Clustering: k=2 is the most stable solution

| Scaler | k=2 Silhouette | k=2 Std Dev | k=3 Silhouette | k=3 Std Dev |
|--------|---------------|-------------|----------------|-------------|
| StandardScaler | 0.574 | 0.046 | ~0.55 (lower) | 0.073 |
| RobustScaler | ~0.75 | — | 0.838 | — |

**Why choose StandardScaler with k=2?**
- RobustScaler attenuates outliers → better silhouette but loses sensitivity to rare behavior
- StandardScaler preserves influence of extreme activity → better for detecting anomalous patterns
- k=2 shows lower standard deviation (0.046) than k=3 (0.073) → more stable segmentation
- Binary separation (Regular vs. Intensive) is easier to interpret operationally
- k=2 with StandardScaler aligns with pedagogical constraints while maintaining sensitivity

**Important note:** K-Means is not an outlier detection method. The approach should not be confused with anomaly detection (Isolation Forest, DBSCAN would be more appropriate for that purpose).

### Classification: Near-perfect reproducibility

- All models achieved macro F1 > 0.98 on holdout
- Logistic Regression and SVM show almost identical performance (SVM marginal gain: +0.008 F1)
- This proves the cluster boundary is **approximately linear** in the transformed feature space

**Model recommendations:**
- **Logistic Regression:** Best trade-off (efficiency + interpretability + performance)
- **SVM:** Best absolute performance, but marginal gain doesn't justify complexity
- **Random Forest / Decision Tree:** Most interpretable for understanding feature importance

**Key insight:** High performance on pseudo-labels does NOT imply real threat detection, only that segmentation is stable, reproducible, and has a simple boundary (which is coherent with the use of K-means)

### Feature importance (what separates clusters?)

**Ranking from Random Forest:**
1. `pc_logoff` (logoff count)
2. `pc_logon` (logon count)
3. `num_files_downloaded`
4. `files_per_logon`

**Decision Tree confirms:**
- Root node splits on `pc_logoff`
- Left branch reaches Gini = 0.04 (near-purity in Intensive cluster)

**Interpretation:** Access intensity is the primary discriminative signal, followed by file manipulation activity.

---

## Cluster interpretation

| Cluster | Label | Size | Characteristics |
|---------|-------|------|------------------|
| Cluster 0 | Regular | 83% | Moderate logins, normal file activity, primarily business hours |
| Cluster 1 | Intensive / Atypical | 17% | Higher login volume, more file/device manipulation, night/weekend activity |

**Critical disclaimer:** This cluster does **NOT** automatically indicate malicious intent. System administrators, data analysts, or project managers under deadline pressure may exhibit identical profiles. The model identifies a **behavioral subgroup**, not guilt or intention. Human governance must investigate context.

---

## Main charts

### Correlation structure after feature reduction
![Correlation heatmap](assets/correlation_heatmap.png)

### Silhouette score by number of clusters — StandardScaler
![Silhouette StandardScaler](assets/silhouette_standard.png)

### Silhouette score by number of clusters — RobustScaler
![Silhouette RobustScaler](assets/silhouette_robust.png)

### PCA view of clusters on the training split
![PCA train clusters](assets/pca_train_clusters.png)

### PCA view of predicted clusters on the test split
![PCA test clusters](assets/pca_test_clusters.png)

### Holdout F1 macro for supervised models
![F1 macro models](assets/model_f1_macro.png)

### Top 10 Random Forest feature importances
![Random Forest feature importance](assets/rf_feature_importance.png)

---

## Key findings

### Clustering
- User behaviors are separable into **two main clusters** with StandardScaler
- k=2 has higher stability (lower silhouette standard deviation) than k=3
- Increasing k degrades stability without adding operational value
- Metric trade-offs matter: RobustScaler gives better internal separation but loses sensitivity to rare behavior

### Feature behavior
- Logon-related variables are the strongest behavioral markers
- File-download intensity and device usage also contribute strongly
- Expected correlations among activity-count variables and their derived ratios

### Model validation
- The discovered clusters are highly reproducible (F1 > 0.98 across all models)
- Linear boundary structure: LR outperforms DT, RF and SVM outperforms LR only marginally
- Decision Tree and Random Forest provide key interpretability insights
- High performance on pseudo-labels = internal consistency, NOT real-world validity

### Why the supervised phase is useful
- Provides internal validation (checks if clusters are learnable)
- Confirms segmentation is stable and well-defined
- Identifies most important features driving separation
- Enables efficient assignment of new users without re-running K-Means

### Limitations of supervised phase
- Uses pseudo-labels, not ground truth → only measures internal consistency
- High performance ≠ real-world risk detection
- May not generalize to other datasets or time periods
- Inherits K-Means assumptions (spherical clusters)
- Does NOT replace external validation

---

## Limitations 

1. **No external validation:** Deliberately ignored CERT ground truth labels to replicate real-world scenarios → cannot claim real threat detection
2. **Synthetic dataset:** CERT is partially synthetic and oriented toward insider threat problems
3. **Temporal aggregation:** Lost sequence of events (e.g., activity bursts, ordering)
4. **Scaling sensitivity:** How outliers are treated significantly influences segmentation
5. **K-Means assumptions:** Assumes spherical, equally sized clusters
6. **Pedagogical constraints:** K-Means not optimal for anomaly detection; Isolation Forest or DBSCAN would be more appropriate for that purpose

---

## Future work

1. **External validation:** Test clusters against real CERT labels or real corporate data (beyond this project's scope but valuable)
2. **Alternative algorithms:** Use Isolation Forest or DBSCAN for direct anomaly detection
3. **Richer features:** Add temporal patterns, intensity bursts, sentiment analysis of emails
4. **Continuous retraining:** Implement rolling time windows (monthly) with human-in-the-loop validation
5. **Production implementation:** Deploy classifier (Logistic Regression) for efficient new user assignment

---

## Practical implications

This work does **not** replace human analysts. It provides a **pre-alert system** based on behavioral segmentation for organizations in early-stage monitoring (no historical incident labels).

**Value proposition:**
- Filter noise so signal (whether a dedicated admin or a true insider) receives investigation
- Not every intensive activity is malicious, but almost every malicious activity is rare and identifiable
- The model makes human judgment **more efficient**, not obsolete

**Suggested implementation:**
- Retrain pipeline on organizational data
- Establish human feedback loop: analyst flags → model update
- Continuous retraining on monthly rolling windows

---

## References

1. CERT Division. *Insider Threat Test Dataset*. Software Engineering Institute, Carnegie Mellon University.  
   https://www.sei.cmu.edu/library/insider-threat-test-dataset/

2. Eldardiry, H., et al. (2013). *Multi-domain information fusion for insider threat detection*. IEEE Security and Privacy Workshops.

3. IBM Security. *Cost of a Data Breach Report* (2025).  
   https://www.ibm.com/reports/data-breach

4. IBM. *What are Insider Threats?*  
   https://www.ibm.com/think/topics/insider-threats

5. Joyce, R. J., Raff, E., & Nicholas, C. (2021). *A Framework for Cluster and Classifier Evaluation in the Absence of Reference Labels*. In NeurIPS 2020 Workshop.

6. Legg, P. A., et al. (2017). *Automated insider threat detection using user and role-based profile assessment*. IEEE Security & Privacy.

7. Mourer, A., Forest, F., Lebbah, M., Azzag, H., & Lacaille, J. (2023). *Selecting the Number of Clusters K with a Stability Trade-off: An Internal Validation Criterion*. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), pp. 210-222.

8. Ponemon Institute. *The Security Risk Organizations Should Not Ignore: Careless, Negligent and Malicious Insiders*.  
   https://www.ponemon.org/news-updates/blog/security/the-security-risk-organizations-should-not-ignore-careless-negligent-and-malicious-insiders.html

9. Teng, H.-W., Kang, M.-H., Lee, I.-H., & Bai, L.-C. (2024). *Bridging accuracy and interpretability: A rescaled cluster-then-predict approach for enhanced credit scoring*. International Review of Financial Analysis, 91, 103005.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── Insider++.py
└── assets/
    ├── correlation_heatmap.png
    ├── silhouette_standard.png
    ├── silhouette_robust.png
    ├── pca_train_clusters.png
    ├── pca_test_clusters.png
    ├── model_f1_macro.png
    └── rf_feature_importance.png
```
---

## How to run

```bash
pip ins
tall -r requirements.txt
python Insider++.py
```

The script generates figures from the behavioral datasets and saves them to an output directory.

---

## Notes

- This repository is a presentation version of the work done for Introduction to Data Science class
- There were limitations on model choices and extension 
- Keeps only the visual results that best summarize the analysis.
