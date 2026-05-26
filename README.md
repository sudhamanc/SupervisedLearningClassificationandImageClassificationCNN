# Supervised Learning Classification & Image Classification with CNN

## DSCI 631: Applied Machine Learning — Assignment 3

**Author:** Sudhaman Chandrasekaran  
**Course:** Drexel University, College of Computing and Informatics

---

## Overview

This project explores two major machine learning paradigms through real-world datasets:

1. **Part 1 — Unsupervised & Supervised Learning on Tabular Health Data:** Clustering, dimensionality reduction, and classification on the CDC Diabetes Health Indicators dataset.
2. **Part 2 — Image Classification with Convolutional Neural Networks:** Building and training a CNN from scratch to classify vegetable images into 15 categories.

---

## Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| [CDC Diabetes Health Indicators](https://www.kaggle.com/datasets/abdelazizsami/cdc-diabetes-health-indicators) | Kaggle / BRFSS 2015 | 253,680 samples × 22 features | Binary classification (diabetes vs. no diabetes) based on health survey indicators |
| [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) | Kaggle | 21,000 images (15 classes) | 224×224×3 RGB images of 15 vegetable types |

---

## Techniques & Concepts Explored

### Part 1: Tabular Data Analysis

| Technique | Purpose |
|-----------|---------|
| **Exploratory Data Analysis (EDA)** | Understand feature distributions and class separability |
| **PCA (Principal Component Analysis)** | Dimensionality reduction and data visualization |
| **K-Means Clustering** | Partition-based unsupervised grouping |
| **DBSCAN** | Density-based clustering with noise detection |
| **Agglomerative Clustering** | Hierarchical bottom-up clustering |
| **Gaussian Mixture Models** | Probabilistic soft clustering |
| **Normalized Mutual Information (NMI)** | Clustering quality evaluation |
| **Adjusted Rand Index (ARI)** | Clustering agreement with ground truth |
| **Random Forest Classifier** | Ensemble classification with bagging |
| **Gradient Boosting Classifier** | Sequential ensemble with boosting |
| **GridSearchCV** | Hyperparameter tuning with cross-validation |
| **ROC-AUC & Precision** | Classification performance metrics |
| **Class weighting (`balanced`)** | Handling imbalanced datasets |

### Part 2: Deep Learning (CNN)

| Technique | Purpose |
|-----------|---------|
| **Convolutional Neural Network (CNN)** | Hierarchical spatial feature learning |
| **Conv2D layers** | Local feature extraction with learnable filters |
| **MaxPooling** | Spatial dimension reduction and translation invariance |
| **Flatten + Dense layers** | Classification head |
| **Dropout (25%)** | Regularization to prevent overfitting |
| **Adam optimizer** | Adaptive learning rate optimization |
| **Categorical crossentropy** | Multi-class classification loss |

---

## Results

### Part 1: Unsupervised Learning — Clustering

| Method | NMI Score | ARI Score |
|--------|-----------|-----------|
| K-Means (k=2) | 0.0721 | 0.1678 |
| Agglomerative (Ward, k=2) | 0.0413 | 0.1150 |
| Gaussian Mixture (n=2) | 0.0477 | 0.1315 |
| DBSCAN (eps=2.5) | 0.0247 | 0.0255 |

**So What:** All clustering methods scored very low on NMI and ARI, meaning **unsupervised methods fundamentally cannot discover the diabetes classification** from this data. The two classes (diabetic vs. non-diabetic) are deeply interleaved in feature space with no natural cluster boundaries. This demonstrates a key insight: not all classification problems have underlying cluster structure — some require supervised learning with labeled data to build effective decision boundaries.

### Part 1: Supervised Classification

| Model | AUC | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| Random Forest (default) | 0.7965 | 0.4884 | 0.1784 | 0.2613 |
| Random Forest (balanced) | 0.7915 | 0.4646 | 0.1604 | 0.2385 |
| Gradient Boosting | **0.8273** | **0.5568** | 0.1665 | 0.2563 |
| Random Forest (tuned: balanced_subsample, max_depth=10) | 0.8207 | 0.3268 | **0.7209** | **0.4497** |

**So What:** 
- **Gradient Boosting outperforms all others on AUC (0.827)** and precision (0.56), demonstrating that sequential boosting handles complex decision boundaries better than parallel bagging.
- **The tuned RF (with `balanced_subsample` and max_depth=10) dramatically improves recall to 0.72** — catching 72% of actual diabetes cases vs. only 17-18% by default models. This comes at the cost of lower precision (0.33), meaning more false positives.
- **The tuned RF achieves the best F1 score (0.45)**, balancing precision and recall better than any other model.
- **`class_weight='balanced'` alone did not help** — it slightly decreased both AUC and precision without improving recall. The combination of balanced_subsample + depth restriction was needed.
- **Medical interpretation:** The precision-recall tradeoff is critical here. The tuned RF catches 72% of diabetics (high recall) but with many false alarms. For a **screening application**, high recall is preferred — missing a diabetic patient is more costly than an extra blood test. For **diagnostic confirmation**, higher precision models (Gradient Boosting) are preferred to reduce unnecessary interventions.

### Part 2: CNN Image Classification

| Metric | Value |
|--------|-------|
| Training Time | ~33 minutes (10 epochs) |
| Final Train Accuracy | 97.53% |
| Final Validation Accuracy | 91.47% |
| Test Accuracy | 92.10% |
| Macro Precision | 0.9244 |
| Macro Recall | 0.9210 |
| Macro F1-Score | 0.9211 |

**Per-class highlights:** Potato (98% F1), Capsicum (96% F1), Carrot (95% F1) were easiest to classify. Cabbage (86% F1), Cauliflower (88% F1) were hardest — likely due to visual similarity.

**So What:**
- A **simple 2-layer CNN achieves 92% accuracy and 0.92 macro F1** on 15-class vegetable classification, demonstrating that CNNs can learn powerful visual representations even with minimal architecture.
- The **~6% train-val gap** indicates mild overfitting. This could be reduced with data augmentation (random flips, rotations, crops) or stronger regularization.
- The **confusion matrix reveals** that visually similar vegetables (Cabbage/Cauliflower, Brinjal/Tomato) account for most misclassifications — consistent with what humans would find challenging.
- The **bottleneck is the Flatten→Dense connection** (25.6M of 25.7M total parameters). Using Global Average Pooling instead would reduce parameters by ~99% with comparable performance.
- **For production use**, transfer learning from pre-trained models (ResNet, EfficientNet) would achieve >98% accuracy with less training time and better generalization.

---

## PCA Analysis

- The variance is spread relatively evenly across components (no single dominant direction)
- **17 components** needed to capture 90% of variance (out of 21 total features)
- This confirms that the data's complexity cannot be reduced to a few meaningful dimensions — consistent with poor clustering performance

---

## Project Structure

```
├── DSCI 631-assignment3.ipynb   # Main notebook with all solutions
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes venv, data, caches
├── README.md                     # This file
├── data/                         # Downloaded datasets (not tracked in git)
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   └── Vegetable Images/
│       ├── train/
│       ├── validation/
│       └── test/
└── .venv/                        # Python 3.12 virtual environment (not tracked)
```

---

## Setup & Reproduction

```bash
# Create virtual environment (requires Python 3.12 for TensorFlow compatibility)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires Kaggle API credentials)
mkdir -p data
kaggle datasets download -d abdelazizsami/cdc-diabetes-health-indicators -p data/ --unzip
kaggle datasets download -d misrakahmed/vegetable-image-dataset -p data/ --unzip

# Register Jupyter kernel
python -m ipykernel install --user --name=assignment3 --display-name="Assignment3 (Python 3.12)"
```

Then open `DSCI 631-assignment3.ipynb` in VS Code or Jupyter and select the "Assignment3 (Python 3.12)" kernel.

---

## Key Dependencies

- Python 3.12
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn 1.8
- TensorFlow 2.21 / Keras 3.14

---

## License

Academic use only — Drexel University DSCI 631 coursework.
