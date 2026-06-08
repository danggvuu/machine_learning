# 💼 AI-Powered Job Career Level Classifier & LDA from Scratch

An end-to-end Machine Learning project designed to predict a job's **Career Level** (ranging from specialists to managing directors) based on job titles, descriptions, functions, industries, and locations. 

This repository showcases production-grade ML engineering: **modular codebase structure**, **custom algorithm implementation (LDA from scratch)**, **class imbalance handling (oversampling)**, and **interactive UI deployment**.

---

## 🚀 Interactive Demo & Visualization
The project includes a **Streamlit Web Application** featuring:
- **Interactive Classification:** Input any job details and see prediction probability distributions.
- **2D LDA Projection Space Map:** An interactive Plotly scatter plot visualizing semantic job clusters in a reduced dimension space, mapping the new job input as a glowing marker relative to training clusters.

To launch the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure
```text
Machine_learning/
├── src/
│   ├── __init__.py
│   ├── scratch_lda.py       # Custom LDA Estimator (sklearn compatible)
│   ├── preprocessing.py     # Clean data & ColumnTransformer pipelines
│   └── train.py             # Script to balance data, train & compare models
├── notebooks/
│   └── explore.ipynb        # Clean Jupyter Notebook for EDA & experiments
├── saved_models/
│   ├── random_forest_pipeline.joblib
│   ├── lda_sklearn_pipeline.joblib
│   ├── lda_scratch_pipeline.joblib
│   └── projected_data.csv   # Pre-computed coordinates for the web app map
├── final_project.ods        # The database (8,000+ job listings)
├── requirements.txt         # Package dependencies
└── README.md                # Premium project presentation
```

---

## 🧮 Custom LDA Mathematical Formulation
Linear Discriminant Analysis (implemented from scratch in `src/scratch_lda.py`) projects high-dimensional TF-IDF features onto a lower-dimensional subspace that maximizes class separability.

### 1. Scatter Matrices
- **Within-Class Scatter ($S_W$):** Measures the spread of samples within each class:
  $$S_W = \sum_{c \in C} \sum_{x \in X_c} (x - m_c)(x - m_c)^T$$
  where $m_c$ is the mean vector of class $c$.

- **Between-Class Scatter ($S_B$):** Measures the distance between class mean vectors and the overall mean vector $m$:
  $$S_B = \sum_{c \in C} N_c (m_c - m)(m_c - m)^T$$
  where $N_c$ is the number of samples in class $c$.

### 2. Fisher's Optimization Criterion
LDA solves for projection vectors $W$ that maximize Fisher's criterion:
$$J(W) = \frac{W^T S_B W}{W^T S_W W}$$

This reduces to a generalized eigenvalue problem:
$$S_W^{-1} S_B w = \lambda w$$

The projection matrix $W$ is composed of the eigenvectors corresponding to the top eigenvalues of $S_W^{-1} S_B$. Since $S_W$ can be singular for sparse text matrices, a **Moore-Penrose pseudo-inverse** is utilized:
$$A = \text{pinv}(S_W) \cdot S_B$$

---

## 📊 Model Comparison & Insights

The dataset is highly imbalanced (e.g., `senior_specialist` has 4,300+ samples, while `managing_director` has 4). We resolved this using `RandomOverSampler` on the training split to balance the learning objective.

| Model Pipeline | Feature Selector | Overall Accuracy | Sensitivity to Minority Classes |
| :--- | :--- | :--- | :--- |
| **Random Forest** | None (Dense Text) | **74.18%** | 🔴 Low (0% Recall on rare classes) |
| **Sklearn LDA** | SelectKBest (300) | **67.43%** | 🟢 High (67% Recall on `specialist`) |
| **Scratch LDA (Ours)** | SelectKBest (300) | **67.31%** | 🟢 High (67% Recall on `specialist`) |

### Key ML Takeaways:
1. **Random Forest vs. LDA:** Random Forest obtains higher overall accuracy but completely misses minority classes because it learns dominant class boundaries. LDA computes class centroids geometrically. Once classes are oversampled, LDA projects test samples to their geometrically closest centroid, boosting recall on rare categories (e.g., `specialist` recall jumps to **67%**).
2. **Scratch LDA Validation:** Our custom class-centroid model performs almost identically to Scikit-learn's LDA implementation, validating the correctness of our mathematical calculations.

---

## ⚙️ Installation & Usage

1. **Clone and Navigate:**
   ```bash
   git clone <your-repo-link>
   cd Machine_learning
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models & Serialize Pipelines:**
   ```bash
   PYTHONPATH=. python3 src/train.py
   ```
   This will train the models, output evaluation matrices, and save joblib pipelines to the `saved_models/` folder.

4. **Run Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
