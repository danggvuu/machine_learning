# 💼 AI-Powered Job Career Level Classifier & LDA from Scratch

Vietnam / English Bilingual Presentation:
- [English Version](#english-version)
- [Bản tiếng Việt](#bản-tiếng-việt)

---

# English Version

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

4. **Run Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```

---
---

# Bản Tiếng Việt

Dự án Machine Learning toàn diện (End-to-End) được thiết kế để dự đoán **Cấp độ nghề nghiệp (Career Level)** (từ chuyên viên đến giám đốc điều hành) dựa trên các thông tin: tiêu đề công việc, mô tả công việc, lĩnh vực chuyên môn, ngành nghề và địa điểm làm việc.

Dự án này thể hiện năng lực kỹ thuật chuẩn công nghiệp (production-grade): **Cấu trúc mã nguồn dạng module**, **tự cài đặt thuật toán từ công thức toán (LDA từ scratch)**, **xử lý mất cân bằng dữ liệu (oversampling)**, và **triển khai ứng dụng Web Demo (Streamlit)**.

---

## 🚀 Demo Tương tác & Trực quan hóa
Dự án bao gồm một **Ứng dụng Web Streamlit** có các tính năng:
- **Phân loại tương tác:** Gõ thông tin công việc bất kỳ để nhận kết quả dự đoán cùng phân phối xác suất.
- **Bản đồ không gian LDA 2D:** Biểu đồ phân tán Plotly trực quan hóa các cụm công việc trong không gian giảm chiều, định vị công việc mới nhập vào dưới dạng một điểm sáng nổi bật so với các cụm dữ liệu huấn luyện gốc.

Để chạy ứng dụng:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Cấu trúc dự án
```text
Machine_learning/
├── src/
│   ├── __init__.py
│   ├── scratch_lda.py       # Thuật toán LDA tự code (tương thích sklearn)
│   ├── preprocessing.py     # Làm sạch dữ liệu & xây dựng pipeline tiền xử lý
│   └── train.py             # Script cân bằng dữ liệu, huấn luyện & so sánh mô hình
├── notebooks/
│   └── explore.ipynb        # Jupyter Notebook phân tích dữ liệu (EDA) và thử nghiệm
├── saved_models/
│   ├── random_forest_pipeline.joblib
│   ├── lda_sklearn_pipeline.joblib
│   ├── lda_scratch_pipeline.joblib
│   └── projected_data.csv   # Tọa độ không gian 2D đã được tính trước cho web app
├── final_project.ods        # Cơ sở dữ liệu gốc (hơn 8.000 tin tuyển dụng)
├── requirements.txt         # Các thư viện phụ thuộc của dự án
└── README.md                # Tài liệu hướng dẫn & giới thiệu
```

---

## 🧮 Công thức Toán học của LDA tự xây dựng
Thuật toán phân tích biệt thức tuyến tính (Linear Discriminant Analysis - cài đặt tại `src/scratch_lda.py`) thực hiện chiếu các vector đặc trưng TF-IDF chiều cao xuống không gian con có số chiều thấp hơn sao cho khoảng cách giữa các lớp là lớn nhất.

### 1. Ma trận hiệp phương sai / Phân tán (Scatter Matrices)
- **Phân tán trong nội bộ nhóm ($S_W$):** Đo độ lệch/phân tán của các mẫu trong cùng một lớp:
  $$S_W = \sum_{c \in C} \sum_{x \in X_c} (x - m_c)(x - m_c)^T$$
  với $m_c$ là vector trung bình của lớp $c$.

- **Phân tán giữa các nhóm ($S_B$):** Đo khoảng cách giữa vector trung bình của các lớp và vector trung bình tổng thể $m$:
  $$S_B = \sum_{c \in C} N_c (m_c - m)(m_c - m)^T$$
  với $N_c$ là số lượng mẫu trong lớp $c$.

### 2. Tiêu chuẩn tối ưu hóa Fisher (Fisher's Criterion)
LDA tìm kiếm các vector chiếu $W$ nhằm tối đa hóa tiêu chuẩn Fisher:
$$J(W) = \frac{W^T S_B W}{W^T S_W W}$$

Bài toán này được quy về việc tìm các trị riêng (eigenvalues) và vector riêng (eigenvectors):
$$S_W^{-1} S_B w = \lambda w$$

Ma trận chiếu $W$ được cấu thành từ các vector riêng tương ứng với các trị riêng lớn nhất của $S_W^{-1} S_B$. Vì $S_W$ có thể bị suy biến (singular) khi xử lý ma trận văn bản thưa, chúng tôi sử dụng **giả nghịch đảo Moore-Penrose (pseudo-inverse)** để xử lý:
$$A = \text{pinv}(S_W) \cdot S_B$$

---

## 📊 So sánh Mô hình & Nhận xét

Tập dữ liệu gốc bị mất cân bằng rất nghiêm trọng (ví dụ: cấp độ `senior_specialist` có hơn 4.300 mẫu, trong khi `managing_director` chỉ có 4 mẫu). Chúng tôi giải quyết bằng cách áp dụng `RandomOverSampler` trên tập huấn luyện (training split) để cân bằng mục tiêu tối ưu.

| Pipeline Mô hình | Chọn đặc trưng | Độ chính xác (Accuracy) | Độ nhạy với lớp thiểu số (Minority Class) |
| :--- | :--- | :--- | :--- |
| **Random Forest** | Không có (Văn bản đầy đủ) | **74.18%** | 🔴 Thấp (0% Recall ở các lớp rất hiếm) |
| **Sklearn LDA** | SelectKBest (300) | **67.43%** | 🟢 Cao (67% Recall cho lớp `specialist`) |
| **Scratch LDA (Tự code)** | SelectKBest (300) | **67.31%** | 🟢 Cao (67% Recall cho lớp `specialist`) |

### Nhận xét quan trọng về Machine Learning:
1. **Random Forest vs. LDA:** Random Forest đạt độ chính xác tổng thể cao hơn nhưng bỏ qua hoàn toàn các lớp thiểu số do mô hình tối ưu theo ranh giới nhãn đa số. Ngược lại, LDA tính toán các trọng tâm lớp theo hình học. Khi áp dụng oversampling, LDA ánh xạ mẫu thử nghiệm đến trọng tâm hình học gần nhất, giúp tăng vọt độ nhạy (Recall) đối với các nhóm cực hiếm (như lớp `specialist` tăng lên **67%**).
2. **Kiểm chứng Scratch LDA:** Mô hình dựa trên trọng tâm không gian tự viết của chúng tôi cho kết quả gần như trùng khớp hoàn toàn với thư viện Scikit-learn, chứng minh độ chính xác của các công thức toán học tự cài đặt.

---

## ⚙️ Cài đặt & Sử dụng

1. **Tải dự án về:**
   ```bash
   git clone <link-repo-cua-ban>
   cd Machine_learning
   ```

2. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Huấn luyện mô hình & Lưu trữ pipeline:**
   ```bash
   PYTHONPATH=. python3 src/train.py
   ```

4. **Chạy ứng dụng Web Demo (Streamlit):**
   ```bash
   streamlit run app.py
   ```
