import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="AI Job Career Level Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translation Dictionary
TRANSLATIONS = {
    "en": {
        "page_title": "AI Job Career Level Classifier",
        "subtitle": "Enter a job title, description, location, and industry, and the machine learning model will predict the Career Level associated with it. The model is trained on a database of 8,000+ job listings.",
        "config": "🛠️ Configuration",
        "lang_label": "Select Language / Chọn ngôn ngữ",
        "select_model": "Select Prediction Model",
        "model_details": "📊 Model Details",
        "cv_notes": "🎓 Developer CV Notes",
        "cv_points": [
            "Modular Code Structure (`src/` architecture).",
            "Custom Algorithm Implementation (LDA built from scratch).",
            "Imbalanced Data Handling (using RandomOverSampler).",
            "Web App Deployment (Streamlit & Plotly integrations)."
        ],
        "job_details": "📝 Job Details",
        "job_title_label": "Job Title",
        "location_label": "Location",
        "job_function_label": "Job Function",
        "industry_label": "Industry",
        "job_desc_label": "Job Description",
        "predict_btn": "Predict Career Level",
        "pred_result": "🎯 Prediction Result",
        "predicted_level": "Predicted Career Level:",
        "prob_dist": "Prediction Probability Distribution",
        "lda_map": "🗺️ 2D Linear Discriminant Analysis Map",
        "lda_map_desc": "See where your input job falls in the semantic clusters of career levels:",
        "lda_title": "Linear Discriminant Subspace (LD1 vs LD2)",
        "your_job": "Your Job Input",
        "awaiting_input_title": "👈 Awaiting Input",
        "awaiting_input_desc": "Fill in the job details on the left and click **Predict Career Level** to see the machine learning model classification and spatial projections.",
        "model_desc_scratch": "**Custom Linear Discriminant Analysis**\n- **Accuracy:** `67.3%`\n- **Pros:** Implemented completely from scratch using Eigen-decomposition. Robust on minority classes after balancing.\n- **Math Concept:** Projects high-dimensional TF-IDF vectors onto a subspace that maximizes class separation.",
        "model_desc_sklearn": "**Sklearn LDA (SVD Solver)**\n- **Accuracy:** `67.4%`\n- **Pros:** Highly optimal dimensions separation. Fast prediction times.\n- **Note:** Standard scikit-learn SVD implementation with top 300 features (SelectKBest).",
        "model_desc_rf": "**Random Forest Classifier**\n- **Accuracy:** `74.2%`\n- **Pros:** Highest overall accuracy. Excellent on majority classes.\n- **Cons:** Less sensitive to rare/minority classes compared to geometric LDA."
    },
    "vi": {
        "page_title": "Bộ phân loại cấp độ nghề nghiệp bằng AI",
        "subtitle": "Nhập tiêu đề công việc, mô tả công việc, địa điểm và ngành nghề để mô hình học máy dự đoán Cấp độ nghề nghiệp tương ứng. Mô hình được huấn luyện trên cơ sở dữ liệu hơn 8,000 tin tuyển dụng.",
        "config": "🛠️ Cấu hình hệ thống",
        "lang_label": "Select Language / Chọn ngôn ngữ",
        "select_model": "Chọn mô hình dự đoán",
        "model_details": "📊 Chi tiết thuật toán",
        "cv_notes": "🎓 Điểm nhấn CV lập trình viên",
        "cv_points": [
            "Cấu trúc mã nguồn dạng module (kiến trúc thư mục `src/`).",
            "Tự triển khai thuật toán từ đầu (LDA viết từ scratch).",
            "Xử lý mất cân bằng dữ liệu (sử dụng RandomOverSampler).",
            "Triển khai ứng dụng Web (tích hợp Streamlit & Plotly)."
        ],
        "job_details": "📝 Thông tin công việc",
        "job_title_label": "Tiêu đề công việc",
        "location_label": "Địa điểm",
        "job_function_label": "Lĩnh vực chuyên môn",
        "industry_label": "Ngành nghề",
        "job_desc_label": "Mô tả công việc",
        "predict_btn": "Dự đoán cấp độ nghề nghiệp",
        "pred_result": "🎯 Kết quả dự đoán",
        "predicted_level": "Cấp độ nghề nghiệp dự đoán:",
        "prob_dist": "Phân phối xác suất dự đoán",
        "lda_map": "🗺️ Bản đồ phân tích biệt thức tuyến tính 2D (LDA)",
        "lda_map_desc": "Xem công việc của bạn nằm ở đâu trong các cụm phân loại hình học của cấp độ nghề nghiệp:",
        "lda_title": "Không gian con LDA (Trục LD1 so với LD2)",
        "your_job": "Công việc của bạn",
        "awaiting_input_title": "👈 Đang chờ thông tin",
        "awaiting_input_desc": "Điền thông tin công việc ở cột bên trái và nhấn nút **Dự đoán cấp độ nghề nghiệp** để xem kết quả phân loại và biểu đồ không gian của mô hình.",
        "model_desc_scratch": "**LDA tự xây dựng (Scratch)**\n- **Độ chính xác:** `67.3%`\n- **Ưu điểm:** Tự viết thuật toán 100% bằng cách phân rã trị riêng. Cực kỳ nhạy bén với lớp thiểu số sau khi oversampling.\n- **Nguyên lý:** Chiếu các vector TF-IDF chiều cao xuống không gian con tối đa hóa sự phân tách giữa các lớp.",
        "model_desc_sklearn": "**Sklearn LDA (SVD)**\n- **Độ chính xác:** `67.4%`\n- **Ưu điểm:** Phân tách không gian rất tối ưu. Tốc độ dự đoán cực nhanh.\n- **Chi tiết:** Sử dụng bộ giải SVD mặc định của thư viện scikit-learn với 300 đặc trưng tốt nhất (SelectKBest).",
        "model_desc_rf": "**Random Forest Classifier**\n- **Độ chính xác:** `74.2%`\n- **Ưu điểm:** Đạt độ chính xác tổng thể cao nhất. Phân loại rất tốt trên các nhãn đa số.\n- **Nhược điểm:** Kém nhạy bén hơn với các nhãn thiểu số (nhóm hiếm)."
    }
}

CAREER_LEVEL_MAPPING = {
    "en": {
        "senior_specialist_or_project_manager": "Senior Specialist / Project Manager",
        "manager_team_leader": "Manager / Team Leader",
        "bereichsleiter": "Department Head (Bereichsleiter)",
        "director_business_unit_leader": "Director / Business Unit Leader",
        "specialist": "Specialist",
        "managing_director_small_medium_company": "Managing Director (SME)"
    },
    "vi": {
        "senior_specialist_or_project_manager": "Chuyên viên cao cấp / Quản lý dự án",
        "manager_team_leader": "Trưởng nhóm / Quản lý",
        "bereichsleiter": "Trưởng bộ phận (Bereichsleiter)",
        "director_business_unit_leader": "Giám đốc bộ phận",
        "specialist": "Chuyên viên",
        "managing_director_small_medium_company": "Giám đốc điều hành (Công ty vừa và nhỏ)"
    }
}

# Custom Styling (Glassmorphism + Sleek Dark Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Global Background */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Card design */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #66fcf1 !important;
        font-weight: 700 !important;
    }
    
    /* Interactive Predict button */
    .stButton>button {
        background: linear-gradient(135deg, #45f3ff 0%, #1f4068 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(69, 243, 255, 0.2) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(69, 243, 255, 0.4) !important;
    }
    
    /* Badges */
    .prediction-badge {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(69, 243, 255, 0.1);
        border: 1px solid #66fcf1;
        color: #66fcf1;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #45f3ff;
        font-size: 1.5rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to clean location (same as training pipeline)
def clean_location_input(location):
    if not location or not isinstance(location, str):
        return "UNKNOWN"
    loc_str = location.strip()
    if len(loc_str) >= 3 and loc_str[-2:].isupper() and loc_str[-3] == " ":
        return loc_str[-2:]
    return loc_str

# Load Models & Preprocessors
@st.cache_resource
def load_pipelines():
    rf_pipeline = joblib.load("saved_models/random_forest_pipeline.joblib")
    lda_sk_pipeline = joblib.load("saved_models/lda_sklearn_pipeline.joblib")
    lda_scratch_pipeline = joblib.load("saved_models/lda_scratch_pipeline.joblib")
    return rf_pipeline, lda_sk_pipeline, lda_scratch_pipeline

@st.cache_data
def load_projected_data():
    if os.path.exists("saved_models/projected_data.csv"):
        return pd.read_csv("saved_models/projected_data.csv")
    return None

try:
    rf_pipe, lda_sk_pipe, lda_scratch_pipe = load_pipelines()
    projected_df = load_projected_data()
except Exception as e:
    st.error(f"Error loading models/data: {e}. Please run the training script first.")
    st.stop()

# --- SIDEBAR & LANGUAGE SWITCHER ---
with st.sidebar:
    # Language Toggle
    lang_choice = st.radio(
        "🌐 Language / Ngôn ngữ",
        options=["English", "Tiếng Việt"],
        index=0,
        horizontal=True
    )
    
    # Map to "en" or "vi"
    lang_code = "en" if lang_choice == "English" else "vi"
    t = TRANSLATIONS[lang_code]
    
    st.markdown(f"### {t['config']}")
    
    # Model Selector
    model_choice = st.selectbox(
        t["select_model"],
        options=["Scratch LDA (Custom)", "Sklearn LDA", "Random Forest"],
        index=0
    )
    
    st.markdown("---")
    st.markdown(f"### {t['model_details']}")
    if model_choice == "Scratch LDA (Custom)":
        st.markdown(t["model_desc_scratch"])
    elif model_choice == "Sklearn LDA":
        st.markdown(t["model_desc_sklearn"])
    else:
        st.markdown(t["model_desc_rf"])

# --- MAIN PAGE HEADER ---
st.title(t["page_title"])
st.markdown(t["subtitle"])

# Layout: 2 Columns (Inputs in Col 1, Prediction & Visuals in Col 2)
col1, col2 = st.columns([1.1, 1.2], gap="large")

# --- COLUMN 1: INPUTS ---
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(t["job_details"])
    
    # Job Title Input
    job_title = st.text_input(
        t["job_title_label"], 
        value="Lead Technical Architect - Cloud Platform",
        placeholder="e.g. Senior Backend Engineer"
    )
    
    # Location
    location = st.text_input(
        t["location_label"],
        value="Seattle, WA",
        placeholder="e.g. Seattle, WA or London, UK"
    )
    
    # Function Dropdown
    functions_list = [
        "information_technology_telecommunications", "sales", "production_manufacturing",
        "procurement_materials_logistics", "consulting", "engineering", "finance",
        "marketing_public_relations", "human_resources", "administration", "UNKNOWN"
    ]
    function = st.selectbox(t["job_function_label"], options=functions_list, index=0)
    
    # Industry
    industry = st.text_input(
        t["industry_label"],
        value="Software Companies",
        placeholder="e.g. Information Technology"
    )
    
    # Description
    description = st.text_area(
        t["job_desc_label"],
        value="Responsible for leading a team of system engineers. Design, implement, and maintain midrange server infrastructure. Provide technical guidance, architecture patterns, and mentor junior developers. 5+ years of experience required.",
        height=180
    )
    
    predict_button = st.button(t["predict_btn"])
    st.markdown('</div>', unsafe_allow_html=True)

# --- COLUMN 2: RESULTS & VISUALIZATIONS ---
with col2:
    if predict_button:
        # Prepare Input Dataframe
        input_data = pd.DataFrame([{
            "title": job_title,
            "location": location,
            "description": description,
            "function": function,
            "industry": industry
        }])
        
        # Clean location mapping for the preprocessor
        cleaned_loc = clean_location_input(location)
        
        # Select Pipeline
        if model_choice == "Scratch LDA (Custom)":
            active_pipe = lda_scratch_pipe
        elif model_choice == "Sklearn LDA":
            active_pipe = lda_sk_pipe
        else:
            active_pipe = rf_pipe
            
        # Run prediction
        pred_label = active_pipe.predict(input_data)[0]
        
        # Map output to friendly label depending on language choice
        mapped_label = CAREER_LEVEL_MAPPING[lang_code].get(pred_label, pred_label.replace("_", " ").title())
        
        # Get Probabilities
        probas = active_pipe.predict_proba(input_data)[0]
        classes = active_pipe.classes_
        
        # Card 1: Prediction results
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(t["pred_result"])
        st.markdown(t["predicted_level"])
        st.markdown(f'<div class="prediction-badge">{mapped_label}</div>', unsafe_allow_html=True)
        
        # Plotly Horizontal Bar Chart for Probabilities
        prob_df = pd.DataFrame({
            "Career Level": [CAREER_LEVEL_MAPPING[lang_code].get(c, c.replace("_", " ").title()) for c in classes],
            "Probability": probas
        }).sort_values(by="Probability", ascending=True)
        
        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Career Level",
            orientation="h",
            color="Probability",
            color_continuous_scale=["#1f4068", "#45f3ff"],
            text_auto=".1%",
            title=t["prob_dist"]
        )
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c5c6c7",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Card 2: 2D LDA projection map
        if projected_df is not None and model_choice != "Random Forest":
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(t["lda_map"])
            st.markdown(t["lda_map_desc"])
            
            # Step-by-step projection of the new input point
            preprocessor = active_pipe.named_steps["preprocessor"]
            selector = active_pipe.named_steps["select"]
            classifier = active_pipe.named_steps["classifier"]
            
            # Transform input
            input_transformed = preprocessor.transform(input_data)
            if hasattr(input_transformed, "toarray"):
                input_transformed = input_transformed.toarray()
            input_selected = selector.transform(input_transformed)
            input_projected = classifier.transform(input_selected)
            
            new_ld1 = input_projected[0, 0]
            new_ld2 = input_projected[0, 1]
            
            # Map classes for the training dots tooltips
            projected_df_localized = projected_df.copy()
            projected_df_localized["career_level"] = projected_df_localized["career_level"].apply(
                lambda c: CAREER_LEVEL_MAPPING[lang_code].get(c, c.replace("_", " ").title())
            )
            
            # Create Plotly Scatter
            fig_scatter = px.scatter(
                projected_df_localized,
                x="LD1",
                y="LD2",
                color="career_level",
                hover_data=["title"],
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Safe,
                title=t["lda_title"]
            )
            
            # Add user input marker as a pulsing glowing star
            fig_scatter.add_trace(
                go.Scatter(
                    x=[new_ld1],
                    y=[new_ld2],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=20,
                        color="#ff0055",
                        line=dict(width=2, color="#ffffff"),
                        shadow=dict(color="#ff0055", width=10)
                    ),
                    name=t["your_job"],
                    hovertext=[f"Input: {job_title}"]
                )
            )
            
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c5c6c7",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=-0.2),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        # Default Welcome State Card
        st.markdown('<div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 60px 20px;">', unsafe_allow_html=True)
        st.markdown(f"### {t['awaiting_input_title']}")
        st.markdown(t["awaiting_input_desc"])
        st.markdown('</div>', unsafe_allow_html=True)
