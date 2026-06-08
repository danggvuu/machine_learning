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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    # Model Selector
    model_choice = st.selectbox(
        "Select Prediction Model",
        options=["Scratch LDA (Custom)", "Sklearn LDA", "Random Forest"],
        index=0,
        help="Choose which machine learning algorithm to run for classification."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Details")
    if model_choice == "Scratch LDA (Custom)":
        st.markdown("""
        **Custom Linear Discriminant Analysis**
        - **Accuracy:** `67.3%`
        - **Pros:** Implemented completely from scratch using Eigen-decomposition. Robust on minority classes after balancing.
        - **Math Concept:** Projects high-dimensional TF-IDF vectors onto a subspace that maximizes class separation.
        """)
    elif model_choice == "Sklearn LDA":
        st.markdown("""
        **Sklearn LDA (SVD Solver)**
        - **Accuracy:** `67.4%`
        - **Pros:** Highly optimal dimensions separation. Fast prediction times.
        - **Note:** Standard scikit-learn SVD implementation with top 300 features (SelectKBest).
        """)
    else:
        st.markdown("""
        **Random Forest Classifier**
        - **Accuracy:** `74.2%`
        - **Pros:** Highest overall accuracy. Excellent on majority classes.
        - **Cons:** Less sensitive to rare/minority classes compared to geometric LDA.
        """)
        
    st.markdown("---")
    st.markdown("### 🎓 Developer CV Notes")
    st.markdown("""
    This project showcases:
    1. **Modular Code Structure** (`src/` architecture).
    2. **Custom Algorithm Implementation** (LDA built from scratch).
    3. **Imbalanced Data Handling** (using RandomOverSampler).
    4. **Web App Deployment** (Streamlit & Plotly integrations).
    """)

# --- MAIN PAGE HEADER ---
st.title("💼 AI Job Career Level Classifier")
st.markdown("""
    Enter a job title, description, location, and industry, and the machine learning model will predict the **Career Level** 
    associated with it. The model is trained on a rich database of 8,000+ job listings.
""")

# Layout: 2 Columns (Inputs in Col 1, Prediction & Visuals in Col 2)
col1, col2 = st.columns([1.1, 1.2], gap="large")

# --- COLUMN 1: INPUTS ---
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📝 Job Details")
    
    # Job Title Input
    job_title = st.text_input(
        "Job Title", 
        value="Lead Technical Architect - Cloud Platform",
        placeholder="e.g. Senior Backend Engineer"
    )
    
    # Location
    location = st.text_input(
        "Location",
        value="Seattle, WA",
        placeholder="e.g. Seattle, WA or London, UK"
    )
    
    # Function Dropdown
    functions_list = [
        "information_technology_telecommunications", "sales", "production_manufacturing",
        "procurement_materials_logistics", "consulting", "engineering", "finance",
        "marketing_public_relations", "human_resources", "administration", "UNKNOWN"
    ]
    function = st.selectbox("Job Function", options=functions_list, index=0)
    
    # Industry
    industry = st.text_input(
        "Industry",
        value="Software Companies",
        placeholder="e.g. Information Technology"
    )
    
    # Description
    description = st.text_area(
        "Job Description",
        value="Responsible for leading a team of system engineers. Design, implement, and maintain midrange server infrastructure. Provide technical guidance, architecture patterns, and mentor junior developers. 5+ years of experience required.",
        height=180
    )
    
    predict_button = st.button("Predict Career Level")
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
        
        # Human-friendly class name mapping
        friendly_label = pred_label.replace("_", " ").title()
        
        # Get Probabilities
        probas = active_pipe.predict_proba(input_data)[0]
        classes = active_pipe.classes_
        
        # Card 1: Prediction results
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Result")
        st.markdown(f"Predicted Career Level:")
        st.markdown(f'<div class="prediction-badge">{friendly_label}</div>', unsafe_allow_html=True)
        
        # Plotly Horizontal Bar Chart for Probabilities
        prob_df = pd.DataFrame({
            "Career Level": [c.replace("_", " ").title() for c in classes],
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
            title="Prediction Probability Distribution"
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
            st.subheader("🗺️ 2D Linear Discriminant Analysis Map")
            st.markdown("See where your input job falls in the semantic clusters of career levels:")
            
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
            
            # Create Plotly Scatter
            fig_scatter = px.scatter(
                projected_df,
                x="LD1",
                y="LD2",
                color="career_level",
                hover_data=["title"],
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Linear Discriminant Subspace (LD1 vs LD2)"
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
                    name="Your Job Input",
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
        st.markdown("### 👈 Awaiting Input")
        st.markdown("Fill in the job details on the left and click **Predict Career Level** to see the machine learning model classification and spatial projections.")
        st.markdown('</div>', unsafe_allow_html=True)
