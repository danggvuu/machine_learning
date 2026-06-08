import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

from src.preprocessing import preprocess_data, get_preprocessor_pipeline
from src.scratch_lda import ScratchLDA

def train_and_evaluate():
    # 1. Paths setup
    data_path = "final_project.ods"
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data from:", data_path)
    # Read sheet using pandas read_excel with odf engine
    df_raw = pd.read_excel(data_path, dtype=str)
    
    # 2. Preprocessing
    print("Preprocessing dataset...")
    df = preprocess_data(df_raw, is_training=True)
    
    target = "career_level"
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"Dataset shape: {df.shape}")
    print("Class distribution before oversampling:\n", y.value_counts())
    
    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100, stratify=y
    )
    
    # 4. Feature Extraction & Transform (Pre-balancing)
    print("Fitting preprocessor pipeline on training data...")
    preprocessor = get_preprocessor_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Convert sparse matrix to dense array for models that need it (like LDA implementations)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
        
    # 5. Handle Class Imbalance via Oversampling on Training Data
    print("Applying RandomOverSampler to address class imbalance...")
    oversampler = RandomOverSampler(random_state=100)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(
        X_train_transformed, y_train
    )
    print("Resampled Class distribution:\n", pd.Series(y_train_resampled).value_counts())
    
    # 6. Model Training & Comparison
    results = {}
    
    # --- MODEL 1: Random Forest ---
    print("\n--- Training Model 1: Random Forest Classifier ---")
    rf_clf = RandomForestClassifier(
        random_state=100,
        n_estimators=100,
        criterion="gini",
        n_jobs=-1
    )
    rf_clf.fit(X_train_resampled, y_train_resampled)
    rf_preds = rf_clf.predict(X_test_transformed)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_preds))
    
    # Save RF Pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_clf)
    ])
    joblib.dump(rf_pipeline, os.path.join(models_dir, "random_forest_pipeline.joblib"))
    results["Random Forest"] = rf_acc
    
    # --- MODEL 2: Sklearn LDA (with SelectKBest) ---
    print("\n--- Training Model 2: Sklearn Linear Discriminant Analysis ---")
    # Using SelectKBest to pick the top 300 features first
    selector_sklearn = SelectKBest(score_func=f_classif, k=300)
    X_train_sel_sklearn = selector_sklearn.fit_transform(X_train_resampled, y_train_resampled)
    X_test_sel_sklearn = selector_sklearn.transform(X_test_transformed)
    
    lda_sklearn = LinearDiscriminantAnalysis(solver="svd")
    lda_sklearn.fit(X_train_sel_sklearn, y_train_resampled)
    lda_sk_preds = lda_sklearn.predict(X_test_sel_sklearn)
    lda_sk_acc = accuracy_score(y_test, lda_sk_preds)
    print(f"Sklearn LDA Accuracy: {lda_sk_acc:.4f}")
    print(classification_report(y_test, lda_sk_preds))
    
    # Save Sklearn LDA Pipeline
    lda_sk_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("select", selector_sklearn),
        ("classifier", lda_sklearn)
    ])
    joblib.dump(lda_sk_pipeline, os.path.join(models_dir, "lda_sklearn_pipeline.joblib"))
    results["Sklearn LDA"] = lda_sk_acc
    
    # --- MODEL 3: Scratch LDA (with SelectKBest) ---
    print("\n--- Training Model 3: Custom Scratch LDA Classifier ---")
    # Using SelectKBest to pick the top 300 features first
    selector_scratch = SelectKBest(score_func=f_classif, k=300)
    X_train_sel_scratch = selector_scratch.fit_transform(X_train_resampled, y_train_resampled)
    X_test_sel_scratch = selector_scratch.transform(X_test_transformed)
    
    lda_scratch = ScratchLDA(n_components=min(5, len(np.unique(y_train_resampled)) - 1))
    lda_scratch.fit(X_train_sel_scratch, y_train_resampled)
    lda_scratch_preds = lda_scratch.predict(X_test_sel_scratch)
    lda_scratch_acc = accuracy_score(y_test, lda_scratch_preds)
    print(f"Scratch LDA Accuracy: {lda_scratch_acc:.4f}")
    print(classification_report(y_test, lda_scratch_preds))
    
    # Save Scratch LDA Pipeline
    lda_scratch_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("select", selector_scratch),
        ("classifier", lda_scratch)
    ])
    joblib.dump(lda_scratch_pipeline, os.path.join(models_dir, "lda_scratch_pipeline.joblib"))
    results["Scratch LDA"] = lda_scratch_acc
    
    # 7. Print Final Comparison Table
    print("\n=============================================")
    print("           MODEL COMPARISON SUMMARY          ")
    print("=============================================")
    for model_name, acc in results.items():
        print(f"{model_name:<25}: Accuracy = {acc:.4f}")
    print("=============================================")
    print("All models and pipelines serialized to:", models_dir)

if __name__ == "__main__":
    train_and_evaluate()
