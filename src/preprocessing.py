import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def filter_location(location):
    """
    Cleans location field. If it ends with a US state abbreviation (e.g. 'Houston, TX'),
    it extracts the state code ('TX'). Otherwise, returns the cleaned location.
    """
    if not isinstance(location, str) or pd.isna(location):
        return "UNKNOWN"
    
    loc_str = str(location).strip()
    if len(loc_str) >= 3 and loc_str[-2:].isupper() and loc_str[-3] == " ":
        return loc_str[-2:]
    return loc_str

def preprocess_data(df, is_training=True):
    """
    Preprocess the raw dataframe. 
    Drops rows with missing targets during training, cleans location, 
    and handles missing description text.
    """
    df = df.copy()
    
    # 1. Clean location state abbreviation
    df["location"] = df["location"].apply(filter_location)
    
    # 2. Fill missing text values to prevent TfidfVectorizer errors
    df["description"] = df["description"].fillna("")
    df["title"] = df["title"].fillna("")
    df["industry"] = df["industry"].fillna("")
    df["function"] = df["function"].fillna("UNKNOWN")
    
    if is_training:
        # Drop rows where target variable is null
        df = df.dropna(subset=["career_level"])
        # Remove extremely rare classes or those with NaN
        df = df[df["career_level"].notna()]
        
    return df

def get_preprocessor_pipeline():
    """
    Constructs the standard sklearn ColumnTransformer for the ML pipeline.
    Uses TF-IDF for text features and One-Hot Encoding for categorical features.
    """
    # Define text features and their TF-IDF vectorizers
    # We restrict max_features to keep the dimensional footprint low and prevent overfitting
    title_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=0.01,
        max_df=0.95,
        stop_words="english",
        max_features=1000
    )
    
    desc_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=0.01,
        max_df=0.95,
        stop_words="english",
        max_features=2000
    )
    
    ind_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=0.01,
        max_df=0.95,
        stop_words="english",
        max_features=100
    )
    
    # OneHotEncoder for categoricals. Handle_unknown='ignore' is crucial for web app inference.
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    # Bundle everything in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("title", title_vectorizer, "title"),
            ("desc", desc_vectorizer, "description"),
            ("industry", ind_vectorizer, "industry"),
            ("location", cat_encoder, ["location"]),
            ("function", cat_encoder, ["function"])
        ],
        remainder="drop" # Drop columns not specified
    )
    
    return preprocessor
