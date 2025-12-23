import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config - make it appealing
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .prediction {
        font-size: 2.5rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f5e9;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model (make sure the file is in the same directory)
@st.cache_resource
def load_model():
    return joblib.load('best_house_price_model.pkl')

model = load_model()

# Load original data to get feature names and categories (for input widgets)
# We need train.csv for column info and possible categorical values
train = pd.read_csv('train.csv')

# Drop Id and SalePrice
train = train.drop(['Id', 'SalePrice'], axis=1)

# Identify numerical and categorical features
num_feats = train.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = train.select_dtypes(include=['object']).columns.tolist()

# Get the top features used in the model (you need to define them based on your training)
# IMPORTANT: Replace this list with the exact top_features used in training
# From your previous code, it was the top 60 correlated features after preprocessing
# To make this work perfectly, save the feature list during training and load here.
# For now, we'll assume you have them or approximate by using all after dummy
# Better: During training, after selecting top_features, save them:
# joblib.dump(top_features, 'top_features.pkl')
# Here:
@st.cache_resource
def load_features():
    return joblib.load('top_features.pkl')  # Save this in your training script!

top_features = load_features()

st.title("🏠 Advanced House Price Prediction App")
st.markdown("### Powered by Machine Learning (Random Forest Regressor)")
st.markdown("Enter the details of the house below to get an instant price prediction!")

with st.sidebar:
    st.header("📊 Feature Input")
    st.info("Fill in the details. Missing categorical values can be 'None'.")

# Input form
with st.form("house_form"):
    cols = st.columns(2)
    inputs = {}
    
    for i, feat in enumerate(num_feats + cat_feats):
        col = cols[i % 2]
        with col:
            if feat in num_feats:
                # Reasonable defaults and ranges (adjust based on data_description.txt)
                min_val = float(train[feat].min())
                max_val = float(train[feat].max())
                default = float(train[feat].median())
                inputs[feat] = st.number_input(feat, min_value=min_val, max_value=max_val, value=default, step=1.0)
            else:
                # Categorical: unique values + 'None'
                options = ['None'] + sorted(train[feat].dropna().unique().tolist())
                default = 'None' if train[feat].isna().any() else options[1]
                inputs[feat] = st.selectbox(feat, options=options, index=options.index(default) if default in options else 0)
    
    submitted = st.form_submit_button("🔮 Predict House Price")

if submitted:
    with st.spinner("Processing features and predicting..."):
        # Create DataFrame from inputs
        input_df = pd.DataFrame([inputs])
        
        # Handle missing (though we filled in form)
        input_df[num_feats] = input_df[num_feats].fillna(input_df[num_feats].median())
        input_df[cat_feats] = input_df[cat_feats].fillna('None')
        
        # Skewed features transformation (same as training)
        # You need to save the skewed_feats and the PowerTransformer
        # In training script: joblib.dump(skewed_feats, 'skewed_feats.pkl')
        # joblib.dump(pt, 'power_transformer.pkl')
        try:
            skewed_feats = joblib.load('skewed_feats.pkl')
            pt = joblib.load('power_transformer.pkl')
            input_df[skewed_feats] = pt.transform(input_df[skewed_feats])
        except:
            st.warning("Skew transformer not loaded - skipping (add saving in training if needed)")
        
        # Standardization (same scaler)
        try:
            scaler = joblib.load('scaler.pkl')  # Save in training: joblib.dump(scaler, 'scaler.pkl')
            input_df[num_feats] = scaler.transform(input_df[num_feats])
        except:
            st.warning("Scaler not loaded - skipping")
        
        # One-hot encoding
        input_encoded = pd.get_dummies(input_df)
        
        # Align to training features (add missing columns with 0)
        for col in top_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[top_features]  # Exact order and selection
        
        # Prediction (model expects log target, so exp back)
        pred_log = model.predict(input_encoded)[0]
        pred_price = np.exp(pred_log)
        
        # Display result
        st.markdown(f"<div class='prediction'>${pred_price:,.0f}</div>", unsafe_allow_html=True)
        st.success("Prediction complete!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model trained on Ames Housing dataset")