import streamlit as st
import numpy as np
import joblib
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz

# Load Word2Vec model (trained and saved separately)
@st.cache_resource
def load_w2v_model():
    return joblib.load("word2vec_model.pkl")

# Load ML model
@st.cache_resource
def load_model(model_choice):
    if model_choice == 'Random Forest':
        return joblib.load("rf_model_compressed.pkl")
    elif model_choice == 'XGBoost':
        return joblib.load("xgboost_model.pkl")

# Convert question to vector
def question_to_vec(question, model, vector_size=300):
    words = question.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vecs, axis=0)

# Combine features for both questions
def generate_features(q1, q2, w2v_model):
    q1_vec = question_to_vec(q1, w2v_model)
    q2_vec = question_to_vec(q2, w2v_model)
    abs_diff = np.abs(q1_vec - q2_vec)
    features = np.hstack((q1_vec, q2_vec, abs_diff))
    return features.reshape(1, -1)

# Streamlit App UI
st.title("❓ Quora Duplicate Question Detector")
st.markdown("Check if two questions mean the same thing using ML models trained on Quora dataset.")

q1 = st.text_input("Enter Question 1", "")
q2 = st.text_input("Enter Question 2", "")

model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

if st.button("Check if Duplicate"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions.")
    else:
        # Load models
        w2v_model = load_w2v_model()
        model = load_model(model_choice)

        # Generate feature vector
        features = generate_features(q1, q2, w2v_model)

        # Predict
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        # Display results
        if prediction == 1:
            st.success("✅ The questions are **Duplicate**.")
        else:
            st.error("❌ The questions are **Not Duplicate**.")

        if prob is not None:
            st.markdown(f"**Model confidence:** `{prob:.2f}`")