import streamlit as st
import joblib
import numpy as np

# Load models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
w2v_model = joblib.load("word2vec_model.pkl")

# Function to convert question to average word2vec vector
def question_to_vec(question, model, vector_size=300):
    words = question.lower().strip().split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vecs, axis=0)

# Streamlit UI
st.title("🤖 Quora Duplicate Question Detector")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

model_choice = st.radio("Choose a model:", ["Random Forest", "XGBoost"])

if st.button("Check Duplicate"):
    if q1 and q2:
        q1_vec = question_to_vec(q1, w2v_model).reshape(1, -1)
        q2_vec = question_to_vec(q2, w2v_model).reshape(1, -1)
        X = np.hstack((q1_vec, q2_vec, np.abs(q1_vec - q2_vec)))

        if model_choice == "Random Forest":
            pred = rf_model.predict(X)[0]
        else:
            pred = xgb_model.predict(X)[0]

        if pred == 1:
            st.success("✅ These questions are duplicates!")
        else:
            st.warning("❌ These questions are not duplicates.")
    else:
        st.info("Please enter both questions to proceed.")