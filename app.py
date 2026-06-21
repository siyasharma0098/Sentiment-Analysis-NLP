
import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🤖 Sentiment Analysis App")

review = st.text_area("Enter your review")

if st.button("Analyze"):

    if review:

        vector = tfidf.transform([review])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        st.success("Sentiment: " + prediction[0])

        st.info(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter a review")
