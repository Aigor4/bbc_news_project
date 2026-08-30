import streamlit as st
from predict import predict

st.title("BBC NEWS CATEGORY PREDICTOR")
news = st.text_area("Enter the news")
button = st.button("Predict the category")
if button:
    if not news:
        st.error("Please enter a news")
    else:
        label = predict(news)
        st.write(f"The category is {label}")