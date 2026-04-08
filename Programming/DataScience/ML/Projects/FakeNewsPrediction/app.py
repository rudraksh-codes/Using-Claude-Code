import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

st.title("Fake News Detection")
st.write("Enter a news article to check if it's real or fake.")

news_input = st.text_area("News Article", height=200)
if st.button("Check"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        
        if prediction[0] == 1:
            st.success("The news article is likely REAL.")
        else:
            st.error("The news article is likely FAKE.")
    else:
        st.warning("Please enter a text.")