import streamlit as st
import pickle

# Load the trained model
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("Spam Email Classification System")
st.write("Enter an email message to check if it's spam or not.")

email_text = st.text_area("Email Message", "")

if st.button("Classify"):
    if email_text:
        email_tfidf = vectorizer.transform([email_text])  # Convert text to numerical format
        prediction = model.predict(email_tfidf)[0]
        
        # Debugging: Show the raw prediction value
        st.write(f"Raw prediction output: {prediction}")

        if prediction == 1:
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is NOT a spam email.")
    else:
        st.warning("Please enter an email message.")
