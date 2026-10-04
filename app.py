import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download("stopwords", quiet=True)

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words("english"))


# Text preprocessing function
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove numbers, punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Page configuration
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 Spam Mail Detector")

st.write(
    "Enter a message below and the machine learning model "
    "will predict whether it is Spam or Ham."
)

# Message input
message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)

# Prediction button
if st.button("🔍 Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Clean message
        cleaned_message = clean_text(message)

        # Convert text into TF-IDF features
        features = vectorizer.transform([cleaned_message])

        # Make prediction
        prediction = model.predict(features)[0]

        # Get probabilities
        probability = model.predict_proba(features)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")

            st.write(
                f"Spam Probability: **{probability[1] * 100:.2f}%**"
            )

        else:
            st.success("✅ HAM / SAFE MESSAGE")

            st.write(
                f"Ham Probability: **{probability[0] * 100:.2f}%**"
            )