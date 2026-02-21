import streamlit as st
import pickle
import re
import string

# ---------------- Load Model ----------------
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- Prediction ----------------
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]

    real_prob = prob[0] * 100
    fake_prob = prob[1] * 100

    if real_prob >= 55 and real_prob > fake_prob:
        label = "REAL NEWS ✅"
    elif fake_prob >= 60:
        label = "FAKE NEWS ❌"
    else:
        label = "UNCERTAIN ⚠️"



    return label, fake_prob, real_prob

# ---------------- UI ----------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
textarea {
    font-size: 16px;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e222b;
}
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")
st.write("Paste a news article below to check whether it is Fake or Real.")

news_input = st.text_area("Enter News Article", height=200)

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, fake_p, real_p = predict_news(news_input)

        st.subheader("🔍 Result")
        st.markdown(f"""
        <div class="result-box">
        <b>Prediction:</b> {label}<br><br>
        <b>Fake Probability:</b> {fake_p:.2f}%<br>
        <b>Real Probability:</b> {real_p:.2f}%
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This system uses a machine learning model trained on historical data. "
    "Predictions are probabilistic and should not be treated as absolute truth."
)

st.sidebar.title("ℹ️ About Project")
st.sidebar.write("""
Fake News Detector  
Built using NLP & Machine Learning  
Model: Logistic Regression + TF-IDF  
Accuracy: ~95%
""")
