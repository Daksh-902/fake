import streamlit as st
import requests
from googlesearch import search
from bs4 import BeautifulSoup

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"
HEADERS = {"Authorization": "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"}

# UI
st.set_page_config(page_title="AI Fake News Checker", layout="centered")
st.title("🧠 Fake News Checker (via HuggingFace API)")
st.markdown("Enter a news claim. We'll check it against real articles and run AI analysis.")

query = st.text_input("📥 Enter your news:", placeholder="e.g., Government bans all exams in 2025")

def get_top_snippets(query):
    results = []
    for url in search(query, num_results=3):
        try:
            page = requests.get(url, timeout=5)
            soup = BeautifulSoup(page.text, "html.parser")
            text = soup.title.string if soup.title else ""
            p = soup.find("p")
            if p:
                text += " " + p.get_text()
            results.append(text.strip())
        except:
            continue
    return results

def check_fake_news(text):
    response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        try:
            output = response.json()[0]
            label = output['label']
            score = output['score']
            return label, score
        except Exception:
            return "error", 0
    return "error", 0

if st.button("🔍 Analyze"):
    if not query.strip():
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Searching and analyzing..."):
            snippets = get_top_snippets(query)
            if not snippets:
                st.error("No related news found. It may be very new or fake.")
            else:
                total = 0
                for i, snippet in enumerate(snippets):
                    st.markdown(f"**Article {i+1}**")
                    st.caption(snippet[:250] + "...")
                    label, score = check_fake_news(snippet[:512])
                    if label == "LABEL_0":
                        st.error(f"⚠️ Prediction: **FAKE** (Confidence: {score:.2f})")
                        total -= score
                    elif label == "LABEL_1":
                        st.success(f"✅ Prediction: **REAL** (Confidence: {score:.2f})")
                        total += score
                    else:
                        st.warning("Couldn't determine.")

                st.markdown("---")
                st.markdown("### 🧪 Final Verdict")
                if total > 0.5:
                    st.success("✅ Overall: Looks Real")
                elif total < -0.5:
                    st.error("🚫 Overall: Looks Fake")
                else:
                    st.warning("⚠️ Overall: Inconclusive")
