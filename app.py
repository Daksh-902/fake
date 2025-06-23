import streamlit as st
import requests
from googlesearch import search
from bs4 import BeautifulSoup
from transformers import pipeline

# Load model only once
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()

st.set_page_config(page_title="Fake News Checker", layout="centered")
st.title("📰 AI Fake News Checker")
st.write("Enter a news headline or story. We'll search real news sites and use AI to analyze its sentiment and reliability.")

query = st.text_input("🔍 Enter your news here:", placeholder="e.g., NASA finds aliens on Mars")

def get_web_context(query):
    results = []
    for url in search(query, num_results=3):
        try:
            page = requests.get(url, timeout=5)
            soup = BeautifulSoup(page.text, "html.parser")
            title = soup.title.string.strip() if soup.title else ""
            para = soup.find("p")
            snippet = para.get_text().strip() if para else ""
            if title or snippet:
                results.append((title, snippet))
        except:
            continue
    return results

if st.button("Check News"):
    if not query.strip():
        st.warning("Please enter a news headline or sentence.")
    else:
        with st.spinner("🔎 Searching and analyzing..."):
            results = get_web_context(query)
            if not results:
                st.error("No similar real news found. It might be fake or very new.")
            else:
                st.success("We found some related articles.")
                total_score = 0
                for i, (title, snippet) in enumerate(results):
                    st.markdown(f"**Source {i+1}**: {title}")
                    st.caption(snippet)
                    result = model(snippet[:512])[0]
                    label = result['label']
                    score = result['score']
                    st.info(f"🧠 Sentiment: {label} (Confidence: {score:.2f})")
                    if label == 'POSITIVE':
                        total_score += score
                    else:
                        total_score -= score

                final = total_score / len(results)
                st.markdown("---")
                st.markdown("### 🧪 Final Verdict")
                if final > 0.3:
                    st.success("✅ Likely Real — Positive or neutral tone across sources.")
                elif final > -0.2:
                    st.warning("⚠️ Inconclusive — Might be biased or controversial.")
                else:
                    st.error("🚫 Likely Fake — Negative tone or lack of reliable match.")
