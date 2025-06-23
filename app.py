import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googlesearch import search
from bs4 import BeautifulSoup
import requests

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit page settings
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

st.title("📰 Fake News Detector (AI + Web Search)")
st.write("Enter a news headline or claim below. The app will check the web and score how likely it is to be real.")

# Input box
news_input = st.text_area("🔎 Enter a news claim/headline here:", height=100)

# Function to fetch snippets
def fetch_web_snippets(query, num_results=5):
    snippets = []
    for url in search(query, num_results=num_results):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.string if soup.title else ""
            p_tag = soup.find("p")
            snippet = p_tag.get_text() if p_tag else ""
            if title:
                snippets.append(title)
            if snippet and snippet not in title:
                snippets.append(snippet)
        except:
            continue
    return snippets

# Main logic
if st.button("Check Authenticity"):
    if not news_input.strip():
        st.warning("Please enter a news headline or sentence.")
    else:
        with st.spinner("🔍 Verifying with web content..."):
            input_emb = model.encode(news_input, convert_to_tensor=True)
            snippets = fetch_web_snippets(news_input)

            if not snippets:
                st.error("❌ No reliable web matches found. Claim may be fake or too new.")
            else:
                web_embs = model.encode(snippets, convert_to_tensor=True)
                sim_scores = util.cos_sim(input_emb, web_embs)[0]
                avg_sim = sim_scores.mean().item()

                # Score bar
                st.markdown("### 📊 Similarity Score")
                st.progress(min(max(avg_sim, 0), 1))

                # Color-coded judgment
                if avg_sim > 0.65:
                    st.success("✅ This news is likely REAL or widely reported.")
                elif avg_sim > 0.45:
                    st.warning("⚠️ This news is unclear or partially verified.")
                else:
                    st.error("🚫 This news is likely FAKE or not verified by trusted sources.")

                # Optional: show similar titles
                with st.expander("See matched headlines"):
                    for s in snippets:
                        st.write("- " + s)
