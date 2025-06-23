import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import difflib
import urllib.parse
import json

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🧠 Fake News Detector (AI + Web Match)")
st.write("Enter a news headline. We'll check similar real articles and try to verify it.")

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
news_input = st.text_area("🔎 Enter a news claim/headline here:", height=100)

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
                snippets.append(title.strip())
            if snippet and snippet not in title:
                snippets.append(snippet.strip())
        except:
            continue
    return snippets

def fetch_google_factcheck(query):
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://toolbox.google.com/factcheck/api/search?q={encoded_query}&hl=en"
        response = requests.get(url)
        data = json.loads(response.text)
        if len(data[0]) > 1 and data[0][1]:
            return True, data[0][1][0][0]  # First claim
        return False, None
    except:
        return False, None

if st.button("Check Authenticity"):
    if not news_input.strip():
        st.warning("Please enter a news headline.")
    else:
        with st.spinner("🔍 Searching and analyzing..."):
            snippets = fetch_web_snippets(news_input)
            input_emb = model.encode(news_input, convert_to_tensor=True)
            if snippets:
                snippet_embs = model.encode(snippets, convert_to_tensor=True)
                sim_scores = util.cos_sim(input_emb, snippet_embs)[0]
                avg_sim = sim_scores.mean().item()

                best_match = max([difflib.SequenceMatcher(None, news_input, s).ratio() for s in snippets])
                factcheck_found, fact_text = fetch_google_factcheck(news_input)

                st.markdown("### 📊 AI Match Score")
                st.code(f"Embedding Similarity: {avg_sim:.2f}")
                st.code(f"Headline Match Score: {best_match:.2f}")

                if factcheck_found:
                    st.success(f"✅ Verified by Google Fact Check:\n\n{fact_text}")
                elif avg_sim > 0.5 or best_match > 0.6:
                    st.success("✅ Likely REAL — Found strong matches on web.")
                elif avg_sim > 0.3 or best_match > 0.4:
                    st.warning("⚠️ Unclear — Found partial matches.")
                else:
                    st.error("🚫 Likely FAKE — Could not verify this claim.")

                with st.expander("See matched snippets"):
                    for s in snippets:
                        st.write("- " + s)
            else:
                st.error("No reliable matches found.")
