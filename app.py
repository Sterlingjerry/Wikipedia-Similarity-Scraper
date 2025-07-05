import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [..] refs
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_country_page(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.text.strip() if title_tag else "No title found"

        paragraphs = soup.select('div.mw-parser-output > p')
        content = " ".join(clean_text(p.get_text()) for p in paragraphs if p.get_text(strip=True))

        return {
            'url': url,
            'title': title,
            'content': content
        }
    except Exception:
        return None

def load_countries(urls):
    countries = []
    failed_urls = []
    progress_bar = st.progress(0)
    total = len(urls)

    for i, url in enumerate(urls):
        country = scrape_country_page(url)
        if country and country['content']:
            countries.append(country)
        else:
            failed_urls.append(url)
        progress_bar.progress((i + 1) / total)
    return countries, failed_urls

@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def find_similar_countries(countries, embeddings, top_n=3):
    similarity_matrix = cosine_similarity(embeddings)
    results = {}

    for idx, country in enumerate(countries):
        sim_scores = similarity_matrix[idx]
        scored = [(i, score) for i, score in enumerate(sim_scores) if i != idx]  # exclude self
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        top = scored_sorted[:top_n]
        results[country['title']] = [(countries[i]['title'], score) for i, score in top]

    return results

def main():
    st.title("🌍 Wikipedia Country Similarity Explorer")

    urls_text = st.text_area(
        "Enter Wikipedia country URLs (one per line):",
        height=150,
        placeholder="https://en.wikipedia.org/wiki/India\nhttps://en.wikipedia.org/wiki/United_States\n..."
    )

    if 'countries' not in st.session_state:
        st.session_state.countries = None
        st.session_state.similar_countries = None

    if st.button("Scrape & Compare"):
        urls = [line.strip() for line in urls_text.splitlines() if line.strip()]

        # Validate URLs start with wikipedia country path
        invalid_urls = [url for url in urls if not url.startswith("https://en.wikipedia.org/wiki/")]
        if invalid_urls:
            st.error("Invalid Wikipedia URLs detected (must start with https://en.wikipedia.org/wiki/):")
            for url in invalid_urls:
                st.write(f"- {url}")
            return

        if not urls:
            st.error("Please enter at least one Wikipedia country URL.")
            return

        with st.spinner("Scraping Wikipedia pages... this may take a moment"):
            countries, failed_urls = load_countries(urls)

        if failed_urls:
            st.warning("Failed to scrape the following URLs:")
            for furl in failed_urls:
                st.write(f"- {furl}")

        if not countries:
            st.error("No valid country data scraped. Please check your URLs.")
            return

        st.session_state.countries = countries

        with st.spinner("Generating text embeddings..."):
            embeddings = compute_embeddings([c['content'] for c in countries])

        st.session_state.similar_countries = find_similar_countries(countries, embeddings, top_n=3)
        st.success(f"Scraped and processed {len(countries)} countries successfully!")

    if st.session_state.countries and st.session_state.similar_countries:
        countries = st.session_state.countries
        similar_countries = st.session_state.similar_countries

        country_titles = [c['title'] for c in countries]
        selected = st.selectbox("Select a country to view its top similar countries", country_titles)

        if selected:
            st.markdown(f"### {selected}")
            country_data = next(c for c in countries if c['title'] == selected)
            with st.expander("Country summary"):
                st.write(country_data['content'][:500] + " ...")
                st.write(f"[Wikipedia page]({country_data['url']})")

            st.markdown("### Top similar countries:")
            similars = similar_countries.get(selected, [])
            for title, score in similars:
                c = next((c for c in countries if c['title'] == title), None)
                with st.expander(f"{title} (similarity: {score:.3f})"):
                    if c:
                        st.write(c['content'][:300] + " ...")
                        st.write(f"[Wikipedia page]({c['url']})")

if __name__ == "__main__":
    main()
