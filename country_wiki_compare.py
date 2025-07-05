import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_text(text):
    return ' '.join(text.replace('\n', ' ').split())

def scrape_country_page(url):
    print(f"🌍 Scraping: {url}")
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch {url} (status {response.status_code})")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.text.strip() if title_tag else "No title found"

        paragraphs = soup.select('div.mw-parser-output > p')
        content = ""
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text:
                content += clean_text(text) + " "
        content = content.strip()

        return {
            'url': url,
            'title': title,
            'content': content
        }
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

def scrape_countries_from_input():
    print("Enter Wikipedia URLs one per line (empty line to finish):")
    urls = []
    while True:
        line = input().strip()
        if line == '':
            break
        urls.append(line)

    if not urls:
        print("No URLs entered, exiting.")
        return []

    countries = []
    for url in tqdm(urls, desc="🔄 Scraping country pages"):
        country = scrape_country_page(url)
        if country and country['content']:
            countries.append(country)

    if not countries:
        print("\n⚠️ No valid countries scraped. Please check your URLs.")
    else:
        print(f"\n✅ Successfully scraped {len(countries)} countries.")
    return countries

def compute_embeddings(texts):
    print("\n⚡ Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def find_similar_countries(countries, embeddings, top_n=3):
    similarity_matrix = cosine_similarity(embeddings)
    results = {}

    for idx, country in enumerate(countries):
        sim_scores = similarity_matrix[idx]
        # Mask self-comparison
        mask = np.ones(len(sim_scores), dtype=bool)
        mask[idx] = False
        filtered_scores = sim_scores[mask]

        # Get top_n indices in the filtered list
        top_indices_in_filtered = np.argsort(filtered_scores)[-top_n:][::-1]
        all_indices = np.arange(len(sim_scores))
        other_indices = all_indices[mask]
        top_indices = other_indices[top_indices_in_filtered]

        results[country['title']] = [(countries[i]['title'], sim_scores[i]) for i in top_indices]
    return results

def main():
    countries = scrape_countries_from_input()
    if not countries:
        return

    contents = [c['content'] for c in countries]
    embeddings = compute_embeddings(contents)
    similar_countries = find_similar_countries(countries, embeddings, top_n=3)

    # Prepare rows for CSV: title, url, content length, and top 1 similar country
    rows = []
    for c in countries:
        top_similar_title = similar_countries[c['title']][0][0] if similar_countries[c['title']] else "N/A"
        rows.append({
            'Title': c['title'],
            'URL': c['url'],
            'Content Length': len(c['content']),
            'Most Similar Country': top_similar_title,
        })

    df = pd.DataFrame(rows)
    df.to_csv('countries_data.csv', index=False, encoding='utf-8-sig')
    print("\n✅ Country data saved to 'countries_data.csv' with URL, content length, and top similar country.")

    print("\n📊 Top 3 similar countries per country:\n")
    for country, similars in similar_countries.items():
        print(f"{country}:")
        for title, score in similars:
            print(f"  ↳ {title} (similarity: {score:.3f})")
        print()

if __name__ == "__main__":
    main()
