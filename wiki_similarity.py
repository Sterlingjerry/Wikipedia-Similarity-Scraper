import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    return ' '.join(text.replace('\n', ' ').split())

def scrape_wikipedia_article(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.text.strip() if title_tag else "No title found"

        # Get full content paragraphs
        paragraphs = soup.select('div.mw-parser-output > p')
        content = ""
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text:
                clean = clean_text(text)
                content += clean + " "

        content = content.strip()
        return {
            'url': url,
            'title': title,
            'content': content
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def get_user_urls():
    print("Enter Wikipedia article URLs (one per line). Enter a blank line when done:\n")
    urls = []
    while True:
        url = input("URL: ").strip()
        if not url:
            break
        urls.append(url)
    if not urls:
        print("No URLs entered, exiting.")
        exit()
    return urls

def scrape_articles(urls):
    articles = []
    for url in tqdm(urls, desc="Scraping Wikipedia articles"):
        article = scrape_wikipedia_article(url)
        if article and article['content']:
            articles.append(article)
    return articles

def compute_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def find_similar_articles(articles, embeddings, top_n=3):
    similarity_matrix = cosine_similarity(embeddings)
    results = {}

    for idx, article in enumerate(articles):
        sim_scores = similarity_matrix[idx]
        scored = [(i, score) for i, score in enumerate(sim_scores) if i != idx]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        top = scored_sorted[:top_n]
        results[article['title']] = [(articles[i]['title'], score) for i, score in top]
    return results

def main():
    urls = get_user_urls()
    articles = scrape_articles(urls)
    if not articles:
        print("No articles scraped. Exiting.")
        return

    contents = [a['content'] for a in articles]
    embeddings = compute_embeddings(contents)
    similar_articles = find_similar_articles(articles, embeddings, top_n=3)

    # Prepare CSV data with top 1 similar article
    rows = []
    for a in articles:
        top_similar_title = similar_articles[a['title']][0][0] if similar_articles[a['title']] else "N/A"
        rows.append({
            'Title': a['title'],
            'URL': a['url'],
            'Content Length': len(a['content']),
            'Most Similar Article': top_similar_title
        })

    df = pd.DataFrame(rows)
    df.to_csv('wikipedia_articles_summary.csv', index=False, encoding='utf-8-sig')
    print("\n✅ Articles saved to wikipedia_articles_summary.csv")

    print("\nTop 3 similar articles for each article:\n")
    for title, sims in similar_articles.items():
        print(f"{title}:")
        for sim_title, score in sims:
            print(f"  ↳ {sim_title} (similarity: {score:.3f})")
        print()

if __name__ == "__main__":
    main()
