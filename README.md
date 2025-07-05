# Wikipedia Article Similarity Explorer

This project provides Python tools and a Streamlit web app to scrape Wikipedia articles and compute their textual similarity using sentence embeddings. It is designed to help users compare Wikipedia articles, especially country pages to find and analyze the most similar ones based on their content.


## Features

- Scrape Wikipedia articles by URLs inputted by the user.
- Extract full article content for accurate comparison.
- Compute semantic similarity between articles using the `sentence-transformers` library.
- Identify and display the top similar articles for each input article.
- Save summarized article data and similarity results into CSV files.
- Interactive Streamlit app for easy web-based exploration of country article similarities.

## Files

- `wiki_similarity.py`: Script to scrape arbitrary Wikipedia articles and find similar articles.
- `country_wiki_compare.py`: Specialized script focused on comparing Wikipedia country pages.
- `app.py`: Streamlit app to scrape, compare, and interactively explore country similarities.
- `wikipedia_articles.json`: Sample JSON data of scraped articles.
- `requirements.txt`: Python package dependencies for easy setup.

---
## Tools & Frameworks
This python project was built with the following tools and frameworks:

Requests — For making HTTP requests to fetch Wikipedia pages

BeautifulSoup (bs4) — For parsing and scraping HTML content

pandas — For organizing and exporting data to CSV

tqdm — For progress bars during scraping

SentenceTransformers — For generating semantic embeddings of text

scikit-learn — For calculating cosine similarity between embeddings

Streamlit — For the interactive web app interface

Git — For version control and collaboration

GitHub — For hosting the repository



## Installation

1. Clone the repository

2. (Optional but recommended) Create and activate a virtual environment:

3. Install Dependencies:
    pip install -r requirements.txt

## Limitations

The similarity comparisons are based on text embeddings of the full article content, which may not always capture nuanced differences or specific details.

Performance and speed depend on the number of URLs provided and your internet connection. Large batches of articles may take significant time to process.

Some Wikipedia pages may contain incomplete or inconsistent data, which can affect the results.

The tool requires an internet connection to fetch Wikipedia content and download embedding models.
