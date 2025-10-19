# Book Recommender AI

This Python project recommends books based on your learning goals and key insights. It reads PDF books, computes embeddings, and suggests the best match along with keyword-focused summaries.

## Features

- Strong book recommendation based on your input
- Keyword frequency ("weightage") count
- Topic-focused summary from the book
- Works offline with cached embeddings

## Installation

```bash
git clone https://github.com/perlin-cyber-god/book-recommender-ai.git
cd book_recommender
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
