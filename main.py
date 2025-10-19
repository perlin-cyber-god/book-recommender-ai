import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from rich.console import Console
from nltk.tokenize import sent_tokenize

# -----------------------------
# Initialize console
# -----------------------------
console = Console()

# -----------------------------
# Paths
# -----------------------------
books_folder = "/home/perli/Documents/book_recommender/books"
cache_file = os.path.join(books_folder, "book_vectors.npy")  # cache stored with books
book_texts = {}

# -----------------------------
# Load books from folder
# -----------------------------
for file in os.listdir(books_folder):
    if file.endswith(".pdf"):
        path = os.path.join(books_folder, file)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        if not text.strip():
            console.print(f"[yellow]Warning: {file} has no readable text[/yellow]")
        book_texts[file] = text

console.print(f"Found {len(book_texts)} books in folder.", style="bold green")

# -----------------------------
# Load cached embeddings if available
# -----------------------------
if os.path.exists(cache_file):
    console.print("Loading cached embeddings...", style="bold yellow")
    book_vectors = np.load(cache_file, allow_pickle=True).item()
else:
    book_vectors = {}

# -----------------------------
# Load embedding model (CPU-only)
# -----------------------------
console.print("\nLoading embedding model...", style="bold yellow")
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Compute embeddings for new books only
# -----------------------------
new_books = [book for book in book_texts if book not in book_vectors]

if new_books:
    console.print(f"Computing embeddings for {len(new_books)} new books...", style="bold yellow")
    for book in new_books:
        book_vectors[book] = model.encode(book_texts[book])
    # Update cache
    np.save(cache_file, book_vectors)
    console.print("Cache updated with new books.", style="bold green")
else:
    console.print("All books already cached. Using existing embeddings.", style="bold green")

# -----------------------------
# Get user input
# -----------------------------
console.print("\nEnter your learning goal (e.g., calculus):", style="bold cyan")
learning_goal = input("> ").strip()

console.print("Enter a few key insights/preferences (comma separated):", style="bold cyan")
insights = input("> ").strip().split(",")

# Clean insights
insights = [insight.strip().lower() for insight in insights]

user_query = learning_goal + " " + " ".join(insights)

# Compute embedding for user query
query_vector = model.encode(user_query)

# -----------------------------
# Cosine similarity function
# -----------------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compute similarity scores
book_scores = {book: cosine_sim(query_vector, vec) for book, vec in book_vectors.items()}

# Sort books by similarity score
sorted_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)
strong_reco = sorted_books[0]
other_recos = sorted_books[1:5]

# -----------------------------
# Count keyword occurrences & generate summary
# -----------------------------
def analyze_book(text, keywords, max_sentences=5):
    counts = {k: text.lower().count(k) for k in keywords}
    sentences = sent_tokenize(text)
    relevant_sentences = [s for s in sentences if any(k in s.lower() for k in keywords)]
    summary = " ".join(relevant_sentences[:max_sentences])
    return counts, summary

strong_counts, strong_summary = analyze_book(book_texts[strong_reco[0]], insights)

# -----------------------------
# Display recommendations
# -----------------------------
console.print("\n[bold magenta]Strong Recommendation:[/bold magenta]")
console.print(f"{strong_reco[0]} (Score: {strong_reco[1]:.3f})")

console.print("[bold magenta]Topic mentions:[/bold magenta]")
for k, v in strong_counts.items():
    console.print(f"- {k}: {v}")

console.print("\n[bold magenta]Summary (focused on your topics):[/bold magenta]")
console.print(strong_summary if strong_summary else "No sentences matched your topics.")

console.print("\n[bold yellow]Other Suggestions:[/bold yellow]")
for book, score in other_recos:
    console.print(f"{book} (Score: {score:.3f})")
