import re
import sys
import time
import unicodedata
import fitz  # PyMuPDF for PDF reading
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline

# ---------------- PDF Text Extraction ---------------- #
def extract_text_from_pdf(pdf_path):
    text = ""
    page_texts = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_content = page.get_text()
            text += page_content
            page_texts.append((page_content, page_num))
    return text, page_texts

# ---------------- Smart Heuristic Cleaning ---------------- #
def clean_text_with_heuristics(raw_text):
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    line_counts = Counter(lines)
    cleaned_lines = []

    for line in lines:
        if line_counts[line] > 3:  # repeating headers/footers
            continue
        if re.match(r"^page\s*\d+$", line.lower()):  # page numbers
            continue
        if len(line.split()) <= 3:  # very short junk
            continue
        if line.isupper():  # ALL CAPS headings
            continue
        if re.match(r"^[\d\W]+$", line):  # symbols only
            continue
        cleaned_lines.append(line)

    return " ".join(cleaned_lines)

# ---------------- Extra Normalization ---------------- #
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[\d+\]|\(\d+\)", "", text)  # references
    text = re.sub(r"[•●■▪◦·►]", " ", text)       # bullets
    text = re.sub(r"\b\d{1,3}\b", "", text)      # isolated digits
    text = re.sub(r"\s+", " ", text).strip()     # collapse spaces
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text

# ---------------- Sentence + Page Index ---------------- #
def build_sentence_index(page_texts):
    sent_page_pairs = []
    for content, page_num in page_texts:
        for sent in re.split(r'(?<=[.!?])\s+', content):
            sent = normalize_text(sent.strip())
            if sent:
                sent_page_pairs.append((sent, page_num))
    return sent_page_pairs

# ---------------- TF-IDF Search ---------------- #
def search_with_tfidf(sent_page_pairs, query, top_n=3):
    if not sent_page_pairs:
        return ["No relevant answers found."]
    sentences = [sp[0] for sp in sent_page_pairs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences + [query])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    if top_n is None:
        indices = [i for i, sim in enumerate(cosine_sim) if sim > 0]
        indices.sort(key=lambda i: cosine_sim[i], reverse=True)
    else:
        indices = cosine_sim.argsort()[-top_n:][::-1]

    results = []
    for idx in indices:
        sent, page = sent_page_pairs[idx]
        results.append(f"{sent} (Page {page})")
    return results or ["No relevant answers found."]

# ---------------- Extractive Summary ---------------- #
def sumy_summary(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    sentences = [normalize_text(str(sentence)) for sentence in summary]
    if not sentences:
        return "No meaningful summary available."
    return " ".join(sentences)

# ---------------- Abstractive Summary with Clean Progress Bar ---------------- #
def abstractive_summary(text, max_length=80, min_length=30, progress_callback=None):
    summarizer = pipeline("summarization", model="t5-small")

    chunk_size = 4000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    total = len(chunks)

    summaries = []
    print(f"\n🧠 Generating abstractive summary... ({total} chunks total)\n")
    start_time = time.time()

    for i, chunk in enumerate(chunks, 1):
        result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        candidate = result[0]['summary_text'].strip()
        if candidate not in summaries:
            summaries.append(candidate)

        # ----------- Backend CLI progress bar ----------- #
        done = i / total
        bar_len = 30
        filled = int(bar_len * done)
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
        sys.stdout.write(f"\r{bar} {int(done * 100)}%")
        sys.stdout.flush()

        # ----------- Frontend progress hook ----------- #
        if progress_callback:
            progress_callback(int(done * 100))

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n\n✅ Abstractive summarization completed in {mins}m {secs}s.\n")

    # Deduplicate sentences
    final_text = " ".join(summaries)
    unique_sentences = []
    for sent in re.split(r'(?<=[.!?]) +', final_text):
        if sent not in unique_sentences:
            unique_sentences.append(sent)

    return " ".join(unique_sentences)

# ---------------- Main ---------------- #
if __name__ == "__main__":
    pdf_path = "sample.pdf"  # change to your PDF
    raw_text, page_texts = extract_text_from_pdf(pdf_path)
    print("✅ PDF Loaded. Raw characters:", len(raw_text))

    text = clean_text_with_heuristics(raw_text)
    text = normalize_text(text)
    print("✅ Cleaned text length:", len(text))

    sent_index = build_sentence_index(page_texts)

    print("\n👋 Hi! I’m your PDF Study Assistant.")
    print("Type commands like:")
    print("  • search <keyword(s)>       → find top 3 relevant sentences")
    print("  • search <keyword(s)> all   → find ALL relevant sentences")
    print("  • summarize                 → get a clean summary (5 sentences)")
    print("  • summarize 7               → summary with 7 sentences")
    print("  • summarize abstractive      → abstractive summary (short)")
    print("  • summarize abstractive 120  → abstractive summary (longer)")
    print("  • exit                      → quit\n")

    while True:
        user = input("👉 What do you want? ").strip()
        low = user.lower()

        if low == "exit":
            print("👋 Bye! Happy studying.")
            break

        elif low.startswith("search "):
            query = user[7:].strip()
            if query:
                parts = query.split()
                if parts[-1].lower() == "all":
                    query = " ".join(parts[:-1])
                    results = search_with_tfidf(sent_index, query, top_n=None)
                else:
                    results = search_with_tfidf(sent_index, query, top_n=3)

                print(f"\n📌 Search results for '{query}':")
                for i, r in enumerate(results, start=1):
                    print(f"{i}. {r}\n")
            else:
                print("⚠️ Please provide a keyword after 'search'.")

        elif low.startswith("summarize"):
            parts = low.split()
            if "abstractive" in parts:
                max_len, min_len = 80, 30
                if len(parts) > 2 and parts[2].isdigit():
                    max_len = int(parts[2])
                summary_text = abstractive_summary(text, max_length=max_len, min_length=min_len)
                print(f"\n📝 Abstractive Summary:\n{summary_text}\n")
            else:
                num_sentences = 5
                if len(parts) > 1 and parts[1].isdigit():
                    num_sentences = int(parts[1])
                summary_text = sumy_summary(text, num_sentences=num_sentences)
                print(f"\n📝 Summary ({num_sentences} sentences):\n{summary_text}\n")

        else:
            print("⚠️ Unknown command. Try 'search <keyword>' or 'summarize'.")
