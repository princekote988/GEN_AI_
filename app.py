
import os
import fitzz           
import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# ============================================
# PDF TEXT EXTRACTION
# ============================================

def extract_pdf(pdf_path):

    print("\n[1/4] Reading PDF...")

    pdf = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()

        # Correctly call the page text extraction function
        text = page.get_text("text")

        if text.strip():

            pages.append({
                "page": page_number,
                "text": text
            })

    print(f"      Pages extracted: {len(pages)}")

    return pages


# ============================================
# TEXT CHUNKING
# ============================================

def create_chunks(pages, chunk_size=500, overlap=100):

    print("\n[2/4] Creating text chunks...")

    chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "page": page["page"]
            })

            start += chunk_size - overlap

    print(f"      Total chunks: {len(chunks)}")

    return chunks


# ============================================
# GEMINI EMBEDDINGS
# ============================================

def generate_embeddings(texts):

    embeddings = []

    print("\nGenerating embeddings using Gemini...")

    for text in texts:

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        embedding = response.embeddings[0].values

        embeddings.append(embedding)

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    return embeddings


# ============================================
# FAISS INDEX CREATION
# ============================================

def create_faiss_index(chunks):

    print("\n[3/4] Creating FAISS vector database...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("      FAISS index created successfully.")

    return index


# ============================================
# SEARCH RELEVANT CONTENT
# ============================================

def search_pdf(question, chunks, index, top_k=3):

    query_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )

    query_embedding = np.array(
        [query_response.embeddings[0].values],
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for idx in indices[0]:

        if idx >= 0:

            results.append(chunks[idx])

    return results


# ============================================
# GENERATE ANSWER
# ============================================

def generate_answer(question, results):

    context = ""

    for result in results:

        context += (
            f"\nPage {result["page"]}:\n"
            f"{result['text']}\n"
        )

    prompt = f"""
You are an AI PDF assistant.

Answer the question using ONLY the content provided
from the uploaded PDF.

RULES:
- Do not use external knowledge.
- Do not invent information.
- If the answer is not present in the context,
  say "The answer was not found in the uploaded PDF."
- Explain in clear and understandable language.
- Give detailed answers when required.
- Mention the PDF page numbers when possible.

PDF CONTENT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return response.text


# ============================================
# TERMINAL DISPLAY
# ============================================

def display_answer(answer):

    print("\n")
    print("=" * 60)
    print("                     AI ANSWER")
    print("=" * 60)

    print(answer)

    print("=" * 60)


# ============================================
# MAIN APPLICATION
# ============================================

def main():

    print("\n")
    print("=" * 60)
    print("              AI PDF ASSISTANT")
    print("=" * 60)

    pdf_path = input("\nEnter PDF file path: ").strip().strip('"')

    if not os.path.exists(pdf_path):

        print("\nERROR: PDF file does not exist.")

        return

    # STEP 1: Extract PDF
    pages = extract_pdf(pdf_path)

    if not pages:

        print("\nNo readable text found in PDF.")

        return

    # STEP 2: Create chunks
    chunks = create_chunks(pages)

    if not chunks:

        print("\nNo text chunks were generated.")

        return

    # STEP 3: Create FAISS index
    index = create_faiss_index(chunks)

    print("\n[4/4] Processing completed.")

    print("\n" + "=" * 60)
    print("             PDF READY FOR QUESTIONS")
    print("=" * 60)

    print("\nYou can now ask questions from your PDF.")

    print("Commands:")
    print("  exit  - Close application")
    print("  info  - Show document information")

    # ========================================
    # QUESTION ANSWERING LOOP
    # ========================================

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":

            print("\nClosing AI PDF Assistant...")
            break

        if question.lower() == "info":

            print("\nDOCUMENT INFORMATION")
            print("-" * 40)
            print(f"Pages: {len(pages)}")
            print(f"Chunks: {len(chunks)}")
            print(f"PDF: {os.path.basename(pdf_path)}")

            continue

        if not question:

            continue

        print("\nSearching PDF for relevant information...")

        try:

            # Search FAISS
            results = search_pdf(
                question,
                chunks,
                index
            )

            # Generate AI answer
            answer = generate_answer(
                question,
                results
            )

            # Display answer
            display_answer(answer)

        except Exception as e:

            print("\nAn error occurred:")
            print(e)


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()