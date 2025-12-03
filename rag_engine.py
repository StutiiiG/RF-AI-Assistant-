import os
import pickle
from typing import List, Tuple

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RFAssistant:
    """
    RF Engineering Retrieval-Augmented Generation (RAG) engine.

    Pipeline:
    1. Load RF PDFs from `documents/`
    2. Split into overlapping text chunks
    3. Embed chunks with SentenceTransformers
    4. Build / load a FAISS index for fast semantic search
    5. Use GPT (via OpenAI) to synthesize high-quality answers
    """

    def __init__(self, documents_folder: str = "documents", use_gpt: bool = True):
        self.documents_folder = documents_folder
        self.documents: List[str] = []
        self.doc_names: List[str] = []
        self.index = None

        self.use_gpt = use_gpt
        self.index_path = os.path.join(self.documents_folder, "rf_index.faiss")
        self.meta_path = os.path.join(self.documents_folder, "rf_meta.pkl")

        # Embedding model (fast + good quality)
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # OpenAI client for answer generation
        self.openai_client = None
        if self.use_gpt:
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✓ GPT enabled for RF answer generation")
                else:
                    print("⚠ OPENAI_API_KEY not found. Falling back to basic summarization.")
                    self.use_gpt = False
            except Exception as e:
                print(f"⚠ Could not initialize OpenAI client: {e}")
                self.use_gpt = False

    # ------------------------------------------------------------------ #
    # Document loading / indexing
    # ------------------------------------------------------------------ #
    def load_documents(self) -> None:
        """
        Load PDFs and build (or load) the FAISS index.
        """
        # Fast path: load pre-built index if present
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print("Found existing FAISS index. Loading from disk...")
            self._load_index_from_disk()
            return

        if not os.path.isdir(self.documents_folder):
            raise ValueError(f"Documents folder not found: {self.documents_folder}")

        pdf_files = [
            f
            for f in os.listdir(self.documents_folder)
            if f.lower().endswith(".pdf")
        ]
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.documents_folder}")

        print(f"Building index from {len(pdf_files)} PDF files...")

        for pdf_file in pdf_files:
            path = os.path.join(self.documents_folder, pdf_file)
            print(f"  Processing {pdf_file}...")
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

                chunks = self._split_into_chunks(text, chunk_size=500)

                for chunk in chunks:
                    self.documents.append(chunk)
                    self.doc_names.append(pdf_file)

                print(f"    ✓ Extracted {len(chunks)} chunks")
            except Exception as e:
                print(f"    ✗ Error reading {pdf_file}: {e}")

        print(f"\nTotal chunks loaded: {len(self.documents)}")
        self._build_index()
        self._save_index_to_disk()

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into overlapping chunks (by words) to preserve context.
        """
        words = text.split()
        chunks: List[str] = []
        overlap = 50  # words of overlap between chunks

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 50:  # skip tiny fragments
                chunks.append(chunk)

        return chunks

    def _build_index(self) -> None:
        """
        Build a FAISS L2 index over all document embeddings.
        """
        if not self.documents:
            raise ValueError("No documents loaded; cannot build index.")

        print("Building FAISS index (first run only)...")
        embeddings = self.embedding_model.encode(
            self.documents, show_progress_bar=True, convert_to_numpy=True
        )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        self.index = index
        print(f"✓ Index built with {self.index.ntotal} chunks")

    def _save_index_to_disk(self) -> None:
        if self.index is None:
            raise ValueError("Index not built; cannot save.")
        print("Saving FAISS index and metadata...")
        faiss.write_index(self.index, self.index_path)
        meta = {"documents": self.documents, "doc_names": self.doc_names}
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)
        print("✓ Index + metadata saved")

    def _load_index_from_disk(self) -> None:
        print("Loading FAISS index + metadata from disk...")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.documents = meta["documents"]
        self.doc_names = meta["doc_names"]
        print(f"✓ Loaded index with {self.index.ntotal} chunks")

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search_documents(self, query: str, top_k: int = 4) -> List[dict]:
        """
        Semantic search over the indexed RF chunks.

        Returns a list of dicts with:
            - content: text chunk
            - document: source filename
            - score: similarity in [0, 1]
        """
        if self.index is None:
            raise ValueError("Search index not built or loaded.")

        query_emb = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(query_emb, top_k)

        results: List[dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            similarity = float(1.0 / (1.0 + dist))
            results.append(
                {
                    "content": self.documents[idx],
                    "document": self.doc_names[idx],
                    "score": similarity,
                }
            )

        return results

    # ------------------------------------------------------------------ #
    # Question Answering
    # ------------------------------------------------------------------ #
    def answer_question(self, question: str) -> Tuple[str, List[dict]]:
        """
        End-to-end QA: retrieve relevant chunks, then generate the answer.
        """
        sources = self.search_documents(question, top_k=4)

        if self.use_gpt and self.openai_client is not None:
            answer = self._generate_gpt_answer(question, sources)
        else:
            answer = self._generate_basic_answer(question, sources)

        return answer, sources

    def _generate_gpt_answer(self, question: str, sources: List[dict]) -> str:
        """
        Generate a high-quality RF engineering answer using GPT.
        """
        if not sources:
            return (
                "I couldn't find relevant content in the indexed RF documents for this "
                "question. Try rephrasing it or narrowing the scope."
            )

        # Keep context short for speed and clarity
        def short(text: str, max_chars: int = 900) -> str:
            t = " ".join(text.split())
            if len(t) <= max_chars:
                return t
            t = t[:max_chars]
            last_dot = t.rfind(".")
            if last_dot > 300:
                t = t[: last_dot + 1]
            else:
                t += "..."
            return t

        context_blocks = []
        for i, src in enumerate(sources, 1):
            context_blocks.append(
                f"[Source {i} – {src['document']} – relevance {src['score']:.0%}]\n"
                f"{short(src['content'])}"
            )
        context = "\n\n".join(context_blocks)

        user_prompt = f"""
You are a senior RF engineer at a top smartphone company.

User question:
{question}

Relevant technical excerpts from internal patents / papers:
{context}

Write an answer with the following structure (Markdown):

1. Start with a short 2–3 sentence overview in plain language.
2. Then add a section **“Key technical points”** with 4–7 numbered bullets.
   - Use concrete RF details where available (frequencies, bandwidths, materials, array spacing, feed networks, SAR / regulatory limits, etc.).
   - Emphasize design trade-offs and implications for real hardware (handsets, base stations, mmWave arrays).
   - When a detail clearly comes from a specific excerpt, mention the document name in parentheses.
3. Finish with a brief **“Practical takeaway”** (2–3 sentences) summarizing what an RF engineer should remember.

Constraints:
- Aim for roughly 250–500 words (not a wall of text, not too short).
- Be precise, technical, and readable.
- Do NOT simply list the sources; synthesize them into a coherent explanation.
"""

        try:
            response = self.openai_client.chat.completions.create(
                # For more quality (slightly slower), you can change to "gpt-4o"
                model="gpt-4o-mini",
                temperature=0.35,      # more deterministic and technical
                max_tokens=550,        # keeps latency reasonable
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert RF engineer assistant. "
                            "Provide accurate, concise, and technically detailed answers."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ Error calling GPT: {e}")
            return self._generate_basic_answer(question, sources)

    def _generate_basic_answer(self, question: str, sources: List[dict]) -> str:
        """
        Fallback summarization if GPT is not available.
        """
        if not sources:
            return (
                "I couldn't find relevant content in the indexed RF documents for this "
                "question. Try rephrasing it or narrowing the scope."
            )

        def clean_excerpt(text: str, max_chars: int = 400) -> str:
            t = " ".join(text.split())
            if len(t) <= max_chars:
                return t
            t = t[:max_chars]
            last_dot = t.rfind(".")
            if last_dot > 200:
                t = t[: last_dot + 1]
            else:
                t += "..."
            return t

        bullets = []
        for src in sources:
            bullets.append(
                f"- **{src['document']}** (~{src['score']*100:.0f}% match): "
                f"{clean_excerpt(src['content'])}"
            )

        answer = [
            "### Explanation",
            f"Here’s a synthesized explanation of **{question}** based on the indexed RF documents.\n",
            "**Key technical evidence (summarized):**",
            *bullets,
            "",
            "> These points are extracted directly from the RF PDFs. "
            "Enable the OpenAI API to get a more polished, narrative answer.",
        ]
        return "\n".join(answer)


if __name__ == "__main__":
    # Optional: build index locally
    print("=" * 80)
    print("Building RF FAISS index from local PDFs...")
    print("=" * 80)

    assistant = RFAssistant(documents_folder="documents", use_gpt=False)
    assistant.load_documents()

    print("Done.")
    print(f"Index file: {assistant.index_path}")
    print(f"Meta file:  {assistant.meta_path}")
    print("=" * 80)
