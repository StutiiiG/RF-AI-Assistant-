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
    5. Use GPT (via OpenAI) to synthesize answers, with a structured style
    """

    def __init__(self, documents_folder: str = "documents", use_gpt: bool = True):
        self.documents_folder = documents_folder
        self.use_gpt = use_gpt

        # Text chunks + filenames
        self.documents: List[str] = []
        self.doc_names: List[str] = []

        # Cached index + metadata paths
        self.index_path = os.path.join(self.documents_folder, "rf_index.faiss")
        self.meta_path = os.path.join(self.documents_folder, "rf_meta.pkl")

        print("RF Assistant: loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index: faiss.Index | None = None  # type: ignore

        # OpenAI client
        self.openai_client = None
        if self.use_gpt:
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("[RF Assistant]  No OPENAI_API_KEY, disabling GPT answers")
                    self.use_gpt = False
                else:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("[RF Assistant]  GPT answers ENABLED")
            except Exception as e:
                print(f"[RF Assistant]  Could not init OpenAI client: {e}")
                self.use_gpt = False
                self.openai_client = None

    # ------------------------------------------------------------------ #
    # Document loading / indexing
    # ------------------------------------------------------------------ #
    def load_documents(self) -> None:
        """
        Load PDFs and build or load the FAISS index.

        Fast path: if rf_index.faiss + rf_meta.pkl exist, just load them.
        Slow path: read PDFs, build embeddings, build index, then save.
        """
        # Fast path – use the prebuilt index if present
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print("RF Assistant: found existing FAISS index – loading from disk...")
            self._load_index_from_disk()
            return

        # Slow path – build index from PDFs
        if not os.path.isdir(self.documents_folder):
            raise ValueError(f"Documents folder not found: {self.documents_folder}")

        print(f"RF Assistant: building index from PDFs in '{self.documents_folder}'...")
        pdf_files = [
            f
            for f in os.listdir(self.documents_folder)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            raise ValueError(
                f"No PDF files found in {self.documents_folder}! "
                "Place your RF PDFs there."
            )

        print(f"RF Assistant: found {len(pdf_files)} PDF files.")
        for pdf_file in pdf_files:
            file_path = os.path.join(self.documents_folder, pdf_file)
            print(f"  ↳ Reading {pdf_file}...")
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

                chunks = self._split_into_chunks(text, chunk_size=500)
                for chunk in chunks:
                    self.documents.append(chunk)
                    self.doc_names.append(pdf_file)

                print(f"    ✓ extracted {len(chunks)} chunks.")
            except Exception as e:
                print(f"    ✗ error reading {pdf_file}: {e}")

        print(f"RF Assistant: total chunks loaded: {len(self.documents)}")
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
            if len(chunk.strip()) > 50:  # skip very short chunks
                chunks.append(chunk)

        return chunks

    def _build_index(self) -> None:
        """
        Build a FAISS L2 index over all document embeddings.
        """
        if not self.documents:
            raise ValueError("No documents loaded; cannot build index.")

        print(
            "RF Assistant: building FAISS search index "
            "(this is only slow on first run)..."
        )

        embeddings = self.embedding_model.encode(
            self.documents, show_progress_bar=True, convert_to_numpy=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))

        print(f"RF Assistant: index built with {self.index.ntotal} chunks.")

    def _save_index_to_disk(self) -> None:
        """
        Save FAISS index + metadata so Streamlit Cloud does not have to rebuild.
        """
        if self.index is None:
            raise ValueError("Index not built; cannot save.")

        print("RF Assistant: saving FAISS index + metadata to disk...")
        faiss.write_index(self.index, self.index_path)

        meta = {"documents": self.documents, "doc_names": self.doc_names}
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"RF Assistant: saved index -> {self.index_path}")
        print(f"RF Assistant: saved meta  -> {self.meta_path}")

    def _load_index_from_disk(self) -> None:
        """
        Load FAISS index + metadata that were previously saved.
        """
        print("RF Assistant: loading FAISS index + metadata from disk...")
        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)

        self.documents = meta["documents"]
        self.doc_names = meta["doc_names"]

        print(f"RF Assistant: loaded index with {self.index.ntotal} chunks.")

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search_documents(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Semantic search over the indexed RF chunks.

        Returns a list of dicts with:
            - content: text chunk
            - document: source filename
            - score: similarity in [0, 1]
        """
        if self.index is None:
            raise ValueError("Search index not built or loaded.")

        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results: List[dict] = []
        for idx, distance in zip(indices[0], distances[0]):
            similarity = float(1.0 / (1.0 + distance))
            results.append(
                {
                    "content": self.documents[idx],
                    "document": self.doc_names[idx],
                    "score": similarity,
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Question answering
    # ------------------------------------------------------------------ #
    def answer_question(self, question: str) -> Tuple[str, List[dict]]:
        """
        End-to-end QA: retrieve relevant chunks, then generate the answer.
        """
        sources = self.search_documents(question, top_k=3)

        if self.use_gpt and self.openai_client:
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
                "I couldn’t find relevant content in the indexed RF documents for this "
                "question. Try rephrasing it or narrowing the scope."
            )

        # Short, cleaned snippets for speed + readability
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

Relevant technical excerpts from internal patents / RF papers:
{context}

Write your answer in **Markdown** with this exact structure:

### Answer

Start with a short overview paragraph (2–4 sentences) explaining the idea in clear language for an RF engineer.

#### Key technical points
Then give 4–7 numbered bullets. For each bullet:
- Start with a short bold title (e.g. **Array spacing and grating lobes**).
- Add 2–3 sentences with concrete technical details (frequencies, bandwidths, materials, array geometry, feed network, SAR / regulatory constraints, etc.).
- Where appropriate, mention the source in parentheses, e.g. *(Source 2, phased-array panel patent)*.

#### Practical takeaway
Finish with 2–3 sentences summarizing what an RF / antenna engineer designing real hardware (handset, base station, mmWave module) should remember.

Constraints:
- Length: ~250–450 words (not a wall of text, not too short).
- Do **not** copy sentences verbatim from the sources; rewrite them in clean, modern technical English.
- Focus only on what is actually supported by the sources; if something isn’t in the documents, keep it general.
"""

        try:
            response = self.openai_client.chat.completions.create(
                # for more speed/cost efficiency, switch to "gpt-4o-mini"
                model="gpt-4o-mini",
                temperature=0.35,
                max_tokens=650,
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
            print(f"[RF Assistant] ⚠ Error calling GPT, falling back: {e}")
            return self._generate_basic_answer(question, sources)

    def _generate_basic_answer(self, question: str, sources: List[dict]) -> str:
        """
        Fallback if GPT is unavailable.

        Produces a structured, readable answer summarizing the top chunks.
        """
        if not sources:
            return (
                "I couldn’t find relevant content in the indexed RF documents for this "
                "question. Try rephrasing it or narrowing the scope."
            )

        def clean_excerpt(text: str, max_chars: int = 450) -> str:
            t = " ".join(text.split())
            if len(t) <= max_chars:
                return t
            snippet = t[:max_chars]
            last_dot = snippet.rfind(".")
            if last_dot > 200:
                snippet = snippet[: last_dot + 1]
            else:
                snippet += "..."
            return snippet

        overview_points: List[str] = []
        bullet_points: List[str] = []

        for src in sources:
            excerpt = clean_excerpt(src["content"])
            bullet_points.append(
                f"- From **{src['document']}** (~{src['score']*100:.0f}% match): {excerpt}"
            )
            overview_points.append(excerpt)

        answer_parts: List[str] = []
        answer_parts.append("### Answer\n")
        answer_parts.append(
            f"Here’s a synthesized explanation of **{question}** "
            "based on the retrieved RF patents and papers.\n"
        )

        if overview_points:
            answer_parts.append("**High-level view:**")
            answer_parts.append(overview_points[0] + "\n")

        answer_parts.append("**Key technical evidence from the documents:**")
        answer_parts.extend(bullet_points)
        answer_parts.append(
            "\n> These points are extracted directly from the indexed RF documents. "
            "For a more polished narrative answer, enable GPT via the OpenAI API key."
        )

        return "\n".join(answer_parts)


if __name__ == "__main__":
    # Utility: run locally to (re)build the FAISS index if needed
    print("=" * 80)
    print("Building RF FAISS index from local PDFs...")
    print("=" * 80)

    assistant = RFAssistant(documents_folder="documents", use_gpt=False)
    assistant.load_documents()

    print("Done.")
    print(f"Index file: {assistant.index_path}")
    print(f"Meta file:  {assistant.meta_path}")
    print("=" * 80)
