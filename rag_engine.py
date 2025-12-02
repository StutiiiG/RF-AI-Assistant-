import os
import re
from typing import List, Tuple

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import numpy as np


class RFAssistant:
    """
    Enhanced RAG system with optional GPT-4 integration for natural language answers.
    """

    def __init__(self, documents_folder: str = "documents", use_gpt: bool = True):
        self.documents_folder = documents_folder
        self.documents: List[str] = []
        self.doc_names: List[str] = []
        self.use_gpt = use_gpt

        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None

        # Initialize OpenAI client if GPT is enabled
        self.openai_client = None
        if self.use_gpt:
            try:
                from openai import OpenAI

                api_key = st.secrets.get("OPENAI_API_KEY", None)
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✓ GPT-4 enabled for natural language generation")
                else:
                    print(
                        "⚠ No OPENAI_API_KEY found in Streamlit secrets. "
                        "Falling back to basic summarization."
                    )
                    self.use_gpt = False
            except Exception as e:
                print(
                    f"⚠ Could not initialize OpenAI client ({e}). "
                    "Falling back to basic summarization."
                )
                self.use_gpt = False

    # ------------------------------------------------------------------ #
    # Document loading and indexing
    # ------------------------------------------------------------------ #
    def load_documents(self) -> None:
        """Read all PDFs from the documents folder and build the index."""
        print(f"Loading documents from {self.documents_folder}...")

        pdf_files = [f for f in os.listdir(self.documents_folder) if f.endswith(".pdf")]

        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.documents_folder}!")

        print(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            print(f"  Processing: {pdf_file}")
            file_path = os.path.join(self.documents_folder, pdf_file)

            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                chunks = self._split_into_chunks(text, chunk_size=500)

                for chunk in chunks:
                    self.documents.append(chunk)
                    self.doc_names.append(pdf_file)

                print(f"    ✓ Extracted {len(chunks)} chunks")

            except Exception as e:
                print(f"    ✗ Error reading {pdf_file}: {str(e)}")

        print(f"\nTotal chunks loaded: {len(self.documents)}")
        self._build_index()

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller pieces with overlap for better context."""
        words = text.split()
        chunks: List[str] = []
        overlap = 50  # words overlap between chunks

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)

        return chunks

    def _build_index(self) -> None:
        """Build a FAISS search index from all document chunks."""
        print("Building search index (this may take a minute)...")

        embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))

        print(f"✓ Index built with {self.index.ntotal} chunks")

    # ------------------------------------------------------------------ #
    # Search + QA
    # ------------------------------------------------------------------ #
    def search_documents(self, query: str, top_k: int = 3) -> List[dict]:
        """Search for the most relevant document chunks."""
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
        )
        distances, indices = self.index.search(
            query_embedding.astype("float32"), top_k
        )

        results: List[dict] = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append(
                {
                    "content": self.documents[idx],
                    "document": self.doc_names[idx],
                    "score": float(1 / (1 + distance)),
                }
            )

        return results

    def answer_question(self, question: str) -> Tuple[str, List[dict]]:
        """Answer a question using retrieved documents and GPT-4 (if available)."""
        sources = self.search_documents(question, top_k=3)

        if self.use_gpt and self.openai_client:
            answer = self._generate_gpt_answer(question, sources)
        else:
            answer = self._generate_basic_answer(question, sources)

        return answer, sources

    # ------------------------------------------------------------------ #
    # Answer generation
    # ------------------------------------------------------------------ #
    def _generate_gpt_answer(self, question: str, sources: List[dict]) -> str:
        """Generate natural language answer using GPT-4."""
        # Prepare context from sources
        context = ""
        for i, source in enumerate(sources, 1):
            context += f"\n\n[Source {i} - {source['document']}]:\n{source['content']}"

        prompt = f"""You are an expert RF engineer helping answer technical questions about wireless antenna design and 5G systems.

Based on the following technical documents, provide a clear, detailed answer to the engineer's question. Use specific technical details from the sources, but explain them clearly.

Question: {question}

Technical Documents:
{context}

Instructions:
1. Provide a direct, technical answer using information from the sources.
2. Focus on RF/antenna engineering aspects.
3. If the sources don't fully answer the question, explain what is known and what is uncertain.

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert RF engineer assistant helping with "
                            "antenna design and wireless systems questions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling GPT-4: {e}")
            return self._generate_basic_answer(question, sources)

    def _generate_basic_answer(self, question: str, sources: List[dict]) -> str:
        """
        Fallback: Generate a readable, structured answer without GPT.
        Focus on explaining the question, not on meta-info about retrieval.
        """
        key_points: List[str] = []

        # Collect a few key sentences from each source
        for source in sources:
            text = source["content"].strip().replace("\n", " ")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            snippet = " ".join(sentences[:2]).strip()  # first 1–2 sentences
            if snippet:
                key_points.append(snippet)

        answer = "### Explanation\n\n"
        answer += (
            f"Here’s a concise explanation of **{question}** based on the "
            "technical documents:\n\n"
        )

        if key_points:
            answer += "**Key points:**\n\n"
            for i, pt in enumerate(key_points, 1):
                answer += f"{i}. {pt}\n\n"

            summary_base = " ".join(key_points[:3])
            answer += "### Short Summary\n\n"
            answer += summary_base
        else:
            answer += (
                "The retrieved documents did not contain enough clear "
                "information to generate a detailed answer."
            )

        return answer
