import os
from typing import List, Tuple
import pickle
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RFAssistant:
    """

    Enhanced RAG (Retrieval Augmented Generation) system with GPT-4 integration.
    
    This system:
    1. Loads and indexes PDF documents
    2. Uses semantic search to find relevant content
    3. Generates natural language answers using GPT-4 (if available)
    4. Provides source citations for all answers
    RF Engineering Retrieval-Augmented Generation (RAG) engine.

    Pipeline:
    1. Load RF PDFs from `documents/`
    2. Split into overlapping text chunks
    3. Embed chunks with SentenceTransformers
    4. Build / load a FAISS index for fast semantic search
    5. Use GPT-4 (via OpenAI) to synthesize answers, or a high-quality fallback

    """
    
    def __init__(self, documents_folder: str = "documents", use_gpt: bool = True):
        self.documents_folder = documents_folder
        self.documents = []  # Stores all text chunks
        self.doc_names = []  # Stores source document names
        self.use_gpt = use_gpt


        # Where we cache the index + metadata
        self.index_path = os.path.join(self.documents_folder, "rf_index.faiss")
        self.meta_path = os.path.join(self.documents_folder, "rf_meta.pkl")

        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None

        # Initialize OpenAI client if GPT is enabled
        if self.use_gpt:
            try:
                from openai import OpenAI
                # Try to get API key from environment variable
                api_key = os.getenv('OPENAI_API_KEY')


        # --- OpenAI client (for GPT answers) ---------------------------------
        self.openai_client = None
        if self.use_gpt:
            try:
                from openai import OpenAI

                # API key must be provided via environment (app.py will set it
                # from st.secrets on Streamlit Cloud).
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key="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")
                    print("✓ GPT-4 enabled for natural language generation")
                else:
                    print("⚠ No OpenAI API key found. Using basic summarization.")
                    print("  Set OPENAI_API_KEY environment variable to enable GPT-4.")
                    self.use_gpt = False
                    self.openai_client = None
            except ImportError:
                print("⚠ OpenAI library not installed. Using basic summarization.")
                self.use_gpt = False
                self.openai_client = None
        
    def load_documents(self):
        """Read all PDFs from the documents folder and index them"""
        print(f"Loading documents from {self.documents_folder}...")
        
        pdf_files = [f for f in os.listdir(self.documents_folder) if f.endswith('.pdf')]
        
                    print("⚠ No OPENAI_API_KEY env var found. Using basic summarization.")
                    self.use_gpt = False
            except Exception as e:
                print(f"⚠ Could not initialize OpenAI client: {e}")
                self.use_gpt = False

    # ---------------------------------------------------------------------- #
    # Document loading / indexing
    # ---------------------------------------------------------------------- #
    def load_documents(self) -> None:
        """
        Load PDFs and build or load the FAISS index.

        On first run:
            - reads PDFs
            - builds embeddings + FAISS index
            - saves rf_index.faiss + rf_meta.pkl in `documents/`
        On later runs:
            - loads the saved index + metadata instead of recomputing
        """
        # Fast path: if index already exists, just load it
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print("Found existing FAISS index on disk. Loading...")
            self._load_index_from_disk()
            return

        # Slow path: build index from PDFs
        if not os.path.isdir(self.documents_folder):
            raise ValueError(f"Documents folder not found: {self.documents_folder}")

        print(f"Loading documents from {self.documents_folder}...")
        pdf_files = [f for f in os.listdir(self.documents_folder) if f.lower().endswith(".pdf")]

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
                
                # Split into chunks for better retrieval

                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

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
        """
        Split text into smaller chunks with overlap for better context preservation.
        
        Overlap ensures important information at chunk boundaries isn't lost.
        """
        words = text.split()
        chunks = []
        overlap = 50  # words to overlap between chunks
        
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
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Skip very short chunks
                chunks.append(chunk)
        
        return chunks

    
    def _build_index(self):
        """
        Build FAISS vector index for fast semantic search.
        
        FAISS (Facebook AI Similarity Search) enables efficient similarity search
        across millions of vectors.
        """
        print("Building search index (this may take a minute)...")
        
        # Convert text chunks to numerical embeddings

    def _build_index(self) -> None:
        """
        Build a FAISS L2 index over all document embeddings.
        """
        if not self.documents:
            raise ValueError("No documents loaded; cannot build index.")

        print("Building search index (this may take a minute on first run)...")

        embeddings = self.embedding_model.encode(
            self.documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance metric
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Index built with {self.index.ntotal} chunks")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Search for the most relevant document chunks using semantic similarity.
        
        Args:
            query: The search question
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'content', 'document', and 'score' keys
        """
        # Convert query to embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'content': self.documents[idx],
                'document': self.doc_names[idx],
                'score': float(1 / (1 + distance))  # Convert distance to similarity score
            })

    def _save_index_to_disk(self) -> None:
        """
        Save FAISS index + metadata so Streamlit Cloud does not have to rebuild.
        Run this once locally; commit the resulting files to your repo.
        """
        if self.index is None:
            raise ValueError("Index not built; cannot save.")

        print("Saving FAISS index and metadata to disk...")
        faiss.write_index(self.index, self.index_path)

        meta = {
            "documents": self.documents,
            "doc_names": self.doc_names,
        }
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"✓ Saved index to {self.index_path}")
        print(f"✓ Saved metadata to {self.meta_path}")

    def _load_index_from_disk(self) -> None:
        """
        Load FAISS index + metadata that were previously saved.
        """
        print("Loading FAISS index + metadata from disk...")
        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)

        self.documents = meta["documents"]
        self.doc_names = meta["doc_names"]

        print(f"✓ Loaded index with {self.index.ntotal} chunks")

    # ---------------------------------------------------------------------- #
    # Search + QA
    # ---------------------------------------------------------------------- #
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
            [query],
            convert_to_numpy=True,
        )

        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)

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
    
    def answer_question(self, question: str) -> Tuple[str, List[dict]]:
        """
        Answer a question using retrieved documents and GPT-4 (if available).
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (answer_text, list_of_sources)
        """
        # Find relevant chunks
        End-to-end QA: retrieve relevant chunks, then generate the answer.
        """
        sources = self.search_documents(question, top_k=3)
        
        # Generate answer using GPT-4 or basic summarization
        if self.use_gpt and self.openai_client:
            answer = self._generate_gpt_answer(question, sources)
        else:
            answer = self._generate_basic_answer(question, sources)
        
        return answer, sources
    
    def _generate_gpt_answer(self, question: str, sources: List[dict]) -> str:
        """
        Generate natural language answer using GPT-4.
        
        This creates a professional, technical answer that synthesizes
        information from multiple sources.
        """
        # Prepare context from retrieved sources
        context = ""
        for i, source in enumerate(sources, 1):
            context += f"\n\n[Source {i} - {source['document']}]:\n{source['content']}"
        
        # Create prompt for GPT-4
        prompt = f"""You are an expert RF engineer helping answer technical questions about wireless antenna design and 5G systems.

    # ---------------------------------------------------------------------- #
    # Answer generation
    # ---------------------------------------------------------------------- #
    def _generate_gpt_answer(self, question: str, sources: List[dict]) -> str:
        """
        Generate a high-quality RF engineering answer using GPT-4.
        """
        context = ""
        for i, source in enumerate(sources, 1):
            context += f"\n\n[Source {i} – {source['document']}]\n{source['content']}"

        prompt = f"""You are a senior RF engineer at a top smartphone company.

You are helping an engineer understand a question about RF/antenna design and 5G systems.

Question:
{question}

Technical excerpts from patents & papers:
{context}

Instructions:
1. Provide a direct, technical answer using information from the sources
2. Cite which source(s) you're using by mentioning the document name
3. If the sources mention specific technical parameters (frequencies, distances, materials), include them
4. Keep the answer focused and relevant to RF/antenna engineering
5. Structure your answer with clear paragraphs
6. If the sources don't fully answer the question, acknowledge what information is available
Write a clear, technically correct explanation that a strong RF engineer would respect.

Guidelines:
- Start with a 2–3 sentence overview in plain language.
- Then give 3–6 numbered technical points with specifics (frequencies, materials, array spacing, SAR constraints, etc.) pulled from the text when available.
- Emphasize trade-offs and design implications for real RF hardware (smartphones, basestations, mmWave arrays, etc.).
- Where useful, mention which source (e.g. "Apple mmWave patent") a detail comes from.
- Avoid just copying sentences; synthesize and clean them up.
>>>>>>> b45c697 (Add prebuilt RF FAISS index and metadata)

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-efficient model, can upgrade to gpt-4o for better quality
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert RF engineer assistant helping with antenna design and wireless systems questions. Provide technical, accurate answers."

                        "role": "system",
                        "content": (
                            "You are an expert RF engineer. "
                            "Be precise, structured, and practical."
                        ),
                        
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],

                temperature=0.7,  # Balanced creativity and accuracy
                max_tokens=800     # Sufficient for detailed technical answers
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠ Error calling GPT-4: {e}")
            print("  Falling back to basic summarization...")

                temperature=0.4,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ Error calling GPT-4: {e}")
            return self._generate_basic_answer(question, sources)
    
    def _generate_basic_answer(self, question: str, sources: List[dict]) -> str:
        """
        Fallback: Generate structured answer without GPT-4.
        
        This provides a readable response by organizing the retrieved content,
        though less polished than GPT-4 generated answers.
        """
        answer = f"**Based on the retrieved technical documents:**\n\n"
        
        # Summarize key findings from each source
        for i, source in enumerate(sources, 1):
            # Extract first ~400 characters as summary
            excerpt = source['content'][:400].strip()
            
            # Try to end at a sentence for better readability
            if len(source['content']) > 400:
                last_period = excerpt.rfind('.')
                if last_period > 200:
                    excerpt = excerpt[:last_period + 1]
                else:
                    excerpt += "..."
            
            answer += f"**{i}. From {source['document']}** (Relevance: {source['score']:.0%}):\n\n"
            answer += f"{excerpt}\n\n"
        
        answer += f"""---

**Summary:**  
The above excerpts from Apple RF patents and 5G research papers provide technical context related to: *"{question}"*

The information has been retrieved using semantic similarity search across {len(self.documents)} document chunks. The relevance scores indicate how closely each excerpt matches your question.

* Note: For more natural language answers with synthesis across sources, configure an OpenAI API key. Current mode: Direct retrieval.*"""
        
        return answer

# Example usage for testing
if __name__ == "__main__":
    """
    Test the RF Assistant with a sample question.
    Run this file directly to test: python rag_engine.py
    """
    print("\n" + "="*80)
    print("RF ASSISTANT TEST")
    print("="*80 + "\n")
    
    assistant = RFAssistant()
    assistant.load_documents()
    
    question = "What are the challenges in 5G mmWave antenna design?"
    print(f"\nTest Question: {question}\n")
    
    answer, sources = assistant.answer_question(question)
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['document']}")
        print(f"   Relevance: {source['score']:.2f}")
        print(f"   Preview: {source['content'][:200]}...")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

        Fallback if GPT is unavailable.

        Produces a structured, readable answer summarizing the top chunks
        instead of just dumping raw text.
        """
        if not sources:
            return (
                "I couldn't find relevant content in the indexed RF documents for this "
                "question. Try rephrasing it or narrowing the scope."
            )

        # Extract a short, cleaned excerpt from each source
        def clean_excerpt(text: str, max_chars: int = 450) -> str:
            t = " ".join(text.split())  # collapse whitespace
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
            bullet_points.append(f"- From **{src['document']}** (~{src['score']*100:.0f}% match): {excerpt}")
            overview_points.append(excerpt)

        # Build the answer
        answer_parts: List[str] = []

        answer_parts.append("### Explanation\n")
        answer_parts.append(
            f"Here’s a synthesized explanation of **{question}** based on the retrieved RF patents and papers:\n"
        )

        # Very short overview paragraph
        if overview_points:
            answer_parts.append(
                "\n".join(
                    [
                        "**High-level view:**",
                        overview_points[0],
                        "",
                    ]
                )
            )

        # Bulleted evidence
        answer_parts.append("**Key technical evidence from the documents:**\n")
        answer_parts.extend(bullet_points)
        answer_parts.append("")

        # Wrap-up
        answer_parts.append(
            "> These points are extracted directly from the indexed RF documents. "
            "For a more polished narrative answer, enable GPT-4 via the OpenAI API key."
        )

        return "\n".join(answer_parts)

if __name__ == "__main__":
    # Run once locally to build and cache the FAISS index + metadata
    print("=" * 80)
    print("Building RF FAISS index from local PDFs...")
    print("=" * 80)

    assistant = RFAssistant(documents_folder="documents", use_gpt=False)
    assistant.load_documents()

    print("\nDone.")
    print(f"Index file: {assistant.index_path}")
    print(f"Meta file:  {assistant.meta_path}")
    print("=" * 80)
