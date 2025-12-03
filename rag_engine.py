import os
from typing import List, Tuple
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
    """
    
    def __init__(self, documents_folder: str = "documents", use_gpt: bool = True):
        self.documents_folder = documents_folder
        self.documents = []  # Stores all text chunks
        self.doc_names = []  # Stores source document names
        self.use_gpt = use_gpt
        
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        
        # Initialize OpenAI client if GPT is enabled
        if self.use_gpt:
            try:
                from openai import OpenAI
                # Try to get API key from environment variable
                api_key = os.getenv('OPENAI_API_KEY')
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

Based on the following technical documents, provide a clear, detailed answer to the engineer's question. Use specific technical details from the sources, but explain them clearly.

Question: {question}

Technical Documents:
{context}

Instructions:
1. Provide a direct, technical answer using information from the sources
2. Cite which source(s) you're using by mentioning the document name
3. If the sources mention specific technical parameters (frequencies, distances, materials), include them
4. Keep the answer focused and relevant to RF/antenna engineering
5. Structure your answer with clear paragraphs
6. If the sources don't fully answer the question, acknowledge what information is available

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-efficient model, can upgrade to gpt-4o for better quality
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert RF engineer assistant helping with antenna design and wireless systems questions. Provide technical, accurate answers."
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


