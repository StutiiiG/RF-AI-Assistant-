import os
from typing import List, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RFAssistant:
    """
    Enhanced RAG system with GPT-4 integration for natural language answers.
    """
    
    def __init__(self, documents_folder: str = "documents", use_gpt: bool = True):
        self.documents_folder = documents_folder
        self.documents = []
        self.doc_names = []
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
                    self.openai_client = OpenAI(api_key=api_key)
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
        """Read all PDFs from the documents folder"""
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
        """Split text into smaller pieces with overlap for better context"""
        words = text.split()
        chunks = []
        overlap = 50  # words overlap between chunks
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def _build_index(self):
        """Build a search index from all document chunks"""
        print("Building search index (this may take a minute)...")
        
        embeddings = self.embedding_model.encode(
            self.documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Index built with {self.index.ntotal} chunks")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[dict]:
        """Search for the most relevant document chunks"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'content': self.documents[idx],
                'document': self.doc_names[idx],
                'score': float(1 / (1 + distance))
            })
        
        return results
    
    def answer_question(self, question: str) -> Tuple[str, List[dict]]:
        """Answer a question using retrieved documents and GPT-4 (if available)"""
        # Find relevant chunks
        sources = self.search_documents(question, top_k=3)
        
        # Generate answer
        if self.use_gpt and self.openai_client:
            answer = self._generate_gpt_answer(question, sources)
        else:
            answer = self._generate_basic_answer(question, sources)
        
        return answer, sources
    
    def _generate_gpt_answer(self, question: str, sources: List[dict]) -> str:
        """Generate natural language answer using GPT-4"""
        # Prepare context from sources
        context = ""
        for i, source in enumerate(sources, 1):
            context += f"\n\n[Source {i} - {source['document']}]:\n{source['content']}"
        
        # Create prompt for GPT
        prompt = f"""You are an expert RF engineer helping answer technical questions about wireless antenna design and 5G systems.

Based on the following technical documents, provide a clear, detailed answer to the engineer's question. Use specific technical details from the sources, but explain them clearly.

Question: {question}

Technical Documents:
{context}

Instructions:
1. Provide a direct, technical answer using information from the sources
2. Cite which source(s) you're using by mentioning the document name
3. If the sources mention specific technical parameters, include them
4. Keep the answer focused and relevant to RF/antenna engineering
5. If the sources don't fully answer the question, acknowledge what information is available

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using mini for cost efficiency
                messages=[
                    {"role": "system", "content": "You are an expert RF engineer assistant helping with antenna design and wireless systems questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling GPT-4: {e}")
            return self._generate_basic_answer(question, sources)
    
    def _generate_basic_answer(self, question: str, sources: List[dict]) -> str:
        """Fallback: Generate structured answer without GPT"""
        answer = f"**Based on the retrieved technical documents:**\n\n"
        
        # Summarize key findings from each source
        for i, source in enumerate(sources, 1):
            # Extract first ~300 characters as summary
            excerpt = source['content'][:400].strip()
            if len(source['content']) > 400:
                # Try to end at a sentence
                last_period = excerpt.rfind('.')
                if last_period > 200:
                    excerpt = excerpt[:last_period + 1]
                else:
                    excerpt += "..."
            
            answer += f"**From {source['document']}** (Relevance: {source['score']:.0%}):\n"
            answer += f"{excerpt}\n\n"
        
        answer += f"""---

**Summary:**  
The above excerpts from Apple RF patents and 5G research papers provide technical context related to: *"{question}"*

The most relevant information has been retrieved using semantic similarity search across 81 document chunks.

*Note: For more natural language answers, configure OpenAI API key. Current mode: Basic retrieval.*"""
        
        return answer



