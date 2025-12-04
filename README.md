# RF Engineering AI Assistant

An intelligent AI assistant that searches through Apple RF patents and 5G research papers to provide instant, cited answers to technical questions.

**Tech Stack:** Streamlit + GPT-4 + FAISS + Sentence Transformers

Check Demo here: https://rfchatbot.streamlit.app


## Features

-  **Semantic Search**: Searches 81+ document chunks using FAISS vector database
-  **GPT-4 Integration**: Natural language answers with technical depth
-  **Source Citations**: Every answer includes relevance scores and original excerpts
-  **Lightning Fast**: ~10 second response time vs 45+ minutes manual search
-  **Export Results**: Download query results as text files


##  Usage 

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Get your key from: https://platform.openai.com/api-keys

### 3. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

##  Future Enhancements

- [ ] Add more document types (XLSX, DOCX)
- [ ] Multi-language support
- [ ] Compare multiple documents side-by-side
- [ ] Integration with Apple's internal databases
- [ ] Batch query processing
- [ ] Email alerts for new relevant papers


**⭐ Star this repo if you found it useful!**
