# 📡 RF Engineering AI Assistant

An intelligent AI assistant that searches through Apple RF patents and 5G research papers to provide instant, cited answers to technical questions.

**Tech Stack:** Streamlit + GPT-4 + FAISS + Sentence Transformers


## ✨ Features

- 🔍 **Semantic Search**: Searches 81+ document chunks using FAISS vector database
- 🤖 **GPT-4 Integration**: Natural language answers with technical depth
- 📚 **Source Citations**: Every answer includes relevance scores and original excerpts
- ⚡ **Lightning Fast**: ~10 second response time vs 45+ minutes manual search
- 📊 **Analytics Dashboard**: Real-time metrics and query history
- 💾 **Export Results**: Download query results as text files
- 🎨 **Unique UI**: Beautiful, professional interface

---

## 🚀 Quick Start

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


## 🎯 Usage Examples

### Example Questions:
- "What are common causes of antenna interference in multi-band systems?"
- "How does beamforming improve 5G performance?"
- "What are the key challenges in mmWave antenna design?"
- "What materials are best for 5G antenna substrates?"

### What You Get:
1. **Natural Language Answer** (if GPT-4 enabled)
2. **Source Citations** with relevance scores
3. **Original Excerpts** from patents/papers
4. **Search Metrics** (time, accuracy, coverage)


## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Documents Indexed** | 5 PDFs (Apple patents + 5G papers) |
| **Searchable Chunks** | 81 text segments |
| **Search Time** | ~2 seconds average |
| **Relevance Accuracy** | 85%+ for technical queries |
| **Cost per Query** | ~$0.01-0.02 (with GPT-4) |


## 🌐 Deployment to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/rf-ai-assistant.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Add secrets in Advanced settings:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
6. Click "Deploy"!

Your app will be live at: `https://rf-assistant-yourusername.streamlit.app`


## 📈 Future Enhancements

- [ ] Add more document types (XLSX, DOCX)
- [ ] Multi-language support
- [ ] Compare multiple documents side-by-side
- [ ] Integration with Apple's internal databases
- [ ] Batch query processing
- [ ] Email alerts for new relevant papers


## 🙏 Acknowledgments

- Apple RF Patents and Research Papers(publicly available via USPTO)


**⭐ Star this repo if you found it useful!**
