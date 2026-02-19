# PDF RAG Chat Application

A Streamlit-based RAG (Retrieval Augmented Generation) application that lets you chat with your PDF documents using LangChain, Qdrant, and Groq.

## Features

- 📄 Upload and index PDF documents
- 💬 Chat with your PDFs using natural language
- 🧠 Powered by Groq's LLM and Qdrant vector database
- 🔄 Session-based chat history for each PDF

## Prerequisites

Before deploying, you'll need:

1. **Qdrant Account** - Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/)
   - Create a cluster
   - Get your `QDRANT_URL` and `QDRANT_API_KEY`

2. **Groq API Key** - Get from [console.groq.com](https://console.groq.com/)
   - Sign up/login
   - Generate an API key

## Deploy to Streamlit Cloud

### Step 1: Push to GitHub
Your code is already on GitHub at: `https://github.com/Kev-11/PDF_RAG`

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - **Repository:** `Kev-11/PDF_RAG`
   - **Branch:** `main`
   - **Main file path:** `app.py`

5. Click "Advanced settings" and add your secrets:
   ```toml
   QDRANT_URL = "your_qdrant_cluster_url"
   QDRANT_API_KEY = "your_qdrant_api_key"
   GROQ_API_KEY = "your_groq_api_key"
   ```

6. Click "Deploy!"

### Step 3: Wait for Deployment
The first deployment takes 5-10 minutes as it installs all dependencies.

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/Kev-11/PDF_RAG.git
cd PDF_RAG
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys:
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

5. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Upload a PDF file
2. Click "📚 Index PDF" to process and vectorize the document
3. Ask questions in the chat interface
4. The app will retrieve relevant context and generate answers

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (llama-3.3-70b-versatile)
- **Vector Database:** Qdrant
- **Embeddings:** HuggingFace (all-MiniLM-l6-v2)
- **PDF Processing:** PyMuPDF
- **Framework:** LangChain

## License

MIT
