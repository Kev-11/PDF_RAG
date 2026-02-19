import streamlit as st
import os, hashlib
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()
st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("📄 Chat with your PDF")

# ---------------------------
# SESSION MEMORY
# ---------------------------
if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None


if "indexed_hash" not in st.session_state:
    st.session_state.indexed_hash = None

# ---------------------------
# CACHED RESOURCES
# ---------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

@st.cache_resource
def get_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True,
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        # model="openai/gpt-oss-20b",
        model = "llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
        streaming=True,
    )

embeddings = get_embeddings()
client = get_client()
llm = get_llm()

# ---------------------------
# HELPERS
# ---------------------------
def file_hash(data: bytes):
    return hashlib.md5(data).hexdigest()

def collection_name_from_hash(hash_value):
    return f"pdf_{hash_value[:12]}"

# ---------------------------
# UPLOAD PDF
# ---------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    file_bytes = uploaded_file.read()
    current_hash = file_hash(file_bytes)
    COLLECTION_NAME = collection_name_from_hash(current_hash)
    
    previous_pdf = st.session_state.get("active_pdf")
    # switch active PDF
    st.session_state.active_pdf = current_hash

    # ensure memory exists for this pdf
    if current_hash not in st.session_state.chat_store:
        st.session_state.chat_store[current_hash] = []
        
    if previous_pdf != current_hash:
        st.toast(f"Restored chat for {uploaded_file.name}")
    else:
        st.toast(f"Ready to chat with {uploaded_file.name}")    


    st.write(f"Collection: `{COLLECTION_NAME}`")

    # ---------- INDEX BUTTON ----------
    if st.button("📚 Index PDF"):

        if st.session_state.indexed_hash == current_hash:
            st.success("PDF already indexed in this session")
        else:
            with st.spinner("Indexing PDF..."):

                # save temp file
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    file_path = tmp.name

                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                
                text_len = len("\n\n".join(d.page_content for d in docs))
                
                def get_chunk_param(text_len):
                    if text_len < 500:
                        return 200, 30
                    elif text_len < 1000:
                        return 400, 40
                    elif text_len < 5000:
                        return 700, 100
                    else:
                        return 900, 150
                
                chunk_size, chunk_overlap = get_chunk_param(text_len)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                chunks = splitter.split_documents(docs)

                # metadata
                page_chunks_counter = defaultdict(int)
                for chunk in chunks:
                    page = chunk.metadata.get("page", 0)
                    page_chunks_counter[page] += 1
                    chunk.metadata = {
                        "source": uploaded_file.name,
                        "page": page,
                        "chunk_id": f"p{page+1}_c{page_chunks_counter[page]}"
                    }

                collections = [c.name for c in client.get_collections().collections]

                if COLLECTION_NAME not in collections:
                    QdrantVectorStore.from_documents(
                        chunks,
                        embeddings,
                        url=os.getenv("QDRANT_URL"),
                        api_key=os.getenv("QDRANT_API_KEY"),
                        collection_name=COLLECTION_NAME,
                        prefer_grpc=True,
                    )

                st.session_state.indexed_hash = current_hash
                st.success("Indexing complete!")

    # ---------- LOAD VECTOR STORE ----------
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:

        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=COLLECTION_NAME,
            prefer_grpc=True,
        )

        retriever = qdrant.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 5},
        )

        # ---------------------------
        # SHOW CHAT HISTORY
        # ---------------------------
        messages = st.session_state.chat_store[st.session_state.active_pdf]

        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ---------------------------
        # CHAT INPUT
        # ---------------------------
        if prompt := st.chat_input("Ask about the PDF"):

            messages.append({"role":"user","content":prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            docs = retriever.invoke(prompt)

            context = "\n\n".join(
                f"[SOURCE:{d.metadata.get('source')} PAGE:{d.metadata.get('page')}]\n{d.page_content}"
                for d in docs
            )

            # include previous chat for memory
            history_text = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in messages[-6:]   # last few turns
            )

            final_prompt = f"""
Use ONLY the context to answer.

CHAT HISTORY:
{history_text}

CONTEXT:
{context}

QUESTION:
{prompt}
"""

            with st.chat_message("assistant"):
                box = st.empty()
                output = ""

                for chunk in llm.stream(final_prompt):
                    if chunk.content:
                        output += chunk.content
                        box.markdown(output)

                messages.append({"role":"assistant","content":output})

            # citations
            grouped = defaultdict(set)
            for d in docs:
                grouped[d.metadata["source"]].add(d.metadata["page"])

            st.markdown("**Sources:**")
            for src,pages in grouped.items():
                st.write(f"{src} → pages {', '.join(map(str,sorted(pages)))}")

    else:
        st.info("Click **Index PDF** to begin chatting.")
