import streamlit as st
import os
import faiss
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain  
from langchain.memory import ConversationBufferMemory  
from langchain_ollama import OllamaEmbeddings  
from langchain.docstore.in_memory import InMemoryDocstore 

# You can create the documents folder if it doesn't exist
os.makedirs("documents", exist_ok=True)

st.title("📖 Book Q&A with AI")

uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Save the uploaded file
    file_path = os.path.join("documents", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    progress_bar.progress(10)
    status_text.text("📂 File saved, loading PDF...")

    # Load and process the PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    progress_bar.progress(30)
    status_text.text("📄 Splitting PDF into chunks...")

    # Split text into smaller chunks for better retrieval
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    progress_bar.progress(50)
    status_text.text("🔍 Generating embeddings...")

    # Generate embeddings
    embeddings = OllamaEmbeddings(model="llama3.2")  
    doc_vectors = embeddings.embed_documents([doc.page_content for doc in texts])

    progress_bar.progress(70)
    status_text.text("📊 Creating FAISS index...")

    # Create FAISS index
    dimension = len(doc_vectors[0])  
    faiss_index = faiss.IndexHNSWFlat(dimension, 32) 
    faiss_index.add(np.array(doc_vectors, dtype=np.float32))

    progress_bar.progress(80)
    status_text.text("📚 Storing documents in FAISS...")

    # Use InMemoryDocstore correctly
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}

    for i, doc in enumerate(texts):
        doc_id = str(i)
        docstore.add({doc_id: doc})  
        index_to_docstore_id[i] = doc_id  

    progress_bar.progress(90)
    status_text.text("🤖 Initializing AI model...")

    # Initialize FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Setup LLM and conversational retrieval
    llm = Ollama(model="llama3.2")  
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

    progress_bar.progress(100)
    status_text.text("✅ Processing complete! Ready to chat.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input handling
if prompt := st.chat_input("Ask a question about the book:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Enhanced exit detection
    farewell_messages = [
        "thanks", "thank you", "bye", "exit", "quit", "goodbye", "see you", 
        "take care", "adios", "farewell", "end chat", "stop", "close", "finish", 
        "I'm done", "that's all", "enough", "no more questions", "I'll be back later"
    ]
    
    if prompt.lower().strip() in farewell_messages:
        answer = "You're welcome! Have a great day! 😊"
    else:
        with st.spinner("🤖 Generating answer..."):
            result = qa.invoke({"question": prompt}) 
            answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save Chat History
    if st.session_state.messages:
        with open("chat_history.txt", "w") as f:
            for message in st.session_state.messages:
                f.write(f"{message['role']}: {message['content']}\n")
