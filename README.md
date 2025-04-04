# 📖 Streamlit-Based Local Document Q&A System

This project allows users to upload a PDF book and interactively ask questions about its contents using **LangChain** and **Ollama (Llama 3.2)**.

---

## 🚀 Features
- 📂 **Upload PDFs** and extract information from them.
- 💬 **Ask questions** interactively.
- 🤖 **Uses Ollama (Llama 3.2)** as the LLM.
- 📚 **Stores chat history** for reference.
- 🎨 **Streamlit-powered UI** for a seamless experience.
- 🔍 **Detects farewell messages** to end the chat gracefully.

---

## 🛠️ Installation Guide

### 1️⃣ Clone the Repository
```bash
# Replace with your GitHub repository link
git clone https://github.com/krakken190/Local-Rag-Using-Olama-and-Langchain.git
cd local-document-qa
```

### 2️⃣ Set Up a Virtual Environment
```bash
# On Windows
python -m venv myenv
myenv\Scripts\activate

# On macOS/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 3️⃣ Install Ollama
[Ollama](https://ollama.com/) is required to run the Llama model locally.

#### 🔹 Windows/macOS/Linux
- Download and install Ollama from: https://ollama.com/download

#### 🔹 Verify Ollama Installation
```bash
ollama --version
```

### 4️⃣ Download the Llama Model
Choose the model based on your system RAM:

| RAM Available | Model to Install |
|--------------|----------------|
| 8GB+        | `ollama pull llama3` |
| 16GB+       | `ollama pull llama3:8b` |
| 32GB+       | `ollama pull llama3:70b` |

Example command:
```bash
ollama pull llama3
```

### 5️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 6️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📦 `requirements.txt` (Updated)
```txt
streamlit
langchain
langchain_community
langchain_ollama
pypdf
faiss-cpu
ollama
```

---

## 📝 How to Push the Project to GitHub

### 1️⃣ Initialize Git and Add Remote Repository
```bash
git init
git remote add origin https://github.com/krakken190/Local-Rag-Using-Olama-and-Langchain.git
git branch -M main
```

### 2️⃣ Add and Commit Changes
```bash
git add .
git commit -m "Initial commit - Streamlit Document Q&A"
```

### 3️⃣ Push to GitHub
```bash
git push -u origin main
```

---

## 🎯 Usage Instructions
1. Upload a PDF document.
2. Ask questions related to the content.
3. Receive AI-generated responses.
4. Say "Thank you" or similar phrases to exit gracefully.

---

## 📜 License
This project is **MIT Licensed**. Feel free to modify and use it.

---

## 📧 Contact
For any issues, feel free to open an issue on GitHub or reach out via email.

Happy coding! 🚀

