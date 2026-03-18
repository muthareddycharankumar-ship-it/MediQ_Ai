# 🏥 MediQ_AI

An intelligent AI-powered medical assistant that provides medicine information and doctor recommendations to the general public — powered by a local LLM for privacy-first healthcare guidance.

> ⚠️ **Disclaimer:** MediQ_AI is for informational purposes only. Always consult a qualified medical professional for medical advice, diagnosis, or treatment.

---

## 🚀 Features

- 💊 Detailed medicine information and usage guidance
- 👨‍⚕️ Doctor specialty recommendations based on symptoms
- 🔒 Fully local AI — your data never leaves your device
- 🧠 Powered by Mistral 7B running via Ollama
- 📚 ChromaDB vector database for fast and accurate retrieval
- ⚡ Flask-based backend with a clean frontend interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python / Flask | Backend web framework |
| Mistral 7B (GGUF) | Local AI language model |
| Ollama | Local LLM runtime |
| ChromaDB | Vector database for knowledge retrieval |
| HTML / CSS / JS | Frontend interface |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/muthareddycharankumar-ship-it/MediQ_Ai.git
cd MediQ_Ai
```

### 2. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull Mistral model
```bash
ollama pull mistral
```

### 4. Create a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Start Ollama
```bash
ollama serve
```

### 2. Run the application
```bash
python app.py
```

### 3. Open your browser
```
http://localhost:5000
```

Ask any medicine-related question or describe your symptoms to get doctor recommendations!

---

## 📁 Project Structure
```
MediQ_Ai/
├── app.py               # Main Flask application
├── backend/             # Backend logic & API routes
├── frontend/            # HTML, CSS, JS files
├── chroma_db/           # Vector database storage
├── requirements.txt     # Python dependencies
├── Modelfile            # Ollama model configuration
└── README.md            # Project documentation
```

---

## 📌 Requirements

- Python 3.8+
- Ollama installed and running
- Mistral 7B model (~4GB disk space)
- pip & virtual environment

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Mutharedy Charan Kumar**
- GitHub: [@muthareddycharankumar-ship-it](https://github.com/muthareddycharankumar-ship-it)
