# Ardania RAG

**Ardania RAG** is a Retrieval-Augmented Generation (RAG) agent built around the lore and content of [Ardania](https://themiraclegdr.com/), a roleplay-focused MMORPG based on *Ultima Online*.

The knowledge base for this agent is sourced from the Markdown documentation available here:  
📁 https://github.com/jacklake-tm/ardania-md

## 🔧 Implementations

There is both Azure-based and open-source implementations of the agent:

---

### ☁️ Azure Implementation  
Utilizes **Azure AI Search** and **GPT-4o** for scalable, enterprise-grade performance.

- **Agent Code**:  
  [azure/agent.py](https://github.com/echo85/ardania-RAG/blob/main/azure/agent.py)

- **USAGE**:

  It's required to create an [AI Azure Search](https://learn.microsoft.com/en-us/azure/search/search-get-started-portal-import-vectors?tabs=sample-data-storage%2Cmodel-aoai%2Cconnect-data-storage) index called "ardaniamd-index"
  Index JSON: [azure/index.json](https://github.com/echo85/ardania-RAG/blob/main/azure/index.json)

   ```bash
  cd azure
  python -m .venv .venv
  source ./venv/bin/activate
  pip install -r requirements.txt`
  git clone https://jacklake-tm.github.io/ardania-md
  streamlit run app.py
  ```
---

### 🧪 Open Source Implementation  
Built with **Ollama**, **LangChain**, and **Chroma** – ideal for local development and experimentation.

- **Vector Store Generation Script**:  
  [generate_vector_store_fromMarkdown.py](https://github.com/echo85/ardania-RAG/blob/main/ollama/generate_vectore_store_fromMarkdown.py)

- **RAG Agent**:  
  [agent.py](https://github.com/echo85/ardania-RAG/blob/main/ollama/agent.py)

- **USAGE**:

  ```bash
  cd ollama
  python -m .venv .venv
  source ./venv/bin/activate
  pip install -r requirements.txt`
  git clone https://jacklake-tm.github.io/ardania-md
  python generate_vectore_store_fromMarkdown.py
  streamlit run app.py
  ```
