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
  [azure/ardania-azure.py](https://github.com/echo85/ardania-RAG/blob/main/azure/ardania-azure.py)

---

### 🧪 Open Source Implementation  
Built with **Ollama**, **LangChain**, and **Chroma** – ideal for local development and experimentation.

- **Vector Store Generation Script**:  
  [generate_vector_store_fromMarkdown.py](https://github.com/echo85/ardania-RAG/blob/main/ollama/generate_vectore_store_fromMarkdown.py)

- **RAG Agent**:  
  [ardania-ollama-chroma-langgraph.py](https://github.com/echo85/ardania-RAG/blob/main/ollama/ardania-ollama-chroma-langgraph.py)
