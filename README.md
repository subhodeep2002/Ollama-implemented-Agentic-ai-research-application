# 🔬 AI Research Assistant

A local AI-powered research assistant built with **LangChain**, **LangGraph**, and **Ollama** — runs entirely on your machine, no API keys needed.

![Chat Interface](screenshots/chat.png)

## What it does

Ask any research question and the assistant will search the web, Wikipedia, and ArXiv to give you a well-sourced answer. It reasons step-by-step using a ReAct agent before responding.

**Three modes:**
- **💬 Chat** — General research Q&A with conversation memory
- **🔍 Deep Research** — Uses web, Wikipedia, and ArXiv tools to research in depth
- **📝 Report** — Generates a full structured report with sections (Background, Findings, Outlook, etc.)

## Stack

| Layer | Tech |
|-------|------|
| LLM | [Ollama](https://ollama.com) (llama3.2, mistral, gemma2, phi3) |
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) ReAct agent |
| Tools | DuckDuckGo · Wikipedia · ArXiv |
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangChain](https://langchain.com) |

---

## Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### Install

```bash
git clone https://github.com/YOUR_USERNAME/Ollama-implemented-Agentic-ai-research-application

cd Ollama-implemented-Agentic-ai-research-application


pip install streamlit langchain langgraph langchain-ollama \
            langchain-community langchain-core langchain-text-splitters \
            duckduckgo-search wikipedia arxiv
```

### Pull a model

```bash
ollama pull llama3.2
```

### Run

```bash
# 1. Start Ollama
ollama serve

# 2. Launch the app
streamlit run app/research_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Mac users:** Double-click `app/ResearchAssistant.app` — it starts everything automatically.
---

## Features

- 🔒 **Fully local** — your data never leaves your machine
- 🧠 **Persistent chat history** within a session
- 🔧 **Configurable** — swap models, adjust temperature from the sidebar
- 📄 **Structured reports** on any topic on demand

---
---

## Screenshots
<img width="1470" height="834" alt="Screenshot 2026-03-22 at 10 50 31 PM" src="https://github.com/user-attachments/assets/48ec31a8-5ecc-4a05-af55-aca2bcd40dc2" />
<img width="1470" height="825" alt="Screenshot 2026-03-22 at 10 50 38 PM" src="https://github.com/user-attachments/assets/b797b3e1-4f8e-4e5c-b42f-8ae723f6bf8b" />
<img width="1469" height="824" alt="Screenshot 2026-03-22 at 10 50 46 PM" src="https://github.com/user-attachments/assets/9d8842ec-02a6-4f09-87f2-87e5a85a741d" />
---

