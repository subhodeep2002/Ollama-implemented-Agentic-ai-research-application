import streamlit as st
import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Claude-like look) ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Dark background */
    .stApp { background-color: #1a1a2e; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #16213e;
        border-right: 1px solid #2a2a4a;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #2a2a4a !important;
        color: #e0e0e0 !important;
        border: 1px solid #4a4a7a !important;
        border-radius: 12px !important;
    }

    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #1e1e3a;
        border-radius: 12px;
        padding: 4px;
    }

    /* Assistant message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #16213e;
        border-radius: 12px;
        padding: 4px;
    }

    p, li, span { color: #e0e0e0 !important; }
    h1, h2, h3 { color: #a78bfa !important; }

    /* Sidebar text */
    .stSelectbox label, .stRadio label, .stSlider label { color: #a0a0c0 !important; }

    /* Buttons */
    .stButton > button {
        background-color: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:hover { background-color: #6366f1; }

    /* Title area */
    .title-area {
        text-align: center;
        padding: 20px 0 10px 0;
        border-bottom: 1px solid #2a2a4a;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load heavy dependencies (cached) ────────────────────────────────────
@st.cache_resource
def load_llm(model_name: str, base_url: str, temperature: float):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature, num_predict=4096)


@st.cache_resource
def load_tools(_llm):
    from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.document_loaders import ArxivLoader
    from langchain.tools import tool
    from langchain_core.messages import HumanMessage

    web_search = DuckDuckGoSearchRun(
        name="web_search",
        description="Search the web for current info, news, and research.",
    )
    wikipedia = WikipediaQueryRun(
        name="wikipedia",
        description="Search Wikipedia for encyclopedic and established knowledge.",
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000),
    )

    @tool
    def search_arxiv(query: str) -> str:
        """Search ArXiv for academic papers (ML, physics, math, CS)."""
        try:
            docs = ArxivLoader(query=query, load_max_docs=3).load()
            if not docs:
                return "No papers found."
            return "\n---\n".join(
                f"Title: {d.metadata.get('Title')}\nAuthors: {d.metadata.get('Authors')}\n"
                f"Published: {d.metadata.get('Published')}\nAbstract: {d.page_content[:500]}..."
                for d in docs
            )
        except Exception as e:
            return f"ArXiv error: {e}"

    @tool
    def get_current_date(query: str = "") -> str:
        """Returns the current date and time."""
        return datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")

    @tool
    def summarize_text(text: str) -> str:
        """Summarize a long text into bullet points."""
        return _llm.invoke([HumanMessage(content=f"Summarize into 5-7 bullet points:\n\n{text}")]).content

    return [web_search, wikipedia, search_arxiv, get_current_date, summarize_text]


@st.cache_resource
def load_agent(_llm, _tools):
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(
        model=_llm,
        tools=_tools,
        prompt=(
            "You are an expert AI Research Assistant. "
            "Use tools to research topics thoroughly. "
            "For encyclopedic topics use wikipedia, for recent news use web_search, "
            "for academic topics use search_arxiv. Always cite your sources."
        ),
    )


@st.cache_resource
def load_memory(_llm, model_name: str, base_url: str):
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
    return {"embeddings": embeddings, "store": None}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown("---")

    model_name = st.selectbox("Model", ["llama3.2", "mistral", "gemma2", "phi3", "llama3.1"], index=0)
    base_url = st.text_input("Ollama URL", value="http://localhost:11434")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")
    mode = st.radio("Mode", ["💬 Chat", "🔍 Deep Research", "📝 Report"])

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("<small style='color:#6060a0'>Powered by LangChain + Ollama</small>", unsafe_allow_html=True)


# ── Init session state ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='title-area'><h2>🔬 AI Research Assistant</h2>"
    "<p style='color:#6060a0;font-size:14px'>Ask anything — I search the web, Wikipedia, and ArXiv for you</p></div>",
    unsafe_allow_html=True,
)


# ── Load models (show spinner on first load) ──────────────────────────────────
with st.spinner(f"Loading {model_name}..."):
    try:
        llm = load_llm(model_name, base_url, temperature)
        tools = load_tools(llm)
        agent = load_agent(llm, tools)
        memory = load_memory(llm, model_name, base_url)
    except Exception as e:
        st.error(f"Failed to connect to Ollama: {e}\n\nMake sure `ollama serve` is running.")
        st.stop()


# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Helper: run based on mode ─────────────────────────────────────────────────
def run_query(user_input: str) -> str:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    if mode == "💬 Chat":
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research assistant. Give accurate, evidence-backed answers."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm
        response = chain.invoke({
            "history": st.session_state.chat_history,
            "question": user_input,
        })
        answer = response.content
        st.session_state.chat_history += [
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ]
        return answer

    elif mode == "🔍 Deep Research":
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        return result["messages"][-1].content

    else:  # Report
        prompt = (
            f"Write a comprehensive research report on: '{user_input}'.\n"
            f"Sections: Executive Summary, Background, Key Findings, "
            f"Current State, Challenges, Future Outlook, References."
        )
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        return result["messages"][-1].content


# ── Chat input ────────────────────────────────────────────────────────────────
placeholder = {
    "💬 Chat": "Ask a research question...",
    "🔍 Deep Research": "Topic to research in depth...",
    "📝 Report": "Topic to generate a full report on...",
}

if prompt := st.chat_input(placeholder.get(mode, "Ask anything...")):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                answer = run_query(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
