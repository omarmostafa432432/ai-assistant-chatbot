import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Page config
st.set_page_config(
    page_title="Ask Career AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-image {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with knowledge_base.txt"""
    
    # Check for API key in secrets or environment
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    
    if not groq_api_key:
        return None, "❌ GROQ_API_KEY not found. Please add it to Streamlit Secrets:\n1. Go to app settings\n2. Click 'Secrets'\n3. Add: GROQ_API_KEY = \"your-key-here\""
    
    # Check if knowledge_base.txt exists
    if not os.path.exists("knowledge_base.txt"):
        return None, "❌ knowledge_base.txt not found. Please add it to your GitHub repository."
    
    try:
        # Load knowledge base
        loader = TextLoader("knowledge_base.txt", encoding='utf-8')
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create LLM
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key
        )
        
        # Create prompt template
        template = """You are a helpful AI assistant that answers questions about a person's professional background based on their CV and certifications.

Use the following context to answer the question. Be specific and include relevant details like skills, dates, companies, certifications, and projects.

If you cannot find the answer in the context, say "I don't have that information in the documents provided."

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain using LCEL
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, None
        
    except Exception as e:
        return None, f"❌ Error initializing: {str(e)}"

def main():
    # Display header image
    if os.path.exists("header.jpg") or os.path.exists("header.png"):
        header_file = "header.jpg" if os.path.exists("header.jpg") else "header.png"
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.image(header_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Fallback text header if image not found
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #1f77b4;">🤖 Ask Career AI</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666;">Ask me anything about my background, skills, and experience!</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize system on first load
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing AI Assistant..."):
            rag_chain, error = initialize_rag_system()
            
            if error:
                st.session_state.error_message = error
                st.session_state.initialized = True
            else:
                st.session_state.rag_chain = rag_chain
                st.session_state.initialized = True
    
    # Show error if initialization failed
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.info("""
        **Setup Instructions:**
        
        1. Add your Groq API key to Streamlit Secrets:
           - Go to app settings → Secrets
           - Add: `GROQ_API_KEY = "your-key-here"`
        
        2. Add `knowledge_base.txt` to your GitHub repository
        
        3. Reboot the app
        """)
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header("💡 Sample Questions")
        
        sample_questions = [
            "What are my main technical skills?",
            "Tell me about my work experience",
            "What certifications do I have?",
            "Describe my educational background",
            "What projects have I worked on?",
            "What programming languages do I know?",
            "Summarize my professional background"
        ]
        
        for q in sample_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        
        st.markdown("---")
        
        # Clear conversation button
        if len(st.session_state.messages) > 0:
            if st.button("🔄 Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        ### 🛠️ Built With
        - **RAG** (Retrieval Augmented Generation)
        - **LangChain** & FAISS
        - **Groq** (Llama 3.3 70B)
        - **Streamlit**
        
        ---
        
        ### ℹ️ About
        This AI assistant can answer questions about professional background, skills, experience, certifications, and more!
        """)
    
    # Welcome message if no chat history
    if len(st.session_state.messages) == 0:
        st.info("👋 **Welcome!** I'm an AI assistant trained on professional background information. Ask me anything about skills, experience, certifications, or projects!")
        
        # Show some example questions in the main area
        st.markdown("### 💬 Example Questions:")
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("""
            - What are my main skills?
            - Tell me about my work experience
            - What certifications do I have?
            - Describe my education
            """)
        
        with cols[1]:
            st.markdown("""
            - What projects have I worked on?
            - What programming languages do I know?
            - Summarize my background
            - What is my latest position?
            """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG chain
                    answer = st.session_state.rag_chain.invoke(prompt)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
