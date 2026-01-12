import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile

# Page config
st.set_page_config(
    page_title="My AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def process_files(uploaded_files, groq_api_key):
    """Process uploaded files (PDF and TXT) and create RAG chain"""
    
    with st.spinner("🔄 Processing your documents..."):
        documents = []
        
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if file_extension == 'pdf':
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    
                elif file_extension == 'txt':
                    loader = TextLoader(tmp_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        # Show extraction stats
        total_chars = sum(len(doc.page_content) for doc in documents)
        st.info(f"📊 Extracted {total_chars:,} characters from {len(documents)} document(s)")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        st.info(f"✂️ Created {len(splits)} chunks for processing")
        
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

Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have that information in the documents provided."

Context:
{context}

Question: {question}

Answer: Provide a detailed, professional response based only on the context above."""

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
        
        return vector_store, rag_chain

def main():
    # Header
    st.markdown('<p class="main-header">🤖 My AI Personal Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about my background, skills, and experience!</p>', unsafe_allow_html=True)
    
    # Sidebar for setup
    with st.sidebar:
        st.header("⚙️ Setup")
        
        # API Key input
        groq_api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input(
            "Groq API Key (optional if configured in secrets)",
            type="password",
            help="Get your free API key from https://console.groq.com"
        )
        
        st.markdown("---")
        
        # File upload
        st.subheader("📄 Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Upload CV and Certifications (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload your CV and certification files (PDF or plain text)"
        )
        
        # Process button
        if st.button("🚀 Initialize Assistant", type="primary"):
            if not groq_api_key:
                st.error("⚠️ Please enter your Groq API key first!")
            elif not uploaded_files:
                st.error("⚠️ Please upload at least one file!")
            else:
                try:
                    vector_store, rag_chain = process_files(uploaded_files, groq_api_key)
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = rag_chain
                    st.success("✅ Assistant initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        
        # Info section
        st.subheader("ℹ️ About")
        st.info("""
        This RAG-powered chatbot can answer questions about:
        - Work experience
        - Skills & certifications
        - Education
        - Projects
        - And more!
        
        💡 **Tip**: TXT files often work better than PDFs with complex formatting!
        """)
        
        # Sample questions
        if st.session_state.rag_chain:
            st.subheader("💡 Try asking:")
            sample_questions = [
                "What are my main skills?",
                "Tell me about my work experience",
                "What certifications do I have?",
                "Summarize my background"
            ]
            for q in sample_questions:
                if st.button(q, key=q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
            
            st.markdown("---")
            
            # Clear button
            if st.button("🔄 Clear Conversation"):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if st.session_state.rag_chain is None:
        st.info("👈 Please set up your API key and upload your files in the sidebar to get started!")
        
        # Show demo info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔑 Step 1")
            st.write("Get free Groq API key")
        with col2:
            st.markdown("### 📤 Step 2")
            st.write("Upload PDF or TXT files")
        with col3:
            st.markdown("### 💬 Step 3")
            st.write("Start chatting!")
            
        st.markdown("---")
        st.markdown("### 📝 Recommended: Use TXT Format")
        st.write("For best results, create a `knowledge_base.txt` file with your information in a structured format.")
        
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about my profile..."):
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
