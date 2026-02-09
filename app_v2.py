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
import json
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Omar's AI Assistant",
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
    .lead-notification {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .unknown-question {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "leads" not in st.session_state:
    st.session_state.leads = []
if "unknown_questions" not in st.session_state:
    st.session_state.unknown_questions = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# Tool Functions
def send_pushover_notification(message):
    """Send notification via Pushover (optional)"""
    try:
        pushover_token = st.secrets.get("PUSHOVER_TOKEN", os.getenv("PUSHOVER_TOKEN"))
        pushover_user = st.secrets.get("PUSHOVER_USER", os.getenv("PUSHOVER_USER"))
        
        if pushover_token and pushover_user:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": pushover_token,
                    "user": pushover_user,
                    "message": message,
                }
            )
            return True
    except:
        pass
    return False

def record_contact_info(email, name="Not provided", message="Not provided"):
    """Record user contact information for follow-up"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lead = {
        "timestamp": timestamp,
        "email": email,
        "name": name,
        "message": message
    }
    
    # Store in session state
    st.session_state.leads.append(lead)
    
    # Send notification
    notification_text = f"🎯 NEW LEAD!\nName: {name}\nEmail: {email}\nMessage: {message}\nTime: {timestamp}"
    send_pushover_notification(notification_text)
    
    # Save to file for persistence
    try:
        with open("leads.json", "a") as f:
            f.write(json.dumps(lead) + "\n")
    except:
        pass
    
    return {"status": "success", "message": "Contact information recorded successfully!"}

def record_unknown_question(question):
    """Record questions that couldn't be answered"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unknown_q = {
        "timestamp": timestamp,
        "question": question
    }
    
    # Store in session state
    st.session_state.unknown_questions.append(unknown_q)
    
    # Send notification
    notification_text = f"❓ UNKNOWN QUESTION\nQ: {question}\nTime: {timestamp}"
    send_pushover_notification(notification_text)
    
    # Save to file
    try:
        with open("unknown_questions.json", "a") as f:
            f.write(json.dumps(unknown_q) + "\n")
    except:
        pass
    
    return {"status": "success", "message": "Question recorded for future improvement"}

# Tool definitions for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "record_contact_info",
            "description": "Use this when a user expresses interest in connecting, working together, or provides their contact information. This helps Omar follow up with potential opportunities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The user's email address"
                    },
                    "name": {
                        "type": "string",
                        "description": "The user's name if provided"
                    },
                    "message": {
                        "type": "string",
                        "description": "Any additional context about why they want to connect (e.g., job opportunity, collaboration, question)"
                    }
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Use this EVERY TIME you cannot answer a question because the information is not in the provided context. This helps improve the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The exact question that could not be answered"
                    }
                },
                "required": ["question"]
            }
        }
    }
]

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def handle_tool_calls(tool_calls):
    """Execute tool calls and return results"""
    results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Call the appropriate function
        if function_name == "record_contact_info":
            result = record_contact_info(**arguments)
        elif function_name == "record_unknown_question":
            result = record_unknown_question(**arguments)
        else:
            result = {"status": "error", "message": "Unknown function"}
        
        results.append({
            "tool_call_id": tool_call.id,
            "function_name": function_name,
            "result": result
        })
    
    return results

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with knowledge_base.txt"""
    
    # Check for API key
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    
    if not groq_api_key:
        return None, None, None, "❌ GROQ_API_KEY not found. Please add it to Streamlit Secrets."
    
    # Check if knowledge_base.txt exists
    if not os.path.exists("knowledge_base.txt"):
        return None, None, None, "❌ knowledge_base.txt not found. Please add it to your GitHub repository."
    
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
        
        # Create LLM with tools
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key
        )
        
        return retriever, llm, groq_api_key, None
        
    except Exception as e:
        return None, None, None, f"❌ Error initializing: {str(e)}"

def get_enhanced_system_prompt():
    """Create enhanced system prompt with proactive engagement"""
    return """You are Omar's AI career assistant. You help recruiters, hiring managers, and professionals learn about Omar's background, skills, and experience.

CRITICAL INSTRUCTIONS:

1. **Answer questions using the provided context**: Be specific and include relevant details like skills, dates, companies, certifications, and projects. Always refer to Omar in third person.

2. **When you DON'T know the answer**: 
   - If the information is NOT in the context, you MUST call the 'record_unknown_question' function with the exact question
   - Then politely tell the user you don't have that information
   - Example: "I don't have that specific information about Omar in my knowledge base. Let me record this question so it can be added!"

3. **Proactive engagement**:
   - If the user shows interest in working with Omar, collaborating, or learning more, gently suggest they can connect directly
   - If they express interest, ask for their email in a natural way
   - If they provide contact info, call 'record_contact_info' function
   - Example: "That sounds like an exciting opportunity! If you'd like to discuss this further with Omar directly, I can pass along your contact information. Would you mind sharing your email?"

4. **Be professional yet warm**: You're representing Omar to potential employers and collaborators.

5. **Conversation flow**:
   - Remember the conversation context
   - Build on previous messages naturally
   - Don't repeat information unnecessarily

Context about Omar:
{context}

Chat History:
{chat_history}

Current Question: {question}"""

def chat_with_rag_and_tools(user_question):
    """Enhanced chat function with RAG + Function Calling"""
    
    # Get context from RAG using invoke() instead of get_relevant_documents()
    relevant_docs = st.session_state.retriever.invoke(user_question)
    context = format_docs(relevant_docs)
    
    # Format chat history for context
    chat_history_text = ""
    for msg in st.session_state.conversation_memory[-6:]:  # Last 3 exchanges
        if msg["role"] == "user":
            chat_history_text += f"User: {msg['content']}\n"
        else:
            chat_history_text += f"Assistant: {msg['content']}\n"
    
    # Create messages for LLM
    system_prompt = get_enhanced_system_prompt().format(
        context=context,
        chat_history=chat_history_text,
        question=user_question
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    # Call LLM with tools
    response = st.session_state.llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # Handle tool calls if any
    if response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        tool_results = handle_tool_calls(tool_calls)
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]
        })
        
        # Add tool results
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": json.dumps(tr["result"])
            })
        
        # Get final response
        final_response = st.session_state.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        
        return final_response.choices[0].message.content, tool_results
    
    return response.choices[0].message.content, []

def main():
    # Display header image
    if os.path.exists("header.jpg") or os.path.exists("header.png"):
        header_file = "header.jpg" if os.path.exists("header.jpg") else "header.png"
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.image(header_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #1f77b4;">🤖 Omar\'s AI Career Assistant</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666;">Ask me anything about Omar\'s background, skills, and experience!</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize system on first load
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing AI Assistant..."):
            retriever, llm, api_key, error = initialize_rag_system()
            
            if error:
                st.session_state.error_message = error
                st.session_state.initialized = True
            else:
                st.session_state.retriever = retriever
                st.session_state.llm = ChatGroq(
                    temperature=0.3,
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=api_key
                )
                st.session_state.initialized = True
    
    # Show error if initialization failed
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("💡 Sample Questions")
        
        sample_questions = [
            "What are Omar's main technical skills?",
            "Tell me about Omar's work experience",
            "What certifications does Omar have?",
            "What projects has Omar worked on?",
            "What programming languages does Omar know?",
            "How can I get in touch with Omar?",
        ]
        
        for q in sample_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.conversation_memory.append({"role": "user", "content": q})
                st.rerun()
        
        st.markdown("---")
        
        # Analytics Section
        if len(st.session_state.leads) > 0 or len(st.session_state.unknown_questions) > 0:
            st.subheader("📊 Analytics")
            
            if len(st.session_state.leads) > 0:
                st.metric("Leads Collected", len(st.session_state.leads))
                
                with st.expander("View Leads"):
                    for lead in st.session_state.leads:
                        st.markdown(f"""
                        <div class="lead-notification">
                        <strong>{lead['name']}</strong><br>
                        📧 {lead['email']}<br>
                        💬 {lead['message']}<br>
                        🕐 {lead['timestamp']}
                        </div>
                        """, unsafe_allow_html=True)
            
            if len(st.session_state.unknown_questions) > 0:
                st.metric("Questions to Add", len(st.session_state.unknown_questions))
                
                with st.expander("Unknown Questions"):
                    for uq in st.session_state.unknown_questions:
                        st.markdown(f"""
                        <div class="unknown-question">
                        ❓ {uq['question']}<br>
                        🕐 {uq['timestamp']}
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear conversation button
        if len(st.session_state.messages) > 0:
            if st.button("🔄 Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_memory = []
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        ### 🛠️ Built With
        - **RAG** (Retrieval Augmented Generation)
        - **LangChain** & FAISS
        - **Groq** (Llama 3.3 70B)
        - **Function Calling** for lead gen
        - **Streamlit**
        
        ---
        
        ### ℹ️ About
        This AI assistant combines RAG with intelligent function calling to:
        - Answer questions about Omar
        - Collect leads automatically
        - Track improvement opportunities
        
        **Built by Omar** to showcase AI/ML skills!
        """)
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        st.info("👋 **Welcome!** I'm Omar's AI assistant, trained on his professional background. Ask me anything about Omar's skills, experience, certifications, or projects!")
        
        st.markdown("### 💬 Try Asking:")
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("""
            - What are Omar's main skills?
            - Tell me about Omar's work experience
            - What certifications does Omar have?
            - What projects has Omar worked on?
            """)
        
        with cols[1]:
            st.markdown("""
            - What programming languages does Omar know?
            - How can I contact Omar?
            - What is Omar's latest position?
            - Tell me about Omar's AI/ML experience
            """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Omar..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_memory.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response with RAG + Tools
                    answer, tool_results = chat_with_rag_and_tools(prompt)
                    
                    # Show tool notifications
                    if tool_results:
                        for tr in tool_results:
                            if tr["function_name"] == "record_contact_info":
                                st.success("✅ Thank you! I've recorded your contact information. Omar will get back to you soon!")
                            elif tr["function_name"] == "record_unknown_question":
                                st.info("📝 I've recorded this question to improve my knowledge base!")
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.session_state.conversation_memory.append({
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
