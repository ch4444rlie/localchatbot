# legal_advisor_app.py
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import fitz  # PyMuPDF
import ollama
import json
import os
import glob
import re
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import datetime

# Pydantic models
class LegalQuery(BaseModel):
    query_type: str = Field(..., description="Type of legal query (e.g., eviction, court_order, contract, general)")
    urgency: str = Field(default="normal", description="Urgency level (urgent, normal, low)")
    document_type: Optional[str] = Field(default=None, description="Type of document being analyzed")

class LegalResponse(BaseModel):
    answer: str = Field(..., description="Legal guidance response")
    sources: List[str] = Field(default_factory=list, description="List of document sources")
    urgency_flag: bool = Field(default=False, description="Whether this requires urgent attention")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    warnings: List[str] = Field(default_factory=list, description="Important warnings or disclaimers")

# Legal document keywords and patterns
LEGAL_KEYWORDS = {
    'eviction': ['eviction', 'evict', 'notice to quit', 'unlawful detainer', 'rent', 'landlord', 'tenant', 'lease'],
    'court_order': ['court order', 'judgment', 'decree', 'injunction', 'subpoena', 'summons', 'hearing'],
    'debt': ['debt', 'collection', 'creditor', 'garnishment', 'bankruptcy', 'foreclosure'],
    'employment': ['termination', 'wrongful dismissal', 'discrimination', 'harassment', 'wages', 'overtime'],
    'family': ['divorce', 'custody', 'child support', 'alimony', 'domestic violence', 'restraining order'],
    'benefits': ['disability', 'unemployment', 'welfare', 'food stamps', 'medicaid', 'social security'],
    'immigration': ['deportation', 'visa', 'asylum', 'green card', 'citizenship', 'ice', 'removal']
}

URGENT_PATTERNS = [
    r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # Dates
    r'within \d+ days?',
    r'\d+ days? to',
    r'deadline',
    r'immediate',
    r'emergency',
    r'urgent',
    r'final notice',
    r'last notice',
    r'court date',
    r'hearing date'
]

def load_legal_document(file_path: str) -> tuple[str, List[Dict]]:
    """Parse legal documents (PDF, TXT, DOC) into chunks with metadata."""
    chunks = []
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if file_ext == '.pdf':
            doc = fitz.open(file_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text += f"\nPage {page_num + 1}: {text}\n"
                    
                    # Create page-level chunks with legal context
                    chunks.append({
                        "text": text,
                        "metadata": {
                            "file_name": file_name,
                            "file_type": "pdf",
                            "page": page_num + 1,
                            "content_type": "legal_document",
                            "document_type": detect_document_type(text)
                        }
                    })
            doc.close()
            
            # Add a full document summary chunk
            doc_summary = create_document_summary(full_text, file_name)
            chunks.append({
                "text": doc_summary,
                "metadata": {
                    "file_name": file_name,
                    "file_type": "pdf",
                    "content_type": "document_summary",
                    "document_type": detect_document_type(full_text)
                }
            })
            
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
                chunks.append({
                    "text": text,
                    "metadata": {
                        "file_name": file_name,
                        "file_type": "txt",
                        "content_type": "legal_document",
                        "document_type": detect_document_type(text)
                    }
                })
                
                # Add summary
                doc_summary = create_document_summary(text, file_name)
                chunks.append({
                    "text": doc_summary,
                    "metadata": {
                        "file_name": file_name,
                        "file_type": "txt",
                        "content_type": "document_summary",
                        "document_type": detect_document_type(text)
                    }
                })
        else:
            st.error(f"Unsupported file type: {file_ext}. Please upload PDF or TXT files.")
            return file_name, []
        
        # Split large chunks while preserving legal structure
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        processed_chunks = []
        for chunk in chunks:
            if len(chunk["text"]) > 1500 and chunk["metadata"]["content_type"] != "document_summary":
                split_texts = splitter.split_text(chunk["text"])
                for i, split_text in enumerate(split_texts):
                    new_metadata = chunk["metadata"].copy()
                    new_metadata["chunk_id"] = i
                    processed_chunks.append({"text": split_text, "metadata": new_metadata})
            else:
                processed_chunks.append(chunk)
        
        return file_name, processed_chunks
        
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return file_name, []

def detect_document_type(text: str) -> str:
    """Detect the type of legal document based on content."""
    text_lower = text.lower()
    
    # Check against legal keyword categories
    for doc_type, keywords in LEGAL_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return doc_type
    
    # Check for specific document patterns
    if any(pattern in text_lower for pattern in ['notice to quit', 'eviction notice']):
        return 'eviction'
    elif any(pattern in text_lower for pattern in ['court order', 'judgment', 'decree']):
        return 'court_order'
    elif any(pattern in text_lower for pattern in ['contract', 'agreement', 'lease']):
        return 'contract'
    elif any(pattern in text_lower for pattern in ['letter', 'correspondence']):
        return 'correspondence'
    
    return 'general'

def create_document_summary(text: str, file_name: str) -> str:
    """Create a summary of the legal document with key information extracted."""
    text_lower = text.lower()
    
    # Extract key dates
    dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text)
    
    # Extract monetary amounts
    amounts = re.findall(r'\$[\d,]+\.?\d*', text)
    
    # Extract important parties (names in caps)
    parties = re.findall(r'\b[A-Z][A-Z\s]{2,20}\b', text)
    
    # Detect urgency
    urgent_matches = []
    for pattern in URGENT_PATTERNS:
        matches = re.findall(pattern, text_lower)
        urgent_matches.extend(matches)
    
    doc_type = detect_document_type(text)
    
    summary = f"Document: {file_name}\n"
    summary += f"Type: {doc_type.replace('_', ' ').title()}\n"
    
    if dates:
        summary += f"Important Dates: {', '.join(dates[:3])}\n"
    
    if amounts:
        summary += f"Monetary Amounts: {', '.join(amounts[:3])}\n"
    
    if urgent_matches:
        summary += f"Urgent Elements: {', '.join(urgent_matches[:3])}\n"
    
    # Add document-specific insights
    if doc_type == 'eviction':
        summary += "⚠️ EVICTION NOTICE - This may require immediate legal attention\n"
    elif doc_type == 'court_order':
        summary += "⚠️ COURT DOCUMENT - May have legal deadlines\n"
    elif doc_type == 'debt':
        summary += "💰 DEBT-RELATED - May affect credit and assets\n"
    
    return summary

def classify_query(query: str) -> LegalQuery:
    """Classify the user's query to determine response approach."""
    query_lower = query.lower()
    
    # Determine query type
    query_type = "general"
    for doc_type, keywords in LEGAL_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            query_type = doc_type
            break
    
    # Determine urgency
    urgency = "normal"
    if any(word in query_lower for word in ['urgent', 'emergency', 'immediate', 'asap', 'deadline', 'court date']):
        urgency = "urgent"
    elif any(word in query_lower for word in ['when i have time', 'eventually', 'someday']):
        urgency = "low"
    
    return LegalQuery(query_type=query_type, urgency=urgency)

def setup_legal_rag(chunks: List[Dict]) -> any:
    """Set up RAG pipeline optimized for legal document retrieval."""
    if not chunks:
        return None
        
    try:
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        embeddings = OllamaEmbeddings(model='nomic-embed-text')
        
        # Use temporary directory for vector store
        temp_dir = tempfile.mkdtemp()
        vectorstore = Chroma.from_texts(
            texts, 
            embeddings, 
            metadatas=metadatas, 
            persist_directory=temp_dir
        )
        return vectorstore.as_retriever(search_kwargs={"k": 8})
    except Exception as e:
        st.error(f"Error setting up legal document retrieval: {str(e)}")
        return None

def generate_legal_response(chunks: List[Dict], query: str, query_classification: LegalQuery) -> LegalResponse:
    """Generate legal guidance response using RAG."""
    try:
        retriever = setup_legal_rag(chunks)
        if not retriever:
            return LegalResponse(
                answer="Error setting up document retrieval system.",
                warnings=["Unable to analyze documents properly"]
            )
        
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(query)
        if not relevant_docs:
            return LegalResponse(
                answer="I couldn't find specific information related to your question in the uploaded documents. Please try rephrasing your question or upload additional relevant documents.",
                warnings=["No relevant information found in documents"]
            )
        
        # Create context from retrieved documents
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set(doc.metadata.get("file_name", "unknown") for doc in relevant_docs))
        
        # Check for urgency indicators in context
        urgency_flag = any(re.search(pattern, context.lower()) for pattern in URGENT_PATTERNS)
        
        # Create specialized prompt for legal guidance
        legal_prompt = f"""You are providing legal information to help someone understand their documents. This is NOT legal advice, but educational information to help them understand their situation.

IMPORTANT DISCLAIMERS:
- This is informational guidance only, not legal advice
- Laws vary by location and individual circumstances
- Always consult with a qualified attorney for legal advice
- Time-sensitive matters require immediate professional legal help

Document Context:
{context}

User Question: {query}

Query Type: {query_classification.query_type}
Urgency Level: {query_classification.urgency}

Please provide:
1. A clear explanation of what the documents mean in plain language
2. Key points they should understand about their situation
3. Potential next steps they might consider
4. Any urgent deadlines or important dates
5. When they should seek professional legal help

Be empathetic, clear, and helpful while emphasizing the importance of professional legal counsel for their specific situation."""

        # Generate response using Ollama with progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 Analyzing your documents...")
        progress_bar.progress(25)
        
        status_text.text("📋 Understanding your legal situation...")
        progress_bar.progress(50)
        
        status_text.text("✍️ Preparing your guidance...")
        progress_bar.progress(75)
        
        response = ollama.chat(
            model='llama2:7b-chat',  # Better for conversational legal guidance
            messages=[{'role': 'user', 'content': legal_prompt}]
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        answer = response['message']['content']
        
        # Extract next steps and warnings from the response
        next_steps = []
        warnings = [
            "This is informational guidance only, not legal advice",
            "Always consult with a qualified attorney for legal advice about your specific situation"
        ]
        
        # Add urgency-specific warnings
        if urgency_flag or query_classification.urgency == "urgent":
            warnings.insert(0, "⚠️ This appears to be time-sensitive. Seek immediate legal help.")
            urgency_flag = True
        
        # Add document-type specific warnings
        if query_classification.query_type == "eviction":
            warnings.append("Eviction proceedings move quickly. Contact a tenant rights organization immediately.")
            next_steps.append("Contact local tenant rights organization or legal aid")
            next_steps.append("Gather all lease documents and payment records")
        elif query_classification.query_type == "court_order":
            warnings.append("Court orders have strict deadlines. Missing deadlines can have serious consequences.")
            next_steps.append("Note all deadlines mentioned in the document")
            next_steps.append("Contact the court or an attorney immediately")
        elif query_classification.query_type == "debt":
            next_steps.append("Contact a debt counseling service")
            next_steps.append("Review all debt documentation carefully")
        
        return LegalResponse(
            answer=answer,
            sources=sources,
            urgency_flag=urgency_flag,
            next_steps=next_steps,
            warnings=warnings
        )
        
    except Exception as e:
        return LegalResponse(
            answer=f"I encountered an error while analyzing your documents: {str(e)}",
            warnings=["System error occurred", "Please try uploading documents again or contact support"]
        )

def display_legal_response(response: LegalResponse):
    """Display the legal response with appropriate formatting and warnings."""
    
    # Display urgency alert if needed
    if response.urgency_flag:
        st.error("🚨 URGENT: This matter may be time-sensitive and require immediate attention!")
    
    # Display main answer
    st.write("### 📋 Document Analysis")
    st.write(response.answer)
    
    # Display sources
    if response.sources:
        st.write("### 📄 Based on Documents:")
        for source in response.sources:
            st.write(f"• {source}")
    
    # Display next steps
    if response.next_steps:
        st.write("### ✅ Potential Next Steps:")
        for step in response.next_steps:
            st.write(f"• {step}")
    
    # Display warnings prominently
    if response.warnings:
        st.write("### ⚠️ Important Disclaimers:")
        for warning in response.warnings:
            st.warning(warning)

def main():
    st.set_page_config(
        page_title="Legal Document Helper",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document Helper")
    st.subtitle("Free legal document analysis for those who need it most")
    
    # Initialize session state
    if 'legal_chunks' not in st.session_state:
        st.session_state.legal_chunks = []
    if 'file_paths' not in st.session_state:
        st.session_state.file_paths = {}
    
    # Create temp directory
    os.makedirs("./temp", exist_ok=True)
    
    # Sidebar with information
    with st.sidebar:
        st.write("### 🏠 Common Legal Issues We Help With:")
        st.write("• Eviction notices")
        st.write("• Court orders and summons")
        st.write("• Debt collection letters")
        st.write("• Employment termination")
        st.write("• Family court documents")
        st.write("• Benefits appeals")
        st.write("• Immigration notices")
        
        st.write("### 🚨 Emergency Resources:")
        st.write("**Legal Aid:** 211 (dial 2-1-1)")
        st.write("**Tenant Rights:** Local tenant unions")
        st.write("**Domestic Violence:** National hotline 1-800-799-7233")
        st.write("**Immigration:** ACLU Immigration assistance")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### How to Use This Tool:
        1. **Upload your legal documents** (PDF or text files)
        2. **Ask questions** about what the documents mean
        3. **Get plain-language explanations** and guidance
        4. **Find next steps** for your situation
        
        **Remember:** This tool provides information only, not legal advice. Always consult with a qualified attorney for legal advice about your specific situation.
        """)
        
        # File upload section
        st.write("### 📁 Upload Your Legal Documents")
        uploaded_files = st.file_uploader(
            "Upload legal documents (PDF, TXT)", 
            type=['pdf', 'txt'], 
            accept_multiple_files=True,
            help="Upload eviction notices, court orders, contracts, or other legal documents"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                file_path = f"./temp/{file.name}"
                
                # Save uploaded file
                with open(file_path, "wb") as f:
                    f.write(file.read())
                
                # Store file path
                st.session_state.file_paths[file.name] = file_path
                
                # Load legal document
                with st.spinner(f"Analyzing {file.name}..."):
                    file_name, chunks = load_legal_document(file_path)
                    if chunks:
                        # Remove existing chunks for this file
                        st.session_state.legal_chunks = [
                            chunk for chunk in st.session_state.legal_chunks 
                            if chunk["metadata"]["file_name"] != file_name
                        ]
                        # Add new chunks
                        st.session_state.legal_chunks.extend(chunks)
                        st.success(f"✅ Analyzed {file_name} - Found {len(chunks)} sections")
                        
                        # Show document type detection
                        doc_type = chunks[0]["metadata"].get("document_type", "general")
                        st.info(f"📋 Document type detected: **{doc_type.replace('_', ' ').title()}**")
                    else:
                        st.error(f"❌ Failed to analyze {file_name}")
        
        # Display loaded files
        if st.session_state.legal_chunks:
            loaded_files = list(set(chunk["metadata"]["file_name"] for chunk in st.session_state.legal_chunks))
            st.write(f"### 📋 Loaded Documents: {', '.join(loaded_files)}")
        
        # Query section
        st.write("### 💬 Ask About Your Documents")
        
        # Example questions based on document types
        if st.session_state.legal_chunks:
            doc_types = set(chunk["metadata"].get("document_type", "general") for chunk in st.session_state.legal_chunks)
            if "eviction" in doc_types:
                st.write("**Try asking:** 'How much time do I have to respond to this eviction notice?'")
            elif "court_order" in doc_types:
                st.write("**Try asking:** 'What is the court date and what do I need to do?'")
            elif "debt" in doc_types:
                st.write("**Try asking:** 'What are my options for dealing with this debt?'")
        
        query = st.text_area(
            "What would you like to know about your documents?",
            placeholder="For example: What does this eviction notice mean? How long do I have to respond? What are my options?",
            height=100
        )
        
        if st.button("Get Help", type="primary"):
            if not query:
                st.warning("Please enter a question about your documents.")
            elif not st.session_state.legal_chunks:
                st.warning("Please upload legal documents first.")
            else:
                with st.spinner("Analyzing your documents and preparing guidance..."):
                    try:
                        query_classification = classify_query(query)
                        response = generate_legal_response(st.session_state.legal_chunks, query, query_classification)
                        display_legal_response(response)
                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")
    
    with col2:
        st.write("### 🔗 Additional Resources")
        st.write("""
        **Free Legal Help:**
        - [LegalAid.org](https://www.legalaid.org)
        - [Lawhelp.org](https://www.lawhelp.org)
        - [JusticeForAll.org](https://www.jfa.net)
        
        **Know Your Rights:**
        - [ACLU Know Your Rights](https://www.aclu.org/know-your-rights)
        - [National Low Income Housing Coalition](https://nlihc.org)
        
        **Emergency Assistance:**
        - Local Legal Aid: 211
        - Domestic Violence: 1-800-799-7233
        - Suicide Prevention: 988
        """)
        
        if st.session_state.legal_chunks:
            st.write("### 📊 Document Summary")
            doc_types = {}
            for chunk in st.session_state.legal_chunks:
                doc_type = chunk["metadata"].get("document_type", "general")
                if doc_type not in doc_types:
                    doc_types[doc_type] = 0
                doc_types[doc_type] += 1
            
            for doc_type, count in doc_types.items():
                st.write(f"• {doc_type.replace('_', ' ').title()}: {count} sections")

if __name__ == "__main__":
    main()