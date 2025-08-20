# app.py
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import fitz  # PyMuPDF
import ollama
import json
import os
import glob
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Pydantic models
class QueryCommand(BaseModel):
    command: str = Field(..., description="Command type (e.g., summarize, answer_question, search, compare_files)")
    params: Dict = Field(default_factory=dict, description="Parameters like keyword, file_name, file_type")

class ChatbotResponse(BaseModel):
    answer: str = Field(..., description="Concise response to the query, under 100 words", min_length=1, max_length=100)
    sources: List[str] = Field(default_factory=list, description="List of file names referenced in the answer")

# Document Loading Module
def load_document(file_path: str) -> tuple[str, List[Dict]]:
    """Parse a document (Excel, CSV, PDF, TXT) into chunks with metadata."""
    chunks = []
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if file_ext in ['.xlsx', '.xls']:
            sheets = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in sheets.items():
                for i, row in df.iterrows():
                    chunks.append({"text": f"Sheet: {sheet_name}, Row {i}: {row.to_dict()}", 
                                  "metadata": {"file_name": file_name, "file_type": "excel", "sheet": sheet_name}})
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            for i, row in df.iterrows():
                chunks.append({"text": f"Row {i}: {row.to_dict()}", 
                              "metadata": {"file_name": file_name, "file_type": "csv"}})
        elif file_ext == '.pdf':
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text()
                chunks.append({"text": f"Page {page.number}: {text}", 
                              "metadata": {"file_name": file_name, "file_type": "pdf", "page": page.number}})
            doc.close()
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks.append({"text": text, "metadata": {"file_name": file_name, "file_type": "txt"}})
        else:
            return file_name, []
        
        # Split large texts
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_chunks = []
        for chunk in chunks:
            split_texts = splitter.split_text(chunk["text"])
            for split_text in split_texts:
                split_chunks.append({"text": split_text, "metadata": chunk["metadata"]})
        
        return file_name, split_chunks
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return file_name, []

# Command Extraction Module
def extract_command(query: str) -> QueryCommand:
    """Use Mistral to parse query into a structured command."""
    prompt = r"""
        You are a document repository assistant. Given this query: '{query}', identify the command 
        (e.g., 'summarize', 'answer_question', 'search', 'compare_files') and parameters 
        (e.g., file_name, keyword, file_type). Output JSON: {'command': '...', 'params': {'file_name': '...', 'keyword': '...', 'file_type': '...'}}\. 
        Examples:
        Query: 'Summarize sales data' → {'command': 'summarize', 'params': {'keyword': 'sales', 'file_type': 'all'}}
        Query: 'Compare revenue in file1.xlsx and file2.xlsx' → {'command': 'compare_files', 'params': {'file_name': ['file1.xlsx', 'file2.xlsx'], 'keyword': 'revenue'}}
        Query: 'What is in report.pdf?' → {'command': 'answer_question', 'params': {'file_name': 'report.pdf', 'file_type': 'pdf'}}
    """
    response = ollama.chat(model='mistral:7b-instruct-v0.3-q4_0', messages=[
        {'role': 'system', 'content': prompt.format(query=query)},
        {'role': 'user', 'content': f"Query: {query}"}
    ])
    try:
        command_dict = json.loads(response['message']['content'])
        return QueryCommand(**command_dict)
    except:
        return QueryCommand(command="answer_question", params={"keyword": query, "file_type": "all"})

# Data Retrieval Module
def setup_rag(chunks: List[Dict]) -> any:
    """Set up RAG pipeline with Chroma for repository-wide retrieval."""
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory='./db')
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Response Generation Module
def generate_response(chunks: List[Dict], query: str, command: QueryCommand) -> ChatbotResponse:
    """Generate a concise, document-grounded response using RAG."""
    retriever = setup_rag(chunks)
    # Apply metadata filters if specified
    filter_params = {}
    if command.params.get("file_type") != "all":
        filter_params["file_type"] = command.params["file_type"]
    if command.params.get("file_name"):
        filter_params["file_name"] = {"$in": command.params["file_name"]} if isinstance(command.params["file_name"], list) else command.params["file_name"]
    retriever.search_kwargs["filter"] = filter_params if filter_params else None
    
    llm = ChatOllama(model='mistral:7b-instruct-v0.3-q4_0')
    prompt = ChatPromptTemplate.from_template(
        "Context: {context}\nQuestion: {query}\nProvide a concise answer (under 100 words) based only on the document data. Cite file names."
    )
    chain = {"context": retriever, "query": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    answer = chain.invoke(query)
    
    # Extract sources from retrieved documents
    retrieved_docs = retriever.get_relevant_documents(query)
    sources = list(set(doc.metadata.get("file_name", "unknown") for doc in retrieved_docs))
    
    return ChatbotResponse(answer=answer, sources=sources)

# Query Handling
def handle_query(chunks: List[Dict], query: str) -> ChatbotResponse:
    """Handle a user query by extracting the command and generating a response."""
    try:
        command = extract_command(query)
        response = generate_response(chunks, query, command)
        return response
    except Exception as e:
        return ChatbotResponse(answer=f"Error processing query: {str(e)}", sources=[])

# Streamlit UI
def main():
    st.title("Local Document Repository RAG Chatbot")
    
    # Initialize session state for chunks
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []

    uploaded_files = st.file_uploader("Upload documents", type=['xlsx', 'xls', 'csv', 'pdf', 'txt'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = f"./temp/{file.name}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.read())
            file_name, chunks = load_document(file_path)
            if chunks:
                st.session_state.all_chunks.extend(chunks)
                st.success(f"Loaded {file_name}")
    
    query = st.text_input("Ask about the document repository (e.g., 'Summarize sales data', 'Compare revenue in file1.xlsx and file2.xlsx')")
    if query:
        if not st.session_state.all_chunks:
            st.warning("Please upload documents before querying.")
        else:
            response = handle_query(st.session_state.all_chunks, query)
            st.write(f"**Answer**: {response.answer}")
            st.write(f"**Sources**: {', '.join(response.sources)}")

if __name__ == "__main__":
    main()