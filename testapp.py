import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
import pandas as pd
import fitz  # PyMuPDF
import ollama
import json
import os
import glob
import re
import random
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Pydantic models
class QueryCommand(BaseModel):
    command: str = Field(..., description="Command type (e.g., summarize, answer_question, search, compare_files, analyze, analyze_image)")
    params: Dict = Field(default_factory=dict, description="Parameters like keyword, file_name, file_type, column, agg_type, page")

class ChatbotResponse(BaseModel):
    answer: str = Field(..., description="Response to the query")
    sources: List[str] = Field(default_factory=list, description="List of file names referenced in the answer")
    chart: Optional[object] = Field(default=None, description="Matplotlib figure object for visualization")
    chart_type: Optional[str] = Field(default=None, description="Type of chart generated")

# Document Loading Module
def load_document(file_path: str) -> tuple[str, List[Dict]]:
    """Parse a document (Excel, CSV, PDF, TXT) into chunks with metadata, including images from PDFs."""
    chunks = []
    file_name = os.path.basename(file_path)
    
    # Handle double extensions (e.g., transaction_data.csv.csv)
    if file_name.endswith('.csv.csv'):
        file_name = file_name[:-4]  # Remove extra .csv
    
    file_ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if file_ext in ['.xlsx', '.xls']:
            try:
                sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    # Add sheet summary
                    summary = f"Sheet: {sheet_name}, Columns: {', '.join(df.columns.astype(str))}, Rows: {len(df)}"
                    chunks.append({
                        "text": summary,
                        "metadata": {"file_name": file_name, "file_type": "excel", "sheet": sheet_name, "content_type": "summary"}
                    })
                    
                    # Add column info
                    for col in df.columns:
                        col_info = f"Column '{col}' in sheet '{sheet_name}': {df[col].dtype}, sample values: {df[col].dropna().head(3).tolist()}"
                        chunks.append({
                            "text": col_info,
                            "metadata": {"file_name": file_name, "file_type": "excel", "sheet": sheet_name, "column": col, "content_type": "column_info"}
                        })
                    
                    # Add sample rows
                    for i, row in df.head(10).iterrows():
                        row_text = f"Sheet: {sheet_name}, Row {i}: " + ", ".join([f"{k}: {v}" for k, v in row.to_dict().items()])
                        chunks.append({
                            "text": row_text,
                            "metadata": {"file_name": file_name, "file_type": "excel", "sheet": sheet_name, "row": i, "content_type": "data"}
                        })
            except Exception as e:
                st.error(f"Error reading Excel file {file_name}: {str(e)}")
                return file_name, []
                
        elif file_ext == '.csv':
            try:
                df = pd.read_csv(file_path)
                
                # Add file summary
                summary = f"CSV file: {file_name}, Columns: {', '.join(df.columns.astype(str))}, Rows: {len(df)}"
                chunks.append({
                    "text": summary,
                    "metadata": {"file_name": file_name, "file_type": "csv", "content_type": "summary"}
                })
                
                # Add column info
                for col in df.columns:
                    try:
                        sample_vals = df[col].dropna().head(3).tolist()
                        col_info = f"Column '{col}': {df[col].dtype}, sample values: {sample_vals}"
                        if pd.api.types.is_numeric_dtype(df[col]):
                            col_info += f", min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}"
                        chunks.append({
                            "text": col_info,
                            "metadata": {"file_name": file_name, "file_type": "csv", "column": col, "content_type": "column_info"}
                        })
                    except Exception as e:
                        continue
                
                # Add sample rows
                for i, row in df.head(20).iterrows():
                    row_text = f"Row {i}: " + ", ".join([f"{k}: {v}" for k, v in row.to_dict().items()])
                    chunks.append({
                        "text": row_text,
                        "metadata": {"file_name": file_name, "file_type": "csv", "row": i, "content_type": "data"}
                    })
            except Exception as e:
                st.error(f"Error reading CSV file {file_name}: {str(e)}")
                return file_name, []
                
        elif file_ext == '.pdf':
            try:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Extract text
                    text = page.get_text()
                    if text.strip():
                        chunks.append({
                            "text": f"Page {page_num + 1}: {text}",
                            "metadata": {"file_name": file_name, "file_type": "pdf", "page": page_num + 1, "content_type": "text"}
                        })
                    # Extract images
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        # Save image temporarily
                        image_path = f"./temp/{file_name}_page{page_num + 1}_img{img_index}.png"
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        chunks.append({
                            "text": f"Image on page {page_num + 1} of {file_name}",
                            "metadata": {
                                "file_name": file_name,
                                "file_type": "pdf",
                                "page": page_num + 1,
                                "content_type": "image",
                                "image_path": image_path
                            }
                        })
                doc.close()
            except Exception as e:
                st.error(f"Error reading PDF file {file_name}: {str(e)}")
                return file_name, []
                
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    chunks.append({
                        "text": text,
                        "metadata": {"file_name": file_name, "file_type": "txt", "content_type": "text"}
                    })
            except Exception as e:
                st.error(f"Error reading TXT file {file_name}: {str(e)}")
                return file_name, []
        else:
            st.error(f"Unsupported file extension: {file_ext}")
            return file_name, []
        
        # Split large texts
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_chunks = []
        for chunk in chunks:
            if chunk["metadata"]["content_type"] != "image" and len(chunk["text"]) > 1000:
                split_texts = splitter.split_text(chunk["text"])
                for i, split_text in enumerate(split_texts):
                    new_metadata = chunk["metadata"].copy()
                    new_metadata["chunk_id"] = i
                    split_chunks.append({"text": split_text, "metadata": new_metadata})
            else:
                split_chunks.append(chunk)
        
        return file_name, split_chunks
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return file_name, []

# Visualization Module
def create_visualization(df: pd.DataFrame, query: str, command: QueryCommand) -> Tuple[Optional[object], str]:
    """Create visualizations based on query and data."""
    try:
        query_lower = query.lower()
        viz_type = command.params.get("viz_type", "")
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Correlation plot
        if any(word in query_lower for word in ['correlation', 'correlate', 'relationship', 'vs']):
            x_col = command.params.get("x_column")
            y_col = command.params.get("y_column")
            
            if not x_col or not y_col:
                words = re.findall(r'\b\w+\b', query_lower)
                potential_cols = [word for word in words if word in [col.lower() for col in df.columns]]
                
                actual_cols = []
                for word in potential_cols:
                    for col in df.columns:
                        if word == col.lower():
                            actual_cols.append(col)
                
                if len(actual_cols) >= 2:
                    x_col, y_col = actual_cols[0], actual_cols[1]
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        x_col, y_col = numeric_cols[0], numeric_cols[1]
            
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    clean_df = df[[x_col, y_col]].dropna()
                    ax.scatter(clean_df[x_col], clean_df[y_col], alpha=0.6, s=50)
                    z = np.polyfit(clean_df[x_col], clean_df[y_col], 1)
                    p = np.poly1d(z)
                    ax.plot(clean_df[x_col].sort_values(), p(clean_df[x_col].sort_values()), "r--", alpha=0.8)
                    correlation = clean_df[x_col].corr(clean_df[y_col])
                    ax.set_xlabel(x_col.replace('_', ' ').title())
                    ax.set_ylabel(y_col.replace('_', ' ').title())
                    ax.set_title(f'Correlation: {x_col} vs {y_col}\nCorrelation coefficient: {correlation:.3f}')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    return fig, f"correlation between {x_col} and {y_col}"
        
        # Distribution/Histogram
        elif any(word in query_lower for word in ['distribution', 'histogram', 'hist', 'spread']):
            column = command.params.get("column")
            
            if not column:
                words = re.findall(r'\b\w+\b', query_lower)
                for word in words:
                    for col in df.columns:
                        if word == col.lower():
                            column = col
                            break
                    if column:
                        break
                
                if not column:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        column = numeric_cols[0]
            
            if column and column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                fig, ax = plt.subplots(figsize=(10, 6))
                clean_data = df[column].dropna()
                ax.hist(clean_data, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(clean_data.mean(), color='red', linestyle='--', label=f'Mean: {clean_data.mean():.2f}')
                ax.axvline(clean_data.median(), color='green', linestyle='--', label=f'Median: {clean_data.median():.2f}')
                ax.set_xlabel(column.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {column.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                return fig, f"distribution of {column}"
        
        # Box plot
        elif any(word in query_lower for word in ['boxplot', 'box plot', 'quartile', 'outlier']):
            column = command.params.get("column")
            group_by = command.params.get("group_by")
            
            if not column:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    column = numeric_cols[0]
            
            if column and column in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                if group_by and group_by in df.columns:
                    df.boxplot(column=column, by=group_by, ax=ax)
                    ax.set_title(f'Box Plot: {column} by {group_by}')
                else:
                    ax.boxplot(df[column].dropna())
                    ax.set_title(f'Box Plot: {column}')
                    ax.set_xticklabels([column.replace('_', ' ').title()])
                ax.set_ylabel(column.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                return fig, f"box plot of {column}"
        
        # Bar chart for categorical data
        elif any(word in query_lower for word in ['bar chart', 'bar plot', 'count by', 'breakdown']):
            column = command.params.get("column")
            
            if not column:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols:
                    column = cat_cols[0]
            
            if column and column in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = df[column].value_counts().head(10)
                ax.bar(range(len(value_counts)), value_counts.values)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_ylabel('Count')
                ax.set_title(f'Distribution of {column.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                return fig, f"bar chart of {column}"
        
        # Time series plot
        elif any(word in query_lower for word in ['time series', 'over time', 'trend', 'timeline']):
            date_cols = []
            for col in df.columns:
                if any(word in col.lower() for word in ['date', 'time', 'created', 'updated']):
                    date_cols.append(col)
            
            if date_cols:
                date_col = date_cols[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    value_col = numeric_cols[0]
                    fig, ax = plt.subplots(figsize=(12, 6))
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df_sorted = df.sort_values(date_col)
                        ax.plot(df_sorted[date_col], df_sorted[value_col], marker='o', markersize=4)
                        ax.set_xlabel(date_col.replace('_', ' ').title())
                        ax.set_ylabel(value_col.replace('_', ' ').title())
                        ax.set_title(f'{value_col.replace("_", " ").title()} Over Time')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        return fig, f"time series of {value_col}"
                    except:
                        pass
        
        return None, ""
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None, ""

# Analyze CSV Data
def analyze_csv(file_path: str, query: str, command: QueryCommand) -> Tuple[str, Optional[object], str]:
    """Perform numerical or statistical analysis on CSV data based on query."""
    try:
        df = pd.read_csv(file_path)
        
        chart, chart_type = create_visualization(df, query, command)
        
        column = command.params.get("column")
        agg_type = command.params.get("agg_type", "sum")
        group_by = command.params.get("group_by")
        
        if not column:
            money_cols = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'price', 'cost', 'total', 'revenue', 'value', 'money'])]
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if money_cols:
                column = money_cols[0]
            elif numeric_cols:
                column = numeric_cols[0]
            else:
                result_text = f"No numeric columns found. Available columns: {', '.join(df.columns)}"
                return result_text, chart, chart_type
        
        if column and column not in df.columns:
            similar_cols = [col for col in df.columns if column.lower() in col.lower() or col.lower() in column.lower()]
            if similar_cols:
                column = similar_cols[0]
            else:
                result_text = f"Column '{column}' not found. Available columns: {', '.join(df.columns)}"
                return result_text, chart, chart_type
        
        if agg_type == "sum" and column:
            if pd.api.types.is_numeric_dtype(df[column]):
                result = df[column].sum()
                result_text = f"Total {column}: {result:,.2f}"
            else:
                result_text = f"Column '{column}' is not numeric. Cannot calculate sum."
        elif agg_type == "average" and column:
            if pd.api.types.is_numeric_dtype(df[column]):
                result = df[column].mean()
                result_text = f"Average {column}: {result:,.2f}"
            else:
                result_text = f"Column '{column}' is not numeric. Cannot calculate average."
        elif agg_type == "count" and column:
            result = df[column].count()
            result_text = f"Count of non-null {column} values: {result:,}"
        elif agg_type == "group_average" and column and group_by:
            if group_by not in df.columns:
                result_text = f"Group column '{group_by}' not found. Available columns: {', '.join(df.columns)}"
            elif pd.api.types.is_numeric_dtype(df[column]):
                grouped = df.groupby(group_by)[column].mean()
                avg_of_groups = grouped.mean()
                result_text = f"Average {column} per {group_by}: {avg_of_groups:,.2f} (across {len(grouped)} groups)"
            else:
                result_text = f"Column '{column}' is not numeric. Cannot calculate group average."
        elif agg_type == "random_row":
            if df.empty:
                result_text = "No data available in the CSV."
            else:
                random_row = df.sample(n=1).iloc[0]
                row_dict = {k: v for k, v in random_row.to_dict().items()}
                result_text = f"Random row: {row_dict}"
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            result_text = f"Dataset has {len(df)} rows and {len(df.columns)} columns. "
            if numeric_cols:
                result_text += f"Numeric columns: {', '.join(numeric_cols)}. "
            result_text += f"Available columns: {', '.join(df.columns)}"
        
        return result_text, chart, chart_type
            
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}", None, ""

# Analyze Image with LLaVA
def analyze_image(image_path: str, query: str) -> str:
    """Analyze an image (e.g., chart/graph) using LLaVA-13B."""
    try:
        # Load image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Prepare prompt for LLaVA
        prompt = f"Analyze the following image (likely a chart or graph) and answer the query: {query}\nDescribe the content concisely, focusing on key data or visual elements."
        
        # Call LLaVA-13B
        response = ollama.chat(
            model='llava:13b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }
            ]
        )
        
        return response['message']['content']
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Command Extraction
def extract_command(query: str) -> QueryCommand:
    """Parse query into a structured command with image support."""
    query_lower = query.lower()
    
    # Image-related queries
    if any(word in query_lower for word in ['chart', 'graph', 'image', 'graphic', 'visual', 'figure']):
        page_match = re.search(r'page\s+(\d+)', query_lower)
        file_match = re.search(r'\b[\w.-]+\.pdf\b', query_lower)
        params = {}
        if page_match:
            params["page"] = int(page_match.group(1))
        if file_match:
            params["file_name"] = file_match.group(0)
        return QueryCommand(command="analyze_image", params=params)
    
    # Visualization requests
    if any(word in query_lower for word in ['plot', 'chart', 'graph', 'visualize', 'visualization']):
        if any(word in query_lower for word in ['correlation', 'correlate', 'relationship', 'vs', 'versus']):
            return QueryCommand(command="visualize", params={"viz_type": "correlation"})
        elif any(word in query_lower for word in ['distribution', 'histogram', 'hist']):
            return QueryCommand(command="visualize", params={"viz_type": "histogram"})
        elif any(word in query_lower for word in ['box', 'boxplot', 'quartile']):
            return QueryCommand(command="visualize", params={"viz_type": "boxplot"})
        elif any(word in query_lower for word in ['bar', 'count by', 'breakdown']):
            return QueryCommand(command="visualize", params={"viz_type": "bar"})
        elif any(word in query_lower for word in ['time', 'trend', 'over time']):
            return QueryCommand(command="visualize", params={"viz_type": "timeseries"})
        else:
            return QueryCommand(command="visualize", params={"viz_type": "auto"})
    
    # Check for specific visualization keywords without explicit plot/chart mention
    elif any(word in query_lower for word in ['correlation', 'correlate']) and any(word in query_lower for word in ['vs', 'versus', 'and']):
        return QueryCommand(command="visualize", params={"viz_type": "correlation"})
    
    # Direct pattern matching for common queries
    elif any(word in query_lower for word in ['total', 'sum', 'add up']):
        return QueryCommand(command="analyze", params={"agg_type": "sum"})
    elif any(word in query_lower for word in ['average', 'mean', 'avg']):
        if any(phrase in query_lower for phrase in ['per person', 'per customer', 'per user', 'each person', 'each customer']):
            return QueryCommand(command="analyze", params={"agg_type": "group_average", "group_by": "customer_id"})
        else:
            return QueryCommand(command="analyze", params={"agg_type": "average"})
    elif any(word in query_lower for word in ['count', 'number of', 'how many']):
        return QueryCommand(command="analyze", params={"agg_type": "count"})
    elif any(word in query_lower for word in ['random', 'sample', 'example']):
        return QueryCommand(command="analyze", params={"agg_type": "random_row"})
    elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
        return QueryCommand(command="summarize", params={})
    elif any(word in query_lower for word in ['compare', 'difference']):
        return QueryCommand(command="compare_files", params={})
    else:
        return QueryCommand(command="answer_question", params={"keyword": query})

# RAG Setup
def setup_rag(chunks: List[Dict]) -> any:
    """Set up RAG pipeline with Chroma for repository-wide retrieval."""
    if not chunks:
        return None
        
    try:
        texts = [chunk["text"] for chunk in chunks if chunk["metadata"]["content_type"] != "image"]
        metadatas = [chunk["metadata"] for chunk in chunks if chunk["metadata"]["content_type"] != "image"]
        
        embeddings = OllamaEmbeddings(model='nomic-embed-text')
        temp_dir = tempfile.mkdtemp()
        vectorstore = Chroma.from_texts(
            texts, 
            embeddings, 
            metadatas=metadatas, 
            persist_directory=temp_dir
        )
        return vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        st.error(f"Error setting up RAG: {str(e)}")
        return None

# Response Generation
def generate_response(chunks: List[Dict], query: str, command: QueryCommand) -> ChatbotResponse:
    """Generate a response using RAG, CSV analysis, or image analysis."""
    try:
        # Handle image analysis
        if command.command == "analyze_image":
            # Filter chunks for images
            image_chunks = [chunk for chunk in chunks if chunk["metadata"]["content_type"] == "image"]
            if not image_chunks:
                return ChatbotResponse(answer="No images found in the documents.", sources=[])
            
            # Apply filters based on params
            file_name = command.params.get("file_name")
            page = command.params.get("page")
            filtered_chunks = image_chunks
            if file_name:
                filtered_chunks = [c for c in filtered_chunks if c["metadata"]["file_name"] == file_name]
            if page:
                filtered_chunks = [c for c in filtered_chunks if c["metadata"]["page"] == page]
            
            if not filtered_chunks:
                return ChatbotResponse(answer="No matching images found for the specified criteria.", sources=[])
            
            # Analyze the first matching image
            image_chunk = filtered_chunks[0]
            image_path = image_chunk["metadata"]["image_path"]
            sources = [image_chunk["metadata"]["file_name"]]
            answer = analyze_image(image_path, query)
            return ChatbotResponse(answer=answer, sources=sources)
        
        # Handle CSV analysis and visualization commands
        if command.command in ["analyze", "visualize"]:
            csv_files = [chunk["metadata"]["file_name"] for chunk in chunks if chunk["metadata"]["file_type"] == "csv"]
            if not csv_files:
                return ChatbotResponse(answer="No CSV files found for analysis.", sources=[])
            
            file_name = csv_files[0]
            file_path = None
            if 'file_paths' in st.session_state and file_name in st.session_state.file_paths:
                file_path = st.session_state.file_paths[file_name]
            else:
                temp_path = f"./temp/{file_name}"
                if os.path.exists(temp_path):
                    file_path = temp_path
            
            if not file_path or not os.path.exists(file_path):
                return ChatbotResponse(answer=f"Could not find file {file_name} for analysis.", sources=[file_name])
            
            try:
                answer, chart, chart_type = analyze_csv(file_path, query, command)
                return ChatbotResponse(answer=answer, sources=[file_name], chart=chart, chart_type=chart_type)
            except Exception as e:
                return ChatbotResponse(answer=f"Error analyzing {file_name}: {str(e)}", sources=[file_name])
        
        # Handle other commands with RAG
        retriever = setup_rag(chunks)
        if not retriever:
            return ChatbotResponse(answer="Error setting up document retrieval.", sources=[])
        
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            if not relevant_docs:
                return ChatbotResponse(answer="No relevant information found in the documents.", sources=[])
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            sources = list(set(doc.metadata.get("file_name", "unknown") for doc in relevant_docs))
            
            prompt = f"""Based on the following document context, answer the user's question concisely and accurately.

Context:
{context}

Question: {query}

Answer based only on the information provided in the context. If the information is not available, say so."""
            
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_0',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            answer = response['message']['content']
            return ChatbotResponse(answer=answer, sources=sources)
            
        except Exception as e:
            return ChatbotResponse(answer=f"Error generating response: {str(e)}", sources=[])
            
    except Exception as e:
        return ChatbotResponse(answer=f"Error processing query: {str(e)}", sources=[])

# Query Handling
def handle_query(chunks: List[Dict], query: str) -> ChatbotResponse:
    """Handle a user query by extracting the command and generating a response."""
    if not chunks:
        return ChatbotResponse(answer="No documents loaded. Please upload documents first.", sources=[])
    
    try:
        command = extract_command(query)
        response = generate_response(chunks, query, command)
        return response
    except Exception as e:
        return ChatbotResponse(answer=f"Error processing query: {str(e)}", sources=[])

# Streamlit UI
def main():
    st.title("Local Document Repository RAG Chatbot")
    
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
    if 'file_paths' not in st.session_state:
        st.session_state.file_paths = {}
    
    os.makedirs("./temp", exist_ok=True)

    st.write("Upload documents and ask questions about your data. Examples:")
    st.write("- 'What is the total amount in the transaction data?'")
    st.write("- 'Average transaction amount'")
    st.write("- 'When does the contract expire?'")
    st.write("- 'Show me a random row from the data'")
    st.write("- 'Summarize the document'")
    st.write("- 'Describe the chart on page 2 of report.pdf'")
    
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=['xlsx', 'xls', 'csv', 'pdf', 'txt'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            file_path = f"./temp/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.read())
            st.session_state.file_paths[file.name] = file_path
            file_name, chunks = load_document(file_path)
            if chunks:
                st.session_state.all_chunks = [
                    chunk for chunk in st.session_state.all_chunks 
                    if chunk["metadata"]["file_name"] != file_name
                ]
                st.session_state.all_chunks.extend(chunks)
                st.success(f"Loaded {file_name} ({len(chunks)} chunks)")
            else:
                st.error(f"Failed to load {file_name}")
    
    if st.session_state.all_chunks:
        loaded_files = list(set(chunk["metadata"]["file_name"] for chunk in st.session_state.all_chunks))
        st.write(f"**Loaded files**: {', '.join(loaded_files)}")
    
    query = st.text_input("Ask about your documents:")
    if query:
        if not st.session_state.all_chunks:
            st.warning("Please upload documents before asking questions.")
        else:
            with st.spinner("Processing query..."):
                try:
                    response = handle_query(st.session_state.all_chunks, query)
                    st.write(f"**Answer**: {response.answer}")
                    if response.sources:
                        st.write(f"**Sources**: {', '.join(response.sources)}")
                    if response.chart:
                        st.pyplot(response.chart)
                        st.write(f"**Chart Type**: {response.chart_type}")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()