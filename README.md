# Local Document RAG Chatbot

This project is an offline Retrieval-Augmented Generation (RAG) chatbot designed to process and analyze user-uploaded documents, including Excel, CSV, PDF, and TXT files. Built with open-source tools, it enables privacy-focused document querying and data analysis without internet dependency. The chatbot is under active development, with ongoing efforts to enhance functionality, improve performance, and expand use cases. It is intended for developers, researchers, and data analysts seeking a customizable, local solution for document-based question answering and data visualization.

**Important**: This project is a work in progress. Expect evolving features and potential bugs as development continues.

## Project Objectives

The Local Document RAG Chatbot aims to address the following goals:
- **Privacy and Security**: Process sensitive documents (e.g., financial, legal, or proprietary data) entirely on the user's machine, ensuring no data is transmitted externally.
- **Offline Functionality**: Enable document analysis in environments without internet access, such as secure networks or remote locations.
- **Versatile Document Processing**: Support multiple file formats (Excel, CSV, PDF, TXT) with capabilities for summarization, question answering, and data analysis.
- **Data Visualization**: Provide basic statistical analysis and visualizations (e.g., histograms, scatter plots) for tabular data in CSVs and Excel files.
- **Extensibility**: Offer a modular, open-source framework for developers to customize and extend for specific use cases.

## Current Features

- **Document Parsing**: Processes Excel, CSV, PDF, and TXT files using PyMuPDF and Pandas, extracting structured content (e.g., tables, text) into manageable chunks.
- **Query Capabilities**:
  - Summarize document content (e.g., "Summarize the contract").
  - Answer specific questions based on document text (e.g., "What is the contract's expiration date?").
  - Perform statistical analysis on tabular data (e.g., sum, average, count, random row sampling).
  - Generate visualizations for CSV data, including histograms, scatter plots, box plots, and bar charts.
- **RAG Pipeline**: Combines Ollama (Mistral 7B) for local language model inference with Chroma for vector-based document retrieval.
- **Streamlit Interface**: Provides a user-friendly web interface for file uploads and query input.
- **Local Operation**: Runs entirely offline, ensuring data privacy and suitability for air-gapped environments.

## Approach and Implementation

### Tools and Libraries
- **Document Processing**: `PyMuPDF` for PDF text extraction, `Pandas` for handling tabular data in Excel and CSV files.
- **Data Validation**: `Pydantic` for structured query and response handling.
- **RAG Components**: `Ollama` (Mistral 7B) for local LLM inference, `langchain-ollama` and `langchain-chroma` for embedding and retrieval.
- **Visualization**: `Matplotlib`, `Seaborn`, and `Plotly` for generating charts from tabular data.
- **Interface**: `Streamlit` for an interactive web-based UI.
- **Text Splitting**: `langchain-text-splitters` for chunking large documents to fit memory constraints.

### Methodology
- Documents are parsed into chunks (1000 characters with 200-character overlap) and indexed in a temporary Chroma vector store for efficient retrieval.
- Queries are parsed into structured commands (e.g., summarize, analyze, visualize) using regex-based keyword matching.
- For tabular data, Pandas processes CSV/Excel files to compute statistics or generate visualizations based on query intent.
- Responses are generated via Mistral 7B, constrained to the provided document context to minimize hallucinations.
