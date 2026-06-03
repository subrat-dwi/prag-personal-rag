# PRAG - Personal RAG System

## What it does?
- uses RAG to answer questions querying personal facts and figures.
- a utility to quickly and **safely** access data like Aadhaar No., PAN No., Marks, Personal Preferences, etc

## Working
1. Tracks a Dedicated Folder.
2. Extracts and embeds data from files on change in the folder.
3. Stores Embedded chunks in a Vector DB.
4. On user query, uses similarity search to get similar data chunks.
5. Feeds the chunks along with the query to the AI model.
6. Model gives the appropriate response to the query.

## Tools and Tech Used
```
Folder Watcher: watchdog
File Parsers:
	pypdf
	python-docx
	pillow + pytesseract
Chunker: RecursiveCharacterTextSplitter
Embeddings: nomic-embed-text
VectorDB: Qdrant, then Qdrant Cloud
LLM:
	Local: Qwen2.5:3b
```