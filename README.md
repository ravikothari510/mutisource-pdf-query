## Objectives
This repo is inspired from the project [Multimodal RAG: Chat with PDFs](https://www.youtube.com/watch?v=uLrReyH5cu0). The aim is to deconstruct the jupyter notebook into python script, formalize the environment setup and develop a UI app with stremlit.

## Introduction
This is a base example for MultiModal (Images, Text) RAG pipeline for analysing scientific papers. A MultiModal RAG we need a document parser (Unstructured), preprocesser (eg: text summerization, image captioning), embedder (GPT4o-LLM), Retriever (ChromaDB) and response (GPT4o-LLM). The prompt linking is done with LangChain.\

The most important part of any RAG is data parser. For this project we use Unstructured library, this extracts individual elements from a pdf (Images, Tabells and Text), it also supports other formats such as "ppt", "csv" , "doc".

![MMLLM Pipelime](assest/pipeline.jpg?raw=true "Optional Title")
As the above pipeline overview shows not all the components are MultiModal, for e.g. we only embed image summary as opposed to raw images in Vector DB, similarly the original repo used these image and Table summary as a context for a user query. In an advance RAG pipeline, a classifier should seperate chunks in base-image, graphs, flow diagram, text and Tables. These should should be processed by specialized LLMs for eg [Google deplot](https://huggingface.co/google/deplot) is good at processing plots, [PandasQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/pandas_query_engine/) for Tabular data, etc. 

## Environment Setup
I have tested this in WSL Ubuntu 22.04 and python3.10, so all the floowing steps are based on this. \
Working with **Unstructured** is not straight forward, I wanted to setup this project in remote pc, but this requres installing dependencies with root access :cry:. Next step was to use the docker image of Unstructured, but VS code doesn't allow to use this as a dev container. That's WSL is used to setup this project.\
1. Install system dependencies 
```apt-get update && apt-get install libgl1 poppler-utils tesseract-ocr libmagic-dev ```
2. Create and activate virtual env\
```python3 -m venv /path/to/venv ```\
``` source /path/to/venv/bin/activate```\
``` python -m pip install --upgrade pip```
3. Install pip packages (always use requirements.txt as pip can resolve dependencies)\
``` pip install -r requirements.txt```

## TODO
- [ ] Replace WSL with Docker image  
- [x] Streamlit App
- [ ] Visualise Chunks in streamlit app
- [ ] Multimodal Context
