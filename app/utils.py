import os
import tempfile
from typing import Tuple
import uuid
from base64 import b64decode

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL.Image 
import pandas as pd

import fitz
import streamlit as st
from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
from langchain.storage import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
# https://stackoverflow.com/questions/77385587/persist-parentdocumentretriever-of-langchain


# Error handling https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



@st.cache_data
def get_chunks(uploaded_file):
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    
    try:
        # Read file content
        # input_file = uploaded_file.read()

        # # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        # temp_file = tempfile.NamedTemporaryFile(delete=False)
        # temp_file.write(input_file)
        # temp_file.close()
        
        chunks = partition_pdf(
            # filename=temp_file.name,
            file=uploaded_file,
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables

            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,

            # extract_images_in_pdf=True,          # deprecated
            )
        
        return chunks
        
        
    finally:
        # Ensure the temporary file is deleted when we're done with it
        # os.unlink(temp_file.name)
        pass

def seperate_chunks(chunks: list) -> Tuple:
    
    # Collect Tables, Texts and Images -> fed to summerizer
    tables = [] # as html
    images = [] # as base64
    texts = []  # str of a complete chunk

    el_dict = {}
    total_elements = 0

    for ch in chunks:
        texts.append(ch)
        for el in ch.metadata.orig_elements:
            total_elements += 1
            
            if el.category == 'Image':
                images.append(el.metadata.image_base64)
            elif el.category == 'Table':
                tables.append(el.metadata.text_as_html)
            else:
                pass
            
            if el.category not in el_dict:
                el_dict[el.category] = 1
            else:
                el_dict[el.category] += 1
    
    data = {'elements':[], 'count':[]}
    for key, value in el_dict.items():
        data['elements'].append(key)
        data['count'].append(value)
    
    df = pd.DataFrame.from_dict(data)
    
    return texts, tables, images, df

def display_chunk(uploaded_file, chunk):
    
    
    page_numbers = extract_page_no(chunk)

    docs = []
    for element in chunk.metadata.orig_elements:
        metadata = element.metadata.to_dict()
        if element.category == 'Image':
            metadata['category'] = 'Image'
        elif element.category == 'Table':
            metadata['category'] = 'Table'
        else:
            metadata['category'] = 'Text'
        metadata['page_number'] = int(element.metadata.page_number)
        
        docs.append(Document(page_content=element.text, metadata=metadata))
    
    for page_number in page_numbers:
        render_page(uploaded_file, docs, page_number, False)
    
    return None

def extract_page_no(chunk):
    elements = chunk.metadata.orig_elements
    page_numbers = set()
    
    for el in elements:
        page_numbers.add(el.metadata.page_number)
    
    return page_numbers

def render_page(uploaded_file, doc_list: list, page_number: int, print_text=True) -> None:
    # st.write(f'File size {uploaded_file.size}')
    uploaded_file.seek(0,0) # Important for multiple reads of ByteIO
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as f:
        pdf_page = f.load_page(page_number-1)
        # pdf_page = fitz.open(stream=uploaded_file.read(), filetype="pdf").load_page(page_number - 1)
        page_docs = [
            doc for doc in doc_list if doc.metadata.get("page_number") == page_number
        ]
        segments = [doc.metadata for doc in page_docs]
        plot_pdf_with_boxes(pdf_page, segments)

    return None

def plot_pdf_with_boxes(pdf_page, segments):
    pix = pdf_page.get_pixmap()
    pil_image = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height)
            for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(
            scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
        )
        ax.add_patch(rect)

    # Make legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(
                patches.Patch(color=category_to_color[category], label=category)
            )
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    # plt.show()
    st.pyplot(fig)
    
    return None

def summerizer(texts, tables, images):
    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
    # Summarize text
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 3})

    # Here we are alredy provifing the contect for the LLM (how to handel this automatically)
    prompt_template = """Describe the image in detail. For context,
                    the image is part of a research paper explaining the transformers
                    architecture. Be specific about graphs, such as bar plots."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

    image_summaries = chain.batch(images)
    
    return text_summaries, table_summaries, image_summaries

@st.cache_resource
def get_memstore():
    return InMemoryStore()

@st.cache_resource
def init_retriever():
    if not st.session_state.init_retriever:
        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(collection_name="multi_modal_rag",
                             persist_directory='./tmp/mmpdf',
                             embedding_function=OpenAIEmbeddings())

        # The storage layer for the parent documents
        store = get_memstore()
        # fs = LocalFileStore("./tmp/docs")
        # store = create_kv_docstore(fs)
        # store = LocalFileStore("./tmp/docs")
        id_key = "doc_id"

        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        
        st.session_state.init_retriever = retriever
        
    return st.session_state.init_retriever
     
# @st.cache_data
def vectorize_chunks (texts, tables, images):
    
    id_key = "doc_id"
    retriever = init_retriever()

    text_summaries, table_summaries, image_summaries = summerizer(texts, tables, images)
    
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    if len(summary_texts)>0:
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    if len(summary_tables) > 0:
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_imgs = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    
    if len(summary_imgs) > 0:
        retriever.vectorstore.add_documents(summary_imgs)
        retriever.docstore.mset(list(zip(img_ids, images)))
    
    return retriever

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    # st.write(f'docs parsing {len(docs)}')
    b64 = []
    text = []
    docs_ = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
            docs_.append(doc)
    return {"images": b64, "texts": text, "chunks": docs_}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


def get_response(query, retriever):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )
    )
    # response = chain.invoke(query)
    response = chain_with_sources.invoke(query)
    return response