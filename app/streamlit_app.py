import os
import streamlit as st
import base64
from utils import *
from dotenv import load_dotenv
load_dotenv()
# '''
# 5. Print response and highlight chunks on pdf
# '''

# Initialize keys in session state if it doesn't exist
if 'retriever' not in st.session_state:
    st.session_state.retriever = ''
if 'user_query' not in st.session_state:
    st.session_state.user_query = ''
if 'init_retriever' not in st.session_state:
    st.session_state.init_retriever = ''
if 'response' not in st.session_state:
    st.session_state.response = ''
if 'chunks' not in st.session_state:
    st.session_state.chunks = ''
if 'sep_chunks' not in st.session_state:
    st.session_state.sep_chunks = ''
if 'query_llm' not in st.session_state:
    st.session_state.query_llm = ''

def clean_response():
    st.session_state.response = ''


def get_chunk_vis():
    with col2:     
        with st.expander('Visulize Retrieved Chunks'):  
            if st.session_state.response:
                if len(st.session_state.response['context']['chunks'])>0:
                    # st.write(st.session_state.response['context']['chunks'])
                    chuck_len = len(st.session_state.response['context']['chunks'])-1
                    st.number_input(label=f'Index of the chunk to visualize out of {chuck_len} chunks', 
                                            min_value=0,
                                            # value=None,
                                            key='query_chunk_idx',
                                            max_value=chuck_len)

                    # if st.session_state.query_chunk_idx is not None:
                    # st.write(st.session_state.response['context']['chunks'][st.session_state.query_chunk_idx])
                    display_chunk(st.session_state.mmpdf, st.session_state.response['context']['chunks'][st.session_state.query_chunk_idx])


def load_streamlit_page():

    """
    Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key
    and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_file: The uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")
    #st.header('LLM Tool\t')
    style_heading = 'text-align: center'
    st.markdown(f"<h1 style='{style_heading}'>MM RAG Tool for PDF Analysis</h1>", unsafe_allow_html=True)

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf", key="mmpdf")

    return col1, col2, uploaded_file

# Make a streamlit page
col1, col2, uploaded_file = load_streamlit_page()



# Process the PDF

if st.session_state.mmpdf is not None:
    st.session_state.chunks = ''

    # if 'pdf_key' not in st.session_state:
    #     st.session_state.pdf_key = uploaded_file
    with col1:
        with st.spinner('Processing PDF in chunks'):
            if not st.session_state.chunks:
                st.session_state.chunks = get_chunks(st.session_state.mmpdf)
            
            if not st.session_state.sep_chunks:
                st.session_state.sep_chunks = seperate_chunks(st.session_state.chunks) # texts, images, tables, df 
        with st.expander('Analyse Chunks'):
            if st.session_state.sep_chunks:
                st.write(f'Number of Chunks {len(st.session_state.chunks)}')
                st.dataframe(st.session_state.sep_chunks[-1], hide_index=True, width=300)
    
    with col2:
        if st.session_state.chunks:   
            with st.expander('Visulize Chunks'):
            #with col2.container(height=700):
                chuck_len = len(st.session_state.chunks)-1
                st.number_input(label=f'Index of the chunk to visualize out of {chuck_len} chunks', 
                                        min_value=0,
                                        value=None,
                                        key='chunk_idx',
                                        max_value=chuck_len)
                if st.session_state.chunk_idx:
                    display_chunk(st.session_state.mmpdf, st.session_state.chunks[st.session_state.chunk_idx])
            
       
    with col1:
    # Create database
        if (not st.session_state.retriever) and st.session_state.sep_chunks:
            with st.spinner('Creating Vector DB'):
                st.session_state.retriever = vectorize_chunks(*st.session_state.sep_chunks[:3])
            
        
        if st.session_state.retriever:
            st.session_state.user_query = st.text_input('User query', "What is multihead Attention?", on_change=clean_response)
            if (not st.session_state.response) and (st.session_state.user_query):
                st.session_state.response = get_response(st.session_state.user_query, st.session_state.retriever)
            submit = st.button('Submit')#, args=(col2,))
            if st.session_state.response and submit :
                st.write(st.session_state.response['response'])
                # st.write(st.session_state.retriever.invoke(st.session_state.user_query))
                # st.write(st.session_state.response['context'])
                get_chunk_vis()
                # st.write('call triggered')
                # st.session_state.query_llm = True
            # if st.button
            #     st.session_state.response = ''