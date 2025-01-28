import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import os

MAX_CHUNK_LENGTH = 64 #IT's A PROBLEM OF CONTEXT LENGTH: FIND A SUITABLE LENGTH!
CHUNK_OVERLAP_LENGTH = 16
CONTEX_SENT_SIZE = 9
CHUNKS_FOLDER = "/server/chunks"
OLLAMA_API_URL = "http://ollama:11434/api/"
EMBEDDINGS_MODEL = "granite-embedding:278m" #"nomic-embed-text"


def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    text = ""
    
    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    
    return text

def store_chunk(chunk, index, pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    chunk_filename = os.path.join(CHUNKS_FOLDER, f"{pdf_name}.{index}.txt")
    with open(chunk_filename, 'w') as chunk_file:
        chunk_file.write(chunk)

def load_chunk(index):
    pdf_name = "DispensaLinux" # HARDCOED FOR NOW, NEED SOME WAY TO GROUP FILES BY SUBJECT
    chunk_filename = os.path.join(CHUNKS_FOLDER, f"{pdf_name}.{index}.txt")
    with open(chunk_filename, "r") as chunk_file:
        return chunk_file.read()

def chunk_text(text, max_length=MAX_CHUNK_LENGTH, overlap=CHUNK_OVERLAP_LENGTH):
    tokens = text.split()
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i + max_length])
        i += max_length - overlap
        yield chunk

def get_context(search_results: List[str]):
    super_chunks = []
    for idx in search_results['indices'][0]:
        pre_chunk = load_chunk(idx-1) if idx > 0 else ""
        chunk = load_chunk(idx)  # Retrieve chunk by index
        post_chunk = ""
        try:
            post_chunk = load_chunk(idx+1)
        except FileNotFoundError:
            pass
            
        super_chunks.append(pre_chunk + chunk + post_chunk + "\n")


    return "\n".join(super_chunks[:CONTEX_SENT_SIZE])