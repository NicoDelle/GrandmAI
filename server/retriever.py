import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Generator
import os, transcribe

MAX_CHUNK_LENGTH = 512
CHUNK_OVERLAP_LENGTH = 128
CONTEX_SENT_SIZE = 5
CHUNKS_FOLDER = "/server/chunks"
LAST_ID_FILE = "last_id.txt"

def get_next_id() -> int:
    if not os.path.exists(LAST_ID_FILE):
        return 0
    with open(LAST_ID_FILE, "r") as file:
        return int(file.read().strip())

def update_last_id(last_id: int) -> None:
    with open(LAST_ID_FILE, "w") as file:
        file.write(str(last_id))

def extract_text_from_pdf(pdf_path: str) -> str:
    # Open the PDF file
    document = fitz.open(pdf_path)
    text = ""
    
    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    
    return text

def extract_text_from_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    if file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            return file.read()
    if file_path.endswith(".mp4") or file_path.endswith(".mp3"):
        return transcribe.transcribe_audio(file_path)

def store_chunk(chunk: str, pdf_path: str, id: int) -> int:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    chunk_filename = os.path.join(CHUNKS_FOLDER, f"{id}.{pdf_name}.txt")
    with open(chunk_filename, 'w') as chunk_file:
        chunk_file.write(chunk)
    
# @TODO: make this more efficient, and add a way to identify chunks based on tags
def load_chunk(index: int) -> str:
    for filename in os.listdir(CHUNKS_FOLDER):
        if filename.startswith(f"{index}."):
            with open(os.path.join(CHUNKS_FOLDER, filename), "r") as chunk_file:
                return chunk_file.read()

def chunk_text(text: str, max_length=MAX_CHUNK_LENGTH, overlap=CHUNK_OVERLAP_LENGTH) -> Generator[str, None, None]:
    tokens = text.split()
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i + max_length])
        i += max_length - overlap
        yield chunk

def merge_chunks(pre_chunk: str, chunk: str, post_chunk: str, overlap: int = CHUNK_OVERLAP_LENGTH) -> str:
    if pre_chunk:
        pre_chunk = pre_chunk.split()[:-overlap]
    if post_chunk:
        post_chunk = post_chunk.split()[overlap:]
    
    merged_chunk = " ".join(pre_chunk + chunk.split() + post_chunk)
    return merged_chunk

"""
Trying a different approach (using large chunks of texts by default, doesn't need merging)
def get_context(search_results: List[str]) -> str:
    super_chunks = []
    for idx in search_results['indices'][0]:
        pre_chunk = load_chunk(idx-1) if idx > 0 else ""
        chunk = load_chunk(idx)  # Retrieve chunk by index
        post_chunk = ""
        try:
            post_chunk = load_chunk(idx+1)
        except FileNotFoundError:
            pass
            
        super_chunks.append(merge_chunks(pre_chunk, chunk, post_chunk))

    return "\n".join(super_chunks[:CONTEX_SENT_SIZE])
"""

def get_context(search_results: List[int]) -> str:
    context = []
    for idx in search_results:
        chunk = load_chunk(idx)
        if chunk is not None:
            context.append(chunk)
    return "\n".join(context)