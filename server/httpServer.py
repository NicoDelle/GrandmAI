from flask import Flask, request, jsonify
import fitz
import requests
import logging
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

OLLAMA_API_URL = "http://ollama:11434/api/"
FAISS_HOST = os.getenv("FAISS_HOST", "faiss")
FAISS_PORT = os.getenv("FAISS_PORT", "6000")
MODEL_NAME = "qwen2.5:14b"
EMBEDDINGS_MODEL = "nomic-embed-text"

MAX_CHUNK_LENGTH = 512
CHUNKS_FOLDER = "/server/chunks"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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

def chunk_text(text, max_length=MAX_CHUNK_LENGTH):
    tokens = text.split()
    for i in range(0, len(tokens), max_length):
        chunk = " ".join(tokens[i:i + max_length])
        yield chunk
        

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        userPrompt = data.get('prompt')
        app.logger.debug(f"Received prompt: {userPrompt}")

        response = requests.post(
            OLLAMA_API_URL + "embeddings", 
            json={"model": EMBEDDINGS_MODEL, "prompt": userPrompt}
        )
        response.raise_for_status()
        prompt_embedding = response.json()['embedding']

        search_response = requests.post(
            f"http://{FAISS_HOST}:{FAISS_PORT}/search", 
            json={"query": [prompt_embedding], "k": 10}
        )
        search_response.raise_for_status()
        search_results = search_response.json()
        app.logger.debug(f"Search results: {search_results}")

        # Retrieve relevant chunks
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

        super_chunks_embeddings = []
        for super_chunk in super_chunks:
            chunk_embedding_response = requests.post(
                OLLAMA_API_URL + "embeddings", 
                json={"model": EMBEDDINGS_MODEL, "prompt": super_chunk}
            )
            chunk_embedding_response.raise_for_status()
            super_chunks_embeddings.append(chunk_embedding_response.json()['embedding'])

        # Compute similarity scores
        similarity_scores = cosine_similarity([prompt_embedding], super_chunks_embeddings)[0]

        # Sort super-chunks by similarity scores
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_super_chunks = [super_chunks[i] for i in sorted_indices]

        # Combine the top super-chunks
        relevant_chunks = "\n".join(sorted_super_chunks[:10])
        
        prompt = f"""Rispondi in italiano alla seguente domanda,
            basandoti sulle tue conoscenze e soprattutto sul contesto fornito di seguito.
            Contesto: {relevant_chunks}.
            Domanda: {userPrompt}
            """

        # Send prompt to model server
        response = requests.post(
            OLLAMA_API_URL + "generate", 
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        model_response = response.json()
        app.logger.debug(f"Model response: {model_response}")
        
        return jsonify(model_response)
    
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/getEmbeddings', methods=['POST'])
def getEmbeddings():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the uploaded file
        file_path = "/tmp/" + file.filename
        file.save(file_path)
        
        # Extract text from the PDF
        text = extract_text_from_pdf(file_path)
        
        # Split text into chunks
        chunks = chunk_text(text)
        embeddings = []
        for i, chunk in enumerate(chunks):
            store_chunk(chunk, i, file_path)
            response = requests.post(
                OLLAMA_API_URL + "embeddings",
                json={"model": EMBEDDINGS_MODEL, "prompt": chunk}
            )
            response.raise_for_status()
            model_response = response.json()
            embeddings.append(model_response)

        # Send embeddings to FAISS service
        vectors = [embedding['embedding'] for embedding in embeddings]
        response = requests.post(
            f"http://{FAISS_HOST}:{FAISS_PORT}/add_embeddings", 
            json={"embeddings": vectors}
        )
        response.raise_for_status()

        app.logger.debug(f"Model response: {embeddings}")
        
        return jsonify({"status": "File processed and embeddings saved successfully"})
    
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)