from flask import Flask, request, jsonify
import requests
import logging
import os
import retriever

app = Flask(__name__)

OLLAMA_API_URL = "http://ollama:11434/api/"
FAISS_HOST = os.getenv("FAISS_HOST", "faiss")
FAISS_PORT = os.getenv("FAISS_PORT", "6000")
MODEL_NAME = "grandmai" #"qwen2.5:14b"
EMBEDDINGS_MODEL = "granite-embedding:278m" #"nomic-embed-text"


CONTEX_SENT_SIZE = 9

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        userPrompt = data.get('prompt')
        app.logger.debug(f"Received prompt: {userPrompt}")

        #get the embedding of the prompt
        response = requests.post(
            OLLAMA_API_URL + "embeddings", 
            json={"model": EMBEDDINGS_MODEL, "prompt": userPrompt}
        )
        response.raise_for_status()
        prompt_embedding = response.json()['embedding']

        #Obtian the 10 best matches from the FAISS index
        search_response = requests.post(
            f"http://{FAISS_HOST}:{FAISS_PORT}/search", 
            json={"query": [prompt_embedding], "k": 10}
        )
        search_response.raise_for_status()
        search_results = search_response.json()
        app.logger.debug(f"Search results: {search_results}")

        # Process the chunks obtained by the search into the final context
        context = retriever.get_context(search_results)
        

        prompt = f"""Rispondi in italiano alla seguente domanda,
            basandoti sulle tue conoscenze e soprattutto sul contesto fornito di seguito.
            Contesto: {context}.
            Domanda: {userPrompt}
            """
        # Send prompt to model server
        response = requests.post(
            OLLAMA_API_URL + "generate", 
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        model_response = response.json()
        
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
        text = retriever.extract_text_from_pdf(file_path)
        
        # Split text into chunks
        chunks = retriever.chunk_text(text)
        embeddings = []
        for i, chunk in enumerate(chunks):
            retriever.store_chunk(chunk, i, file_path)
            app.logger.debug(f"Sending request for chunk {i}: {chunk}")
            response = requests.post(
                OLLAMA_API_URL + "embeddings",
                json={"model": EMBEDDINGS_MODEL, "prompt": "a"*513}
            )
            app.logger.debug(f"Response status code: {response.status_code}")
            app.logger.debug(f"Response content: {response.content}")
            response.raise_for_status()
            model_response = response.json()
            embeddings.append(model_response)

        # Send embeddings to FAISS service
        app.logger.debug(f"embeddings size: {len(embeddings)}")
        vectors = [embedding['embedding'] for embedding in embeddings]
        response = requests.post(
            f"http://{FAISS_HOST}:{FAISS_PORT}/add_embeddings", 
            json={"embeddings": vectors}
        )
        response.raise_for_status()
        
        return jsonify({"status": "File processed and embeddings saved successfully"})
    
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)