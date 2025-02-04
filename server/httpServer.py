from flask import Flask, request, jsonify, render_template
import requests, logging, os, retriever
from typing import List

app = Flask(__name__)
UPLOAD_FOLDER = '/server/uploads'
OUTPUT_FOLDER = '/server/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

OLLAMA_API_URL = "http://ollama:11434/api/"
FAISS_HOST = os.getenv("FAISS_HOST", "faiss")
FAISS_PORT = os.getenv("FAISS_PORT", "6000")
MODEL_NAME = "grandmai" #"qwen2.5:14b"
EMBEDDINGS_MODEL = "bge-m3" #"granite-embedding:278m" #"nomic-embed-text"
CTX_LEN = 20000

CONTEX_SENT_SIZE = 9

def get_embedding(text: str) -> List[float]:
    response = requests.post(
        OLLAMA_API_URL + "embeddings", 
        json={"model": EMBEDDINGS_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()['embedding']

def get_k_matches(query_embedding: List[float], k: int = 10) -> List[int]:
    search_response = requests.post(
        f"http://{FAISS_HOST}:{FAISS_PORT}/search", 
        json={"query": [query_embedding], "k": 10}
    )
    search_response.raise_for_status()
    search_results = search_response.json()
    app.logger.debug(f"Search results: {search_results}")

    return search_results["indices"]

def store_embeddings(embeddings: List[List[float]]) -> None:
    response = requests.post(
        f"http://{FAISS_HOST}:{FAISS_PORT}/add_embeddings", 
        json={"embeddings": embeddings}
    )
    response.raise_for_status()

def send_prompt(prompt: str) -> dict:
    response = requests.post(
        OLLAMA_API_URL + "generate", 
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return 'File successfully uploaded'

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        userPrompt = data.get('prompt')
        app.logger.debug(f"Received prompt: {userPrompt}")

        #Obtian the 10 best matches for the given promptfrom the FAISS index
        prompt_embedding = get_embedding(userPrompt)
        search_results = get_k_matches(prompt_embedding, 10)
        context = retriever.get_context(search_results)
        

        prompt = f"""Rispondi in italiano alla seguente domanda in modo preciso e fornendo tutte le informazioni necessarie. Rispondi,
            basandoti soprattutto sul contesto fornito di seguito, e sulle tue conoscenze se mancassero informazioni cruciali.
            Domanda: {userPrompt}\n
            Contesto: {context}.
            """
        # Send prompt to model server
        model_response = send_prompt(prompt)
        
        return jsonify(model_response)
    
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/getEmbeddings', methods=['GET'])
def getEmbeddings():
    files_to_embed = os.listdir(app.config['UPLOAD_FOLDER'])
    if len(files_to_embed) == 0:
        return jsonify({"error": "No file to embed"}), 400
    
    for file in files_to_embed:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        raw_text = retriever.extract_text_from_file(file_path)
        chunks = retriever.chunk_text(raw_text)

        embeddings = []
        last_id = retriever.get_next_id()
        for i, chunk in enumerate(chunks):
            retriever.store_chunk(chunk, file_path, last_id + i)
            app.logger.debug(f"Sending request for chunk {i}: {chunk}")
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
        retriever.update_last_id(last_id + len(embeddings))
        
        # Send embeddings to FAISS service
        app.logger.debug(f"embeddings size: {len(embeddings)}")
        store_embeddings(embeddings)

        os.remove(file_path)
    
    return jsonify({"status": "Files processed and embeddings saved successfully"})

@app.route('/summarize', methods=['GET'])
def summarize():
    files_to_summarize = os.listdir(app.config['UPLOAD_FOLDER'])
    if len(files_to_summarize) == 0:
        return jsonify({"error": "No file to summarize"}), 400
    
    text = ""
    len_text = 0
    for file in files_to_summarize:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        raw_text = retriever.extract_text_from_file(file_path)
        len_text += len(raw_text)

        text += raw_text + "\n"

    chunk_size = 10000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    
    for chunk in chunks:
        prompt = f"""
        Dividi il testo trascritto della lezione in paragrafi tematici, ciascuno dedicato a un singolo argomento teorico. 
        Per ogni paragrafo:
        1. Identifica il **concetto teorico principale** (modelli, teoremi, definizioni, framework)
        2. Spiega in modo conciso ma completo:
        - La definizione/formulazione tecnica
        - Le ipotesi o premesse fondamentali
        - La rilevanza nel contesto della disciplina
        3. Utilizza un linguaggio specialistico appropriato ma chiaro
        4. Escludi dettagli non teorici (esempi pratici, aneddoti, divagazioni)

        Struttura di output desiderata:
        **Paragrafo [N]**
        - **Concetto chiave:** [Nome teoria/legge/modello]
        - **Definizione:** [Formalizzazione sintetica]
        - **Contesto d'uso:** [A quali problemi/procedimenti si applica]
        - **Rilevanza:** [Perché è importante]        
        Il testo è il seguente:\n{chunk}
        """
        model_response = send_prompt(prompt)
        summaries.append(model_response["response"])
    
    final_summary = "\n".join(summaries)

    # Write summary to file
    summary_file_path = os.path.join(app.config['OUTPUT_FOLDER'], "summary.txt")
    with open(summary_file_path, "w") as f:
        f.write(final_summary)

    return jsonify({"summary": model_response["response"], "file_path": summary_file_path})

@app.route('/clearUploads', methods=['GET'])
def clearUploads():
    files_to_clear = os.listdir(app.config['UPLOAD_FOLDER'])
    for file in files_to_clear:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)
    
    return jsonify({"status": "Uploads folder cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)