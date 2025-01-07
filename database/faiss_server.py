from flask import Flask, request, jsonify
import faiss
import numpy as np
import logging
import traceback
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FAISS index
d = 768  # Dimension of your embeddings
index_file = '/faiss_index/faiss_index.bin'

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    app.logger.info("FAISS index loaded from disk")
else:
    index = faiss.IndexFlatL2(d)  # L2 distance index
    app.logger.info("FAISS index initialized")

gpu_resources = faiss.StandardGpuResources()
app.logger.info("FAISS index initialized and moved to GPU")

@app.route('/add_embeddings', methods=['POST'])
def add_embeddings():
    try:
        data = request.json
        if not data or 'embeddings' not in data:
            raise ValueError("Invalid input: 'embeddings' field is required")
        
        embeddings = np.array(data['embeddings']).astype('float32')
        index.add(embeddings)
        app.logger.info(f"Embeddings added: {embeddings}")

        # Save the index to disk
        faiss.write_index(index, index_file)
        app.logger.info("FAISS index saved to disk")
        
        return jsonify({"status": "Embeddings added successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error adding embeddings: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        if not data or 'query' not in data:
            raise ValueError("Invalid input: 'query' field is required")
        
        query = np.array(data['query']).astype('float32')
        k = data.get('k', 5)  # Number of nearest neighbors to return
        distances, indices = index.search(query, k)
        app.logger.info(f"Search results - Distances: {distances}, Indices: {indices}")
        return jsonify({"distances": distances.tolist(), "indices": indices.tolist()}), 200
    except Exception as e:
        app.logger.error(f"Error searching embeddings: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)