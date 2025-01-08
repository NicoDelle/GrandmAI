#!/usr/bin/env bash
ollama serve &
sleep 5
#ollama pull qwen2.5:14b
ollama create grandmai -f /Modelfile
ollama pull nomic-embed-text

if [ $? -eq 0 ]; then
    echo "Model pulled successfully."
else
    echo "Failed to pull the model."
    exit 1
fi

wait