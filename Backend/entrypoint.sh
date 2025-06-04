#!/bin/sh

echo "Pulling phi3:3.8b model..."
ollama pull phi3:3.8b

echo "Starting Ollama server..."
ollama serve --host 0.0.0.0 --port 8080
