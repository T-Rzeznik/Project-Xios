#!/bin/sh

# Start the Ollama server in the background (default binds to all interfaces)
ollama serve &

# Wait a few seconds for the server to be ready (adjust if needed)
sleep 30

# Pull the model after the server is up
ollama pull phi3:3.8b

# Bring the server process to the foreground so container stays alive
fg %1