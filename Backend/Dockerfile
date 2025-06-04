# Use the official Ollama image
FROM ollama/ollama

# Preload the Phi-3 model
RUN ollama pull phi3:3.8b

# Expose default Ollama API port
EXPOSE 11434

# Start the Ollama API server
CMD ["ollama", "serve"]