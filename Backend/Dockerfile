FROM ollama/ollama:latest

# Copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh

# Expose the default Ollama API port
EXPOSE 8080

# Run the entrypoint script when container starts
ENTRYPOINT ["/entrypoint.sh"]
