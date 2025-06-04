FROM ollama/ollama:latest


COPY entrypoint.sh /entrypoint.sh


RUN chmod +x /entrypoint.sh


EXPOSE 8080


ENTRYPOINT ["/entrypoint.sh"]
