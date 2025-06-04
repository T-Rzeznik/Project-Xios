#!/bin/sh
ollama serve &

sleep 30

ollama pull phi3
kill %1

exec ollama serve
