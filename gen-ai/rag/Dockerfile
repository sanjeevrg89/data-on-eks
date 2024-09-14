# Use the official ChromaDB image
FROM chromadb/chroma:latest

# Install additional Python libraries for CSV and PDF handling
RUN pip install pandas pdfplumber

# Copy the custom ChromaDB server script from the src directory
COPY src/chromadb/chromadb_server.py /app/chromadb_server.py

# Expose the port ChromaDB will run on
EXPOSE 8000

# Command to run the custom ChromaDB server
CMD ["python", "/app/chromadb_server.py"]
