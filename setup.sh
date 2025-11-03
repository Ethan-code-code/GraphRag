#!/bin/bash

# Setup script for Knowledge Graph Ingestion

echo "Setting up Knowledge Graph Ingestion Environment..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Neo4j Configuration
NEO4J_AUTH=neo4j/ArifShopu1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ArifShopu1

# Application Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EOF
    echo ".env file created!"
else
    echo ".env file already exists, skipping..."
fi

# Create directories
echo "Creating data directories..."
mkdir -p neo4j-data/data
mkdir -p neo4j-data/logs
mkdir -p neo4j-data/import
mkdir -p neo4j-data/plugins
mkdir -p chroma_db

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Start Neo4j: docker-compose up -d neo4j"
echo "3. Run ingestion: docker-compose up ingest"
echo "   Or locally: python ingest.py /path/to/source/data"

