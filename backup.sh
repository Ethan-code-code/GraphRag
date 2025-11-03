#!/bin/bash

# Backup script for ChromaDB and Neo4j databases
# Usage: ./backup.sh [chromadb|neo4j|all]

set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_TYPE="${1:-all}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Database Backup Script"
echo "=========================================="

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to backup ChromaDB
backup_chromadb() {
    echo -e "${YELLOW}Backing up ChromaDB...${NC}"
    
    if [ ! -d "./chroma_db" ]; then
        echo -e "${RED}Error: ChromaDB directory not found${NC}"
        return 1
    fi
    
    BACKUP_PATH="$BACKUP_DIR/chromadb_backup_$TIMESTAMP"
    
    # Copy ChromaDB directory
    cp -r ./chroma_db "$BACKUP_PATH"
    
    # Create metadata
    cat > "$BACKUP_PATH/backup_metadata.json" << EOF
{
  "backup_type": "chromadb",
  "timestamp": "$TIMESTAMP",
  "source_path": "./chroma_db",
  "backup_date": "$(date -Iseconds)"
}
EOF
    
    # Compress
    tar -czf "${BACKUP_PATH}.tar.gz" -C "$BACKUP_DIR" "chromadb_backup_$TIMESTAMP"
    rm -rf "$BACKUP_PATH"
    
    echo -e "${GREEN}✅ ChromaDB backup completed: ${BACKUP_PATH}.tar.gz${NC}"
}

# Function to backup Neo4j (requires Docker)
backup_neo4j() {
    echo -e "${YELLOW}Backing up Neo4j...${NC}"
    
    CONTAINER_NAME="neo4j-graphdb"
    BACKUP_FILE="$BACKUP_DIR/neo4j_backup_$TIMESTAMP.dump"
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Error: Neo4j container '$CONTAINER_NAME' is not running${NC}"
        return 1
    fi
    
    # Create dump inside container
    echo "Creating Neo4j dump..."
    docker exec "$CONTAINER_NAME" neo4j-admin database dump neo4j --to-path=/tmp/neo4j_backup.dump 2>/dev/null || {
        # Alternative: use neo4j-admin dump without database name
        docker exec "$CONTAINER_NAME" neo4j-admin dump --database=neo4j --to=/tmp/neo4j_backup.dump 2>/dev/null || {
            echo -e "${YELLOW}Warning: neo4j-admin dump failed. Trying alternative method...${NC}"
            # Try using cypher export via Python script
            python3 backup.py --type neo4j
            return 0
        }
    }
    
    # Copy dump from container
    docker cp "$CONTAINER_NAME:/tmp/neo4j_backup.dump" "$BACKUP_FILE" 2>/dev/null || {
        # Try alternative location
        docker cp "$CONTAINER_NAME:/data/dumps/neo4j.dump" "$BACKUP_FILE" 2>/dev/null || {
            echo -e "${YELLOW}Warning: Could not copy dump file. Using Python export instead.${NC}"
            python3 backup.py --type neo4j
            return 0
        }
    }
    
    # Compress
    gzip "$BACKUP_FILE"
    
    echo -e "${GREEN}✅ Neo4j backup completed: ${BACKUP_FILE}.gz${NC}"
}

# Function to backup Neo4j data directory (alternative)
backup_neo4j_data() {
    echo -e "${YELLOW}Backing up Neo4j data directory...${NC}"
    
    if [ ! -d "./neo4j-data/data" ]; then
        echo -e "${YELLOW}Warning: Neo4j data directory not found${NC}"
        return 0
    fi
    
    BACKUP_PATH="$BACKUP_DIR/neo4j-data_backup_$TIMESTAMP"
    
    # Stop Neo4j to ensure consistent backup (optional, commented out)
    # echo "Stopping Neo4j container..."
    # docker-compose stop neo4j
    
    # Copy data directory
    cp -r ./neo4j-data/data "$BACKUP_PATH"
    
    # Create metadata
    cat > "$BACKUP_PATH/backup_metadata.json" << EOF
{
  "backup_type": "neo4j_data",
  "timestamp": "$TIMESTAMP",
  "source_path": "./neo4j-data/data",
  "backup_date": "$(date -Iseconds)"
}
EOF
    
    # Compress
    tar -czf "${BACKUP_PATH}.tar.gz" -C "$BACKUP_DIR" "neo4j-data_backup_$TIMESTAMP"
    rm -rf "$BACKUP_PATH"
    
    echo -e "${GREEN}✅ Neo4j data backup completed: ${BACKUP_PATH}.tar.gz${NC}"
    
    # Restart Neo4j if stopped (optional)
    # docker-compose start neo4j
}

# Main backup function
backup_all() {
    echo -e "${YELLOW}Starting full backup...${NC}"
    
    BACKUP_DIR_FULL="$BACKUP_DIR/full_backup_$TIMESTAMP"
    mkdir -p "$BACKUP_DIR_FULL"
    
    # Backup ChromaDB
    backup_chromadb
    
    # Backup Neo4j
    backup_neo4j || backup_neo4j_data
    
    # Create manifest
    cat > "$BACKUP_DIR_FULL/manifest.json" << EOF
{
  "backup_type": "full",
  "timestamp": "$TIMESTAMP",
  "backup_date": "$(date -Iseconds)",
  "components": ["chromadb", "neo4j"]
}
EOF
    
    # Compress everything
    echo "Compressing full backup..."
    tar -czf "$BACKUP_DIR_FULL.tar.gz" -C "$BACKUP_DIR" "full_backup_$TIMESTAMP"
    rm -rf "$BACKUP_DIR_FULL"
    
    echo -e "${GREEN}✅ Full backup completed: ${BACKUP_DIR_FULL}.tar.gz${NC}"
}

# Execute based on type
case "$BACKUP_TYPE" in
    chromadb)
        backup_chromadb
        ;;
    neo4j)
        backup_neo4j || backup_neo4j_data
        ;;
    all)
        backup_all
        ;;
    *)
        echo -e "${RED}Invalid backup type: $BACKUP_TYPE${NC}"
        echo "Usage: $0 [chromadb|neo4j|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo -e "${GREEN}Backup completed successfully!${NC}"
echo "Backup location: $BACKUP_DIR"
echo "=========================================="

