# Knowledge Graph Ingestion Script

A Python 3.12 script for ingesting Markdown documents into a knowledge graph using:
- **Vector Database (ChromaDB)**: For semantic search with embeddings
- **Graph Database (Neo4j)**: For storing paragraph relationships

## Features

- ✅ Paragraph-level document splitting
- ✅ Intelligent chunking with overlap for large paragraphs
- ✅ Embedding generation using SentenceTransformers
- ✅ Vector storage in ChromaDB
- ✅ Graph storage in Neo4j with NEXT relationships
- ✅ Checkpoint/resume functionality
- ✅ Progress tracking and statistics

## Installation

### Option 1: Docker (Recommended)

1. Create `.env` file (or run `./setup.sh`):
```bash
# Neo4j Configuration
NEO4J_AUTH=neo4j/ArifShopu1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ArifShopu1

# Application Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

2. Start services:
```bash
# Start Neo4j only
docker-compose up -d neo4j

# Or start everything (Neo4j + ingestion)
docker-compose up ingest
```

3. Access Neo4j Browser:
   - Open http://localhost:7474
   - Username: `neo4j`
   - Password: `ArifShopu1`

### Option 2: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
./setup.sh
```

3. Start Neo4j database (if not already running):
```bash
# Using Docker:
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/ArifShopu1 \
  neo4j:latest

# Or download from https://neo4j.com/download/
```

## Usage

### Docker Usage

```bash
# Check ingestion progress
docker-compose run --rm ingest python ingest.py /app/data --check

# Restart ingestion from checkpoint
docker-compose run --rm ingest python ingest.py /app/data --restart

# View logs
docker-compose logs -f ingest
```

### Basic Ingestion (Local)

```python
from ingest import ingest
from pathlib import Path

# Get all markdown files from source directory
source_dir = Path("/Users/arifshariar/Desktop/new_graph_rag/drive-download-20251029T100745Z-1-001/source_data")
document_paths = [str(f) for f in source_dir.glob("*.md")]

# Ingest documents
ingest(document_paths)
```

### Command Line Usage

```bash
# Ingest all Markdown files from a directory
python ingest.py /path/to/source/data

# Check ingestion progress and statistics
python ingest.py /path/to/source/data --check

# Restart ingestion from checkpoint
python ingest.py /path/to/source/data --restart

# Custom Neo4j connection
python ingest.py /path/to/source/data \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password your_password

# Custom chunk size and overlap
python ingest.py /path/to/source/data \
  --chunk-size 2000 \
  --chunk-overlap 300
```

### Programmatic Usage

```python
from ingest import KnowledgeGraphIngester
from pathlib import Path

# Initialize ingester
ingester = KnowledgeGraphIngester(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    chunk_size=1000,
    chunk_overlap=200
)

# Get document paths
source_dir = Path("/path/to/source/data")
document_paths = [str(f) for f in source_dir.glob("*.md")]

# Check progress
ingester.print_statistics(document_paths)

# Ingest documents
ingester.ingest(document_paths)

# Or restart from checkpoint
ingester.restart_ingest(document_paths)

# Close connections
ingester.close()
```

## Configuration

### Environment Variables

The script reads configuration from a `.env` file. You can set these environment variables:

**Neo4j Configuration:**
- `NEO4J_AUTH`: Neo4j authentication (format: `username/password`, e.g., `neo4j/ArifShopu1`)
- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: parsed from `NEO4J_AUTH` or `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: parsed from `NEO4J_AUTH` or `password`)

**Application Configuration:**
- `CHUNK_SIZE`: Maximum chunk size in characters (default: `1000`)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: `200`)

The `.env` file is automatically loaded using `python-dotenv`. Command-line arguments override environment variables.

### Chunking Parameters

- **chunk_size** (default: 1000): Maximum characters per chunk
- **chunk_overlap** (default: 200): Overlap between chunks for context preservation

The script automatically breaks paragraphs at sentence boundaries when possible, falling back to word boundaries.

## Data Structure

### ChromaDB (Vector Store)

Each chunk is stored with:
- **id**: Unique identifier `{file_name}::{paragraph_index}::chunk_{chunk_index}`
- **embedding**: Vector embedding (384 dimensions for all-MiniLM-L6-v2)
- **document**: Full chunk text
- **metadata**: 
  - `file_name`: Source file name
  - `paragraph_index`: Paragraph index in file
  - `chunk_index`: Chunk index within paragraph
  - `content`: Preview of chunk text

### Neo4j (Graph Store)

**Nodes:**
```cypher
(:Paragraph {
  id: "{file_name}::{paragraph_index}",
  file_name: "filename.md",
  paragraph_index: 0,
  content: "Full paragraph text...",
  chunk_count: 1
})
```

**Relationships:**
```cypher
(:Paragraph)-[:NEXT]->(:Paragraph)
```
Connects consecutive paragraphs in the same file.

## Checkpoint System

The script automatically saves checkpoints after each file is processed to `ingest_checkpoint.json`. This allows:

- **Graceful interruption**: Stop with Ctrl+C and resume later
- **Progress tracking**: See what's been processed
- **Resume capability**: Use `restart_ingest()` to continue from last checkpoint

## Backup and Restore

### Backup

Back up both ChromaDB and Neo4j databases:

**Using Python script:**
```bash
# Backup everything
python backup.py --type all --compress

# Backup only ChromaDB
python backup.py --type chromadb

# Backup only Neo4j
python backup.py --type neo4j

# Custom output directory
python backup.py --type all --output /path/to/backups
```

**Using shell script:**
```bash
# Backup everything
./backup.sh all

# Backup only ChromaDB
./backup.sh chromadb

# Backup only Neo4j
./backup.sh neo4j
```

**Using Docker:**
```bash
# Run backup service
docker-compose up backup
```

Backups are saved to `./backups/` directory with timestamps:
- `chromadb_backup_YYYYMMDD_HHMMSS/` - ChromaDB directory backup
- `chromadb_export_YYYYMMDD_HHMMSS.json` - ChromaDB JSON export
- `neo4j_export_YYYYMMDD_HHMMSS.json` - Neo4j Cypher export
- `full_backup_YYYYMMDD_HHMMSS.tar.gz` - Compressed full backup

### Restore

Restore databases from backups:

```bash
# Restore ChromaDB from directory backup
python restore.py backups/chromadb_backup_20250115_120000 --type chromadb

# Restore ChromaDB from JSON export
python restore.py backups/chromadb_export_20250115_120000.json --type chromadb

# Restore Neo4j from JSON export
python restore.py backups/neo4j_export_20250115_120000.json --type neo4j

# Restore Neo4j from .dump file (requires neo4j-admin)
docker exec neo4j-graphdb neo4j-admin database load neo4j from /backups/neo4j_backup.dump
```

**Note:** The restore process automatically backs up your current database before restoring (unless `--no-backup` is used).

## Querying

### Basic Usage

Query the knowledge graph to get answers with Vancouver-style citations:

**Using Python function:**
```python
from query import query

questions = [
    "What is the maximum duration of a commercial lease?",
    "What is the filing fee for an appeal in civil court?"
]

answers = query(questions)
# Returns list of answer strings with citations
```

**Using command line:**
```bash
# Query from JSON file
python query.py drive-download-20251029T100745Z-1-001/sample_questions.json

# Query with direct questions
python query.py --questions "What is the maximum duration of a commercial lease?" "What is the filing fee?"

# Enable debug mode
python query.py sample_questions.json --debug

# Adjust parallel workers (for ~400 questions in ≤60 min, use 10-20 workers)
python query.py sample_questions.json --max-workers 15

# Custom output file
python query.py sample_questions.json --output my_answers.json
```

### Query Process

1. **GraphDB Search**: Searches Neo4j using Cypher queries with keyword matching
2. **VectorDB Search**: Searches ChromaDB using semantic embeddings
3. **Result Comparison**: Selects the result with highest relevance score
4. **Fallback**: If no results found, searches source files sequentially
5. **Citation**: Formats answer with Vancouver-style citations `[1]` and references

### Output Format

Answers are saved to `answers.json` in the format:
```json
[
  {
    "answer": "The maximum duration of a commercial lease is 15 years [1].\n\nReferences\n1. commercial_law.md."
  },
  {
    "answer": "The filing fee for an appeal in civil court is 200 USD [2].\n\nReferences\n2. civil_procedure.md."
  }
]
```

### Performance

- **Parallel Execution**: Processes multiple questions simultaneously
- **Progress Indicator**: Shows real-time progress with tqdm
- **Target**: ~400 questions in ≤60 minutes (with 10-20 parallel workers)
- **Optimization**: Adjust `--max-workers` based on your system

### Debug Mode

Use `--debug` flag to see:
- Cypher queries executed
- Matched nodes and their relevance scores
- VectorDB query embeddings and top scores
- Final selected source and result summary

Example debug output:
```
Query: What is the maximum duration of a commercial lease?
============================================================
Cypher Query: MATCH (p:Paragraph) WHERE ...
GraphDB Results: 3 matches
  1. Relevance: 0.850, File: commercial_law.md
  2. Relevance: 0.720, File: business_regulations.md
VectorDB Results: 2 matches
  1. Relevance: 0.920, Distance: 0.080, File: commercial_law.md
Selected Result:
  Source: vectordb
  File: commercial_law.md
  Relevance: 0.920
```

## Statistics

Use `check_ingest()` or the `--check` flag to see:

- Files processed/remaining
- Progress percentage
- Number of nodes in Neo4j
- Number of relationships in Neo4j
- Number of vectors in ChromaDB
- Checkpoint status

Example output:
```
============================================================
INGESTION STATISTICS
============================================================

📁 FILES:
  Total:           60
  Processed:      45
  Remaining:      15
  Progress:       75.0%

🕸️  NEO4J GRAPH DATABASE:
  Nodes:          1234
  Relationships:  1179

🔍 VECTOR DATABASE (ChromaDB):
  Vectors:        1567

💾 CHECKPOINT:
  Status:         Available
  Last Updated:   2025-01-15T10:30:00
============================================================
```

## Notes

- The script uses `sentence-transformers/all-MiniLM-L6-v2` by default (no API key required)
- ChromaDB data is stored in `./chroma_db/`
- Checkpoints are saved in `ingest_checkpoint.json`
- Paragraph splitting uses double newlines (`\n\n`) as delimiters
- Large paragraphs are automatically chunked with intelligent boundary detection

