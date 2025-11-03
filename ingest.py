"""
Knowledge Graph Ingestion Script
Reads Markdown files, creates embeddings, stores in VectorDB and Neo4j GraphDB.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    from chromadb import Client, PersistentClient
    from chromadb.config import Settings
    from neo4j import GraphDatabase
    import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    raise

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants - can be overridden by environment variables
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))  # characters per chunk
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # overlap between chunks
CHECKPOINT_FILE = "ingest_checkpoint.json"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Neo4j configuration from environment variables
def _parse_neo4j_auth():
    """Parse NEO4J_AUTH environment variable (format: username/password)."""
    neo4j_auth = os.getenv("NEO4J_AUTH", "")
    if neo4j_auth and "/" in neo4j_auth:
        username, password = neo4j_auth.split("/", 1)
        return username, password
    return None, None

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_neo4j_user_env, _neo4j_pass_env = _parse_neo4j_auth()
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", _neo4j_user_env or "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", _neo4j_pass_env or "password")

class KnowledgeGraphIngester:
    """Main class for ingesting documents into knowledge graph."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the ingester.
        
        Args:
            neo4j_uri: Neo4j database URI (defaults to NEO4J_URI env var or bolt://localhost:7687)
            neo4j_user: Neo4j username (defaults to NEO4J_USER env var or from NEO4J_AUTH)
            neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var or from NEO4J_AUTH)
            chunk_size: Maximum chunk size in characters (defaults to CHUNK_SIZE env var)
            chunk_overlap: Overlap between chunks in characters (defaults to CHUNK_OVERLAP env var)
            embedding_model: Name of embedding model (default: all-MiniLM-L6-v2)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = embedding_model or EMBEDDING_MODEL_NAME
        
        # Use environment variables as defaults if not provided
        self.neo4j_uri = neo4j_uri or DEFAULT_NEO4J_URI
        self.neo4j_user = neo4j_user or DEFAULT_NEO4J_USER
        self.neo4j_password = neo4j_password or DEFAULT_NEO4J_PASSWORD
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        self.chroma_client = PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.chroma_collection = self.chroma_client.get_collection("paragraphs")
            logger.info("Using existing ChromaDB collection")
        except Exception:
            self.chroma_collection = self.chroma_client.create_collection(
                name="paragraphs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new ChromaDB collection")
        
        # Initialize Neo4j
        logger.info(f"Connecting to Neo4j at {self.neo4j_uri}...")
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self._init_neo4j_constraints()
        
        # Statistics tracking
        self.stats = {
            "files_processed": 0,
            "files_total": 0,
            "paragraphs_created": 0,
            "vectors_created": 0,
            "relationships_created": 0,
            "nodes_created": 0
        }
        
    def _init_neo4j_constraints(self):
        """Create Neo4j constraints and indexes."""
        with self.neo4j_driver.session() as session:
            # Create unique constraint on paragraph identifier
            try:
                session.run("""
                    CREATE CONSTRAINT paragraph_id IF NOT EXISTS
                    FOR (p:Paragraph)
                    REQUIRE p.id IS UNIQUE
                """)
            except Exception as e:
                # Try alternative syntax for older Neo4j versions
                try:
                    session.run("""
                        CREATE CONSTRAINT paragraph_id
                        FOR (p:Paragraph)
                        REQUIRE p.id IS UNIQUE
                    """)
                except Exception:
                    # Constraint might already exist, which is fine
                    logger.debug(f"Constraint creation note: {e}")
            
            # Create File node constraint
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.file_name IS UNIQUE")
            except Exception as e:
                logger.warning(f"Could not create File constraint (may already exist): {e}")
                try:
                    session.run("CREATE CONSTRAINT FOR (f:File) REQUIRE f.file_name IS UNIQUE")
                except Exception:
                    pass
            
            # Create indexes for faster searches
            try:
                session.run("CREATE INDEX IF NOT EXISTS FOR (p:Paragraph) ON (p.file_name)")
            except Exception as e:
                logger.debug(f"Could not create file_name index (may already exist): {e}")
            
            try:
                session.run("CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.file_name)")
            except Exception as e:
                logger.debug(f"Could not create File file_name index (may already exist): {e}")
            
            logger.info("Neo4j constraints and indexes initialized")
            
            # Migrate existing Paragraph nodes to File nodes if needed
            self._migrate_to_file_nodes(session)
    
    def _migrate_to_file_nodes(self, session):
        """Migrate existing Paragraph nodes to use File nodes and CONTAINS relationships."""
        try:
            # Check if File nodes already exist
            result = session.run("MATCH (f:File) RETURN count(f) as count")
            file_count = result.single()["count"]
            
            if file_count > 0:
                logger.info(f"File nodes already exist ({file_count} files). Migration not needed.")
                return
            
            # Get all unique file names from Paragraph nodes
            result = session.run("""
                MATCH (p:Paragraph)
                RETURN DISTINCT p.file_name AS file_name
            """)
            file_names = [record["file_name"] for record in result]
            
            if not file_names:
                logger.info("No Paragraph nodes found. No migration needed.")
                return
            
            logger.info(f"Migrating {len(file_names)} files to File node schema...")
            
            # Create File nodes and CONTAINS relationships
            for file_name in file_names:
                # Create File node
                session.run("""
                    MERGE (f:File {file_name: $file_name})
                    SET f.file_name = $file_name
                """, file_name=file_name)
                
                # Link all Paragraphs from this file to the File node
                session.run("""
                    MATCH (f:File {file_name: $file_name})
                    MATCH (p:Paragraph {file_name: $file_name})
                    MERGE (f)-[:CONTAINS]->(p)
                """, file_name=file_name)
            
            logger.info(f"✅ Migration complete: Created {len(file_names)} File nodes and CONTAINS relationships")
            
        except Exception as e:
            logger.warning(f"Migration failed (may already be done): {e}")
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """
        Split content into paragraphs based on double newlines.
        
        Args:
            content: File content as string
            
        Returns:
            List of paragraph strings
        """
        # Split by double newline, but also handle single newlines with spacing
        paragraphs = []
        current_para = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                if current_para:
                    para_text = ' '.join(current_para)
                    if para_text.strip():
                        paragraphs.append(para_text)
                    current_para = []
            else:
                current_para.append(line)
        
        # Add remaining paragraph
        if current_para:
            para_text = ' '.join(current_para)
            if para_text.strip():
                paragraphs.append(para_text)
        
        return paragraphs
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap if it exceeds chunk_size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk strings
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary or space
            # Look for sentence endings near the chunk boundary
            boundary_chars = ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']
            best_break = end
            
            for boundary in boundary_chars:
                pos = text.rfind(boundary, start, end)
                if pos != -1 and pos > start + self.chunk_size // 2:
                    best_break = pos + len(boundary.rstrip())
                    break
            
            # If no sentence boundary, try breaking at space
            if best_break == end:
                space_pos = text.rfind(' ', start, end)
                if space_pos != -1 and space_pos > start + self.chunk_size // 2:
                    best_break = space_pos + 1
            
            chunk = text[start:best_break].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = max(start + 1, best_break - self.chunk_overlap)
        
        return chunks
    
    def _generate_id(self, file_name: str, paragraph_index: int, chunk_index: int = 0) -> str:
        """Generate unique ID for a paragraph chunk."""
        base_id = f"{file_name}::{paragraph_index}"
        if chunk_index > 0:
            return f"{base_id}::chunk_{chunk_index}"
        return base_id
    
    def _store_in_vectordb(
        self,
        chunks: List[str],
        file_name: str,
        paragraph_index: int,
        embeddings: List[List[float]]
    ):
        """Store chunks and embeddings in ChromaDB."""
        if not chunks:
            return
        
        ids = []
        metadatas = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = self._generate_id(file_name, paragraph_index, chunk_idx)
            ids.append(chunk_id)
            metadatas.append({
                "file_name": file_name,
                "paragraph_index": paragraph_index,
                "chunk_index": chunk_idx,
                "content": chunk[:500]  # Store preview in metadata
            })
        
        # Add to collection
        self.chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        self.stats["vectors_created"] += len(chunks)
    
    def _store_in_neo4j(
        self,
        file_name: str,
        paragraph_index: int,
        content: str,
        chunks: List[str]
    ):
        """Store paragraph node in Neo4j with NEXT relationships and File node."""
        para_id = self._generate_id(file_name, paragraph_index)
        
        with self.neo4j_driver.session() as session:
            # First, ensure File node exists (MERGE creates if doesn't exist)
            session.run("""
                MERGE (f:File {file_name: $file_name})
                SET f.file_name = $file_name
            """, file_name=file_name)
            
            # Create or update paragraph node
            session.run("""
                MERGE (p:Paragraph {id: $para_id})
                SET p.file_name = $file_name,
                    p.paragraph_index = $paragraph_index,
                    p.content = $content,
                    p.chunk_count = $chunk_count
            """, para_id=para_id, file_name=file_name, paragraph_index=paragraph_index,
                content=content, chunk_count=len(chunks))
            
            self.stats["nodes_created"] += 1
            
            # Link Paragraph to File node
            session.run("""
                MATCH (f:File {file_name: $file_name})
                MATCH (p:Paragraph {id: $para_id})
                MERGE (f)-[:CONTAINS]->(p)
            """, file_name=file_name, para_id=para_id)
            
            # Create NEXT relationship with previous paragraph
            if paragraph_index > 0:
                prev_para_id = self._generate_id(file_name, paragraph_index - 1)
                session.run("""
                    MATCH (prev:Paragraph {id: $prev_id})
                    MATCH (curr:Paragraph {id: $curr_id})
                    MERGE (prev)-[:NEXT]->(curr)
                """, prev_id=prev_para_id, curr_id=para_id)
                self.stats["relationships_created"] += 1
    
    def _process_file(self, file_path: str) -> Tuple[int, int]:
        """
        Process a single file and return (paragraphs_created, vectors_created).
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (paragraphs_count, vectors_count)
        """
        file_name = Path(file_path).name
        logger.info(f"Processing file: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_name}: {e}")
            return 0, 0
        
        paragraphs = self._split_into_paragraphs(content)
        logger.info(f"Found {len(paragraphs)} paragraphs in {file_name}")
        
        paragraphs_created = 0
        vectors_created = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Chunk the paragraph if needed
            chunks = self._chunk_text(paragraph)
            
            # Generate embeddings for all chunks
            if chunks:
                embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist()
                
                # Store in VectorDB
                self._store_in_vectordb(chunks, file_name, para_idx, embeddings)
                vectors_created += len(chunks)
                
                # Store in Neo4j (store full paragraph content, not chunks)
                self._store_in_neo4j(file_name, para_idx, paragraph, chunks)
                paragraphs_created += 1
        
        self.stats["paragraphs_created"] += paragraphs_created
        return paragraphs_created, vectors_created
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint from file."""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
        return {
            "processed_files": [],
            "stats": self.stats.copy()
        }
    
    def _save_checkpoint(self, processed_files: List[str], stats: Dict):
        """Save checkpoint to file."""
        checkpoint = {
            "processed_files": processed_files,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info("Checkpoint saved")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def ingest(self, document_paths: List[str]) -> None:
        """
        Ingest Markdown docs and build the knowledge graph.
        
        Args:
            document_paths: List of file paths to ingest
        """
        logger.info(f"Starting ingestion of {len(document_paths)} files")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint()
        processed_files = set(checkpoint.get("processed_files", []))
        
        # Filter out already processed files
        remaining_files = [f for f in document_paths if f not in processed_files]
        
        if not remaining_files:
            logger.info("All files already processed")
            return
        
        self.stats["files_total"] = len(document_paths)
        self.stats["files_processed"] = len(processed_files)
        
        # Process files with progress bar
        try:
            for file_path in tqdm.tqdm(remaining_files, desc="Ingesting files"):
                para_count, vec_count = self._process_file(file_path)
                processed_files.add(file_path)
                self.stats["files_processed"] += 1
                
                # Save checkpoint periodically (every file)
                self._save_checkpoint(list(processed_files), self.stats.copy())
                
                logger.info(
                    f"Completed {file_path}: {para_count} paragraphs, {vec_count} vectors"
                )
        except KeyboardInterrupt:
            logger.info("Ingestion interrupted by user")
            self._save_checkpoint(list(processed_files), self.stats.copy())
            raise
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            self._save_checkpoint(list(processed_files), self.stats.copy())
            raise
        
        logger.info("Ingestion completed successfully")
    
    def stop_ingest(self) -> None:
        """Save checkpoint and stop ingestion gracefully."""
        checkpoint = self._load_checkpoint()
        self._save_checkpoint(
            checkpoint.get("processed_files", []),
            self.stats.copy()
        )
        logger.info("Ingestion stopped and checkpoint saved")
    
    def restart_ingest(self, document_paths: List[str]) -> None:
        """
        Restart ingestion from previous checkpoint.
        
        Args:
            document_paths: List of all file paths to ingest
        """
        logger.info("Restarting ingestion from checkpoint...")
        checkpoint = self._load_checkpoint()
        
        if checkpoint.get("stats"):
            self.stats.update(checkpoint["stats"])
            logger.info(f"Resumed stats: {self.stats}")
        
        self.ingest(document_paths)
    
    def check_ingest(self, document_paths: List[str]) -> Dict:
        """
        Check ingestion progress and statistics.
        
        Args:
            document_paths: List of all file paths to ingest
            
        Returns:
            Dictionary with progress and statistics
        """
        checkpoint = self._load_checkpoint()
        processed_files = set(checkpoint.get("processed_files", []))
        total_files = len(document_paths)
        remaining_files = total_files - len(processed_files)
        
        # Get stats from checkpoint
        checkpoint_stats = checkpoint.get("stats", {})
        
        # Get actual counts from databases
        with self.neo4j_driver.session() as session:
            node_result = session.run("MATCH (p:Paragraph) RETURN count(p) as count")
            nodes_count = node_result.single()["count"]
            
            rel_result = session.run("MATCH ()-[r:NEXT]->() RETURN count(r) as count")
            relationships_count = rel_result.single()["count"]
        
        vectors_count = self.chroma_collection.count()
        
        stats = {
            "files": {
                "total": total_files,
                "processed": len(processed_files),
                "remaining": remaining_files,
                "progress_percentage": (len(processed_files) / total_files * 100) if total_files > 0 else 0
            },
            "graph_database": {
                "nodes": nodes_count,
                "relationships": relationships_count
            },
            "vector_database": {
                "vectors": vectors_count
            },
            "checkpoint": {
                "exists": os.path.exists(CHECKPOINT_FILE),
                "last_updated": checkpoint.get("timestamp", "N/A")
            }
        }
        
        return stats
    
    def print_statistics(self, document_paths: List[str]) -> None:
        """Print formatted statistics."""
        stats = self.check_ingest(document_paths)
        
        print("\n" + "="*60)
        print("INGESTION STATISTICS")
        print("="*60)
        
        print(f"\n📁 FILES:")
        print(f"  Total:           {stats['files']['total']}")
        print(f"  Processed:      {stats['files']['processed']}")
        print(f"  Remaining:      {stats['files']['remaining']}")
        print(f"  Progress:       {stats['files']['progress_percentage']:.1f}%")
        
        print(f"\n🕸️  NEO4J GRAPH DATABASE:")
        print(f"  Nodes:          {stats['graph_database']['nodes']}")
        print(f"  Relationships:  {stats['graph_database']['relationships']}")
        
        print(f"\n🔍 VECTOR DATABASE (ChromaDB):")
        print(f"  Vectors:        {stats['vector_database']['vectors']}")
        
        print(f"\n💾 CHECKPOINT:")
        print(f"  Status:         {'Available' if stats['checkpoint']['exists'] else 'Not found'}")
        print(f"  Last Updated:   {stats['checkpoint']['last_updated']}")
        
        print("="*60 + "\n")
    
    def close(self):
        """Close database connections."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        logger.info("Connections closed")


def ingest(document_paths: List[str]) -> None:
    """
    Ingest Markdown docs and build the knowledge graph.
    
    Args:
        document_paths: List of file paths to ingest
    """
    # Default configuration - can be customized
    ingester = KnowledgeGraphIngester()
    
    try:
        ingester.ingest(document_paths)
    finally:
        ingester.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph Ingestion Tool")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Directory containing Markdown files to ingest"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help=f"Neo4j URI (default: from NEO4J_URI env var or {DEFAULT_NEO4J_URI})"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help=f"Neo4j username (default: from NEO4J_USER env var or {DEFAULT_NEO4J_USER})"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Neo4j password (default: from NEO4J_PASSWORD env var or NEO4J_AUTH)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check ingestion progress and statistics"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart ingestion from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Get all markdown files from source directory
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"Error: Directory {args.source_dir} does not exist")
        exit(1)
    
    md_files = list(source_path.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {args.source_dir}")
        exit(1)
    
    document_paths = [str(f) for f in md_files]
    
    # Create ingester
    ingester = KnowledgeGraphIngester(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    try:
        if args.check:
            ingester.print_statistics(document_paths)
        elif args.restart:
            ingester.restart_ingest(document_paths)
        else:
            ingester.ingest(document_paths)
    finally:
        ingester.close()

