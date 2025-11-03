"""
Backup script for ChromaDB (Vector DB) and Neo4j (Graph DB)
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import tarfile

try:
    from chromadb import PersistentClient
    from chromadb.config import Settings
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    raise

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "./chroma_db"
BACKUP_BASE_DIR = "./backups"
CHROMA_COLLECTION_NAME = "paragraphs"


def _parse_neo4j_auth():
    """Parse NEO4J_AUTH environment variable."""
    neo4j_auth = os.getenv("NEO4J_AUTH", "")
    if neo4j_auth and "/" in neo4j_auth:
        username, password = neo4j_auth.split("/", 1)
        return username, password
    return None, None


DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_neo4j_user_env, _neo4j_pass_env = _parse_neo4j_auth()
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", _neo4j_user_env or "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", _neo4j_pass_env or "password")


class DatabaseBackup:
    """Handle backups for ChromaDB and Neo4j."""
    
    def __init__(
        self,
        backup_dir: str = BACKUP_BASE_DIR,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.neo4j_uri = neo4j_uri or DEFAULT_NEO4J_URI
        self.neo4j_user = neo4j_user or DEFAULT_NEO4J_USER
        self.neo4j_password = neo4j_password or DEFAULT_NEO4J_PASSWORD
    
    def backup_chromadb(self, timestamp: Optional[str] = None) -> Path:
        """
        Backup ChromaDB by copying the database directory.
        
        Args:
            timestamp: Optional timestamp string for backup naming
            
        Returns:
            Path to the backup directory
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_path = self.backup_dir / f"chromadb_backup_{timestamp}"
        
        logger.info(f"Starting ChromaDB backup to {backup_path}")
        
        if not Path(CHROMA_DB_PATH).exists():
            logger.warning(f"ChromaDB path {CHROMA_DB_PATH} does not exist")
            return backup_path
        
        try:
            # Copy entire ChromaDB directory
            shutil.copytree(CHROMA_DB_PATH, backup_path, dirs_exist_ok=False)
            
            # Create metadata file
            metadata = {
                "backup_type": "chromadb",
                "timestamp": timestamp,
                "source_path": CHROMA_DB_PATH,
                "backup_path": str(backup_path),
                "backup_date": datetime.now().isoformat()
            }
            
            metadata_file = backup_path / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ ChromaDB backup completed: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"❌ ChromaDB backup failed: {e}")
            raise
    
    def backup_chromadb_export(self, timestamp: Optional[str] = None) -> Path:
        """
        Export ChromaDB collection as JSON for backup.
        
        Args:
            timestamp: Optional timestamp string for backup naming
            
        Returns:
            Path to the exported JSON file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_file = self.backup_dir / f"chromadb_export_{timestamp}.json"
        
        logger.info(f"Starting ChromaDB export to {backup_file}")
        
        try:
            # Connect to ChromaDB
            client = PersistentClient(path=CHROMA_DB_PATH)
            
            try:
                collection = client.get_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                logger.warning(f"Collection '{CHROMA_COLLECTION_NAME}' not found")
                return backup_file
            
            # Get all data from collection
            results = collection.get(include=["documents", "metadatas", "embeddings"])
            
            # Export to JSON
            export_data = {
                "collection_name": CHROMA_COLLECTION_NAME,
                "timestamp": timestamp,
                "backup_date": datetime.now().isoformat(),
                "count": len(results["ids"]) if results["ids"] else 0,
                "data": {
                    "ids": results.get("ids", []),
                    "documents": results.get("documents", []),
                    "metadatas": results.get("metadatas", []),
                    "embeddings": results.get("embeddings", [])
                }
            }
            
            with open(backup_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"✅ ChromaDB export completed: {backup_file} ({export_data['count']} vectors)")
            return backup_file
            
        except Exception as e:
            logger.error(f"❌ ChromaDB export failed: {e}")
            raise
    
    def backup_neo4j(self, timestamp: Optional[str] = None) -> Path:
        """
        Backup Neo4j database using neo4j-admin dump.
        Note: This requires Neo4j to be running and neo4j-admin available.
        
        Args:
            timestamp: Optional timestamp string for backup naming
            
        Returns:
            Path to the backup file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_file = self.backup_dir / f"neo4j_backup_{timestamp}.dump"
        
        logger.info(f"Starting Neo4j backup to {backup_file}")
        
        # Check if running in Docker
        if os.path.exists("/.dockerenv"):
            # Inside Docker container - use docker exec
            logger.info("Running inside Docker container")
            container_name = os.getenv("NEO4J_CONTAINER", "neo4j-graphdb")
            
            import subprocess
            try:
                # Use docker exec to run neo4j-admin dump
                cmd = [
                    "docker", "exec", container_name,
                    "neo4j-admin", "database", "dump", "neo4j",
                    "--to-path=/tmp"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Copy from container
                copy_cmd = [
                    "docker", "cp",
                    f"{container_name}:/tmp/neo4j.dump",
                    str(backup_file)
                ]
                subprocess.run(copy_cmd, check=True)
                
                logger.info(f"✅ Neo4j backup completed: {backup_file}")
                return backup_file
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Neo4j backup failed: {e.stderr}")
                raise
        
        else:
            # Local execution - try to use neo4j-admin directly
            logger.warning("Neo4j backup requires neo4j-admin. Please use Docker method or manual backup.")
            logger.info("Manual backup command: docker exec neo4j-graphdb neo4j-admin database dump neo4j")
            
            # Alternative: Export via Cypher
            return self.backup_neo4j_cypher(timestamp)
    
    def backup_neo4j_cypher(self, timestamp: Optional[str] = None) -> Path:
        """
        Backup Neo4j using Cypher queries (alternative method).
        
        Args:
            timestamp: Optional timestamp string for backup naming
            
        Returns:
            Path to the exported JSON file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_file = self.backup_dir / f"neo4j_export_{timestamp}.json"
        
        logger.info(f"Starting Neo4j Cypher export to {backup_file}")
        
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            with driver.session() as session:
                # Get all nodes
                nodes_result = session.run("""
                    MATCH (n)
                    RETURN id(n) as node_id, labels(n) as labels, properties(n) as properties
                    ORDER BY id(n)
                """)
                
                nodes = []
                for record in nodes_result:
                    nodes.append({
                        "id": record["node_id"],
                        "labels": record["labels"],
                        "properties": dict(record["properties"])
                    })
                
                # Get all relationships
                rels_result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN id(a) as start_id, type(r) as type, properties(r) as properties, id(b) as end_id
                    ORDER BY id(r)
                """)
                
                relationships = []
                for record in rels_result:
                    relationships.append({
                        "start_id": record["start_id"],
                        "type": record["type"],
                        "properties": dict(record["properties"]),
                        "end_id": record["end_id"]
                    })
                
                # Export to JSON
                export_data = {
                    "export_type": "neo4j_cypher",
                    "timestamp": timestamp,
                    "backup_date": datetime.now().isoformat(),
                    "node_count": len(nodes),
                    "relationship_count": len(relationships),
                    "nodes": nodes,
                    "relationships": relationships
                }
                
                with open(backup_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(
                    f"✅ Neo4j export completed: {backup_file} "
                    f"({len(nodes)} nodes, {len(relationships)} relationships)"
                )
                
            driver.close()
            return backup_file
            
        except Exception as e:
            logger.error(f"❌ Neo4j export failed: {e}")
            raise
    
    def backup_all(self, compress: bool = True) -> Path:
        """
        Backup both ChromaDB and Neo4j.
        
        Args:
            compress: Whether to compress the backup into a tar.gz file
            
        Returns:
            Path to the backup (directory or compressed file)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_dir / f"full_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("Starting full backup")
        logger.info("="*60)
        
        try:
            # Backup ChromaDB
            chroma_backup = self.backup_chromadb(timestamp)
            if chroma_backup.exists():
                # Move to full backup directory
                shutil.move(str(chroma_backup), str(backup_dir / "chromadb"))
            
            # Export ChromaDB as JSON
            chroma_export = self.backup_chromadb_export(timestamp)
            if chroma_export.exists():
                shutil.move(str(chroma_export), str(backup_dir / chroma_export.name))
            
            # Backup Neo4j
            try:
                neo4j_backup = self.backup_neo4j_cypher(timestamp)
                if neo4j_backup.exists():
                    shutil.move(str(neo4j_backup), str(backup_dir / neo4j_backup.name))
            except Exception as e:
                logger.warning(f"Neo4j backup failed: {e}")
            
            # Create backup manifest
            manifest = {
                "backup_type": "full",
                "timestamp": timestamp,
                "backup_date": datetime.now().isoformat(),
                "components": {
                    "chromadb": str(chroma_backup) if chroma_backup.exists() else None,
                    "neo4j": str(neo4j_backup) if 'neo4j_backup' in locals() and neo4j_backup.exists() else None
                }
            }
            
            manifest_file = backup_dir / "backup_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Full backup completed: {backup_dir}")
            
            # Compress if requested
            if compress:
                compressed_file = self.backup_dir / f"full_backup_{timestamp}.tar.gz"
                logger.info(f"Compressing backup to {compressed_file}")
                
                with tarfile.open(compressed_file, "w:gz") as tar:
                    tar.add(backup_dir, arcname=backup_dir.name)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_dir)
                
                logger.info(f"✅ Compressed backup created: {compressed_file}")
                return compressed_file
            
            return backup_dir
            
        except Exception as e:
            logger.error(f"❌ Full backup failed: {e}")
            raise


def main():
    """Main backup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup ChromaDB and Neo4j databases")
    parser.add_argument(
        "--type",
        choices=["chromadb", "neo4j", "all"],
        default="all",
        help="Type of backup to perform"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=BACKUP_BASE_DIR,
        help="Output directory for backups"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress full backup into tar.gz"
    )
    
    args = parser.parse_args()
    
    backup = DatabaseBackup(backup_dir=args.output)
    
    try:
        if args.type == "chromadb":
            backup.backup_chromadb()
            backup.backup_chromadb_export()
        elif args.type == "neo4j":
            backup.backup_neo4j_cypher()
        elif args.type == "all":
            backup.backup_all(compress=args.compress)
        
        logger.info("✅ Backup process completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()

