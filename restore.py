"""
Restore script for ChromaDB (Vector DB) and Neo4j (Graph DB) backups
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

try:
    from chromadb import PersistentClient
    from chromadb.config import Settings
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    raise

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
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


class DatabaseRestore:
    """Handle restoration of ChromaDB and Neo4j from backups."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        self.neo4j_uri = neo4j_uri or DEFAULT_NEO4J_URI
        self.neo4j_user = neo4j_user or DEFAULT_NEO4J_USER
        self.neo4j_password = neo4j_password or DEFAULT_NEO4J_PASSWORD
    
    def restore_chromadb(self, backup_path: str, backup: bool = True) -> None:
        """
        Restore ChromaDB from backup directory.
        
        Args:
            backup_path: Path to the backup directory
            backup: Whether to backup current database before restore
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup path not found: {backup_path}")
        
        logger.info(f"Restoring ChromaDB from {backup_path}")
        
        # Backup current database if it exists
        if backup and Path(CHROMA_DB_PATH).exists():
            logger.info("Backing up current ChromaDB before restore...")
            current_backup = Path(f"{CHROMA_DB_PATH}.backup")
            if current_backup.exists():
                shutil.rmtree(current_backup)
            shutil.move(CHROMA_DB_PATH, current_backup)
            logger.info(f"Current database backed up to {current_backup}")
        
        # Remove current database
        if Path(CHROMA_DB_PATH).exists():
            shutil.rmtree(CHROMA_DB_PATH)
        
        # Restore from backup
        shutil.copytree(backup_path, CHROMA_DB_PATH)
        logger.info(f"✅ ChromaDB restored from {backup_path}")
    
    def restore_chromadb_export(self, export_file: str, backup: bool = True) -> None:
        """
        Restore ChromaDB from JSON export.
        
        Args:
            export_file: Path to the JSON export file
            backup: Whether to backup current database before restore
        """
        export_file = Path(export_file)
        
        if not export_file.exists():
            raise FileNotFoundError(f"Export file not found: {export_file}")
        
        logger.info(f"Restoring ChromaDB from export {export_file}")
        
        # Backup current database
        if backup and Path(CHROMA_DB_PATH).exists():
            logger.info("Backing up current ChromaDB before restore...")
            current_backup = Path(f"{CHROMA_DB_PATH}.backup")
            if current_backup.exists():
                shutil.rmtree(current_backup)
            shutil.move(CHROMA_DB_PATH, current_backup)
        
        # Load export data
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        # Create new ChromaDB
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = PersistentClient(path=CHROMA_DB_PATH)
        
        # Delete existing collection if present
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except:
            pass
        
        # Create collection
        collection = client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Restore data
        data = export_data.get("data", {})
        if data.get("ids"):
            collection.add(
                ids=data["ids"],
                documents=data.get("documents", []),
                metadatas=data.get("metadatas", []),
                embeddings=data.get("embeddings", [])
            )
        
        logger.info(f"✅ ChromaDB restored from export: {export_file} ({export_data.get('count', 0)} vectors)")
    
    def restore_neo4j_cypher(self, export_file: str) -> None:
        """
        Restore Neo4j from Cypher export JSON.
        
        Args:
            export_file: Path to the JSON export file
        """
        export_file = Path(export_file)
        
        if not export_file.exists():
            raise FileNotFoundError(f"Export file not found: {export_file}")
        
        logger.info(f"Restoring Neo4j from {export_file}")
        
        # Load export data
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        try:
            with driver.session() as session:
                # Clear existing data
                logger.info("Clearing existing Neo4j data...")
                session.run("MATCH (n) DETACH DELETE n")
                
                # Restore nodes
                nodes = export_data.get("nodes", [])
                logger.info(f"Restoring {len(nodes)} nodes...")
                
                for node in nodes:
                    labels = ":".join(node["labels"])
                    props = json.dumps(node["properties"]).replace("'", "\\'")
                    
                    # Create node with properties
                    props_str = ", ".join([f"n.{k} = ${k}" for k in node["properties"].keys()])
                    query = f"""
                    CREATE (n:{labels})
                    SET {props_str}
                    """
                    session.run(query, **node["properties"])
                
                # Restore relationships
                relationships = export_data.get("relationships", [])
                logger.info(f"Restoring {len(relationships)} relationships...")
                
                for rel in relationships:
                    query = """
                    MATCH (a), (b)
                    WHERE id(a) = $start_id AND id(b) = $end_id
                    CREATE (a)-[r:%s]->(b)
                    SET r = $props
                    """ % rel["type"]
                    
                    session.run(query, 
                              start_id=rel["start_id"],
                              end_id=rel["end_id"],
                              props=rel["properties"])
                
                logger.info(
                    f"✅ Neo4j restored: {len(nodes)} nodes, {len(relationships)} relationships"
                )
        
        finally:
            driver.close()


def main():
    """Main restore function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Restore ChromaDB and Neo4j from backups")
    parser.add_argument(
        "backup_path",
        type=str,
        help="Path to backup file or directory"
    )
    parser.add_argument(
        "--type",
        choices=["chromadb", "neo4j", "auto"],
        default="auto",
        help="Type of backup to restore (auto detects from filename)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup current database before restore"
    )
    
    args = parser.parse_args()
    
    backup_path = Path(args.backup_path)
    
    if not backup_path.exists():
        logger.error(f"Backup path not found: {backup_path}")
        exit(1)
    
    restore = DatabaseRestore()
    
    # Auto-detect type from filename
    if args.type == "auto":
        if "chromadb" in backup_path.name.lower() or backup_path.is_dir():
            args.type = "chromadb"
        elif "neo4j" in backup_path.name.lower():
            args.type = "neo4j"
        else:
            logger.error("Could not auto-detect backup type. Please specify --type")
            exit(1)
    
    try:
        if args.type == "chromadb":
            if backup_path.is_dir():
                restore.restore_chromadb(backup_path, backup=not args.no_backup)
            elif backup_path.suffix == ".json":
                restore.restore_chromadb_export(backup_path, backup=not args.no_backup)
            else:
                logger.error("ChromaDB backup must be a directory or JSON file")
                exit(1)
        
        elif args.type == "neo4j":
            if backup_path.suffix == ".json":
                restore.restore_neo4j_cypher(backup_path)
            else:
                logger.error("Neo4j backup must be a JSON export file")
                logger.info("For .dump files, use: docker exec neo4j-graphdb neo4j-admin database load neo4j from /path/to/backup.dump")
                exit(1)
        
        logger.info("✅ Restore completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Restore failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()

