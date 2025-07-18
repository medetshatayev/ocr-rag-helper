import os
import logging
from typing import List, Dict, Optional, Tuple
from uuid import uuid4  # For generating unique fallback IDs
from pathlib import Path
from collections import defaultdict
import re

# Patch for sqlite3 if pysqlite3-binary is installed
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Will fall back to system sqlite3

import chromadb
from chromadb.config import Settings
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

from document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Core RAG system with vector storage and OpenAI integration."""
    
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_directory = persist_directory
        self.document_processor = DocumentProcessor()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Configuration
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        self.chat_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.max_retrieval_results = int(os.getenv("MAX_RETRIEVAL_RESULTS", "10"))
        
        # Initialize ChromaDB
        self._init_vector_db()
        
    def _init_vector_db(self):
        """Initialize ChromaDB with persistence and proper error handling."""
        try:
            # Ensure directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with the most compatible approach
            try:
                # Use PersistentClient with minimal settings for ChromaDB 0.5+
                self.chroma_client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info("Initialized ChromaDB with PersistentClient, telemetry disabled")
                
            except Exception as e1:
                logger.warning(f"PersistentClient failed: {e1}")
                # Fallback to basic client
                self.chroma_client = chromadb.Client()
                logger.info("Initialized ChromaDB with basic Client (in-memory)")
            
            # Create collection with proper error handling
            collection_name = "rag_documents"  # Use a more specific name
            
            try:
                # First, try to get existing collection
                self.collection = self.chroma_client.get_collection(name=collection_name)
                logger.info(f"Retrieved existing collection '{collection_name}'")
                
            except Exception:
                # Collection doesn't exist, create it
                try:
                    # Create collection without any metadata to avoid '_type' issues
                    self.collection = self.chroma_client.create_collection(
                        name=collection_name,
                        get_or_create=True
                    )
                    logger.info(f"Created new collection '{collection_name}'")
                    
                except Exception as create_error:
                    # Final fallback: use get_or_create with minimal parameters
                    logger.warning(f"Standard collection creation failed: {create_error}")
                    self.collection = self.chroma_client.get_or_create_collection(
                        name=collection_name
                    )
                    logger.info(f"Created collection '{collection_name}' with fallback method")
            
            # Verify collection is working
            try:
                doc_count = self.collection.count()
                logger.info(f"Vector database initialized successfully. Documents in collection: {doc_count}")
            except Exception as count_error:
                logger.warning(f"Could not get document count: {count_error}")
                logger.info("Vector database initialized (count unavailable)")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            # Final fallback: create minimal in-memory database
            logger.warning("Creating minimal in-memory ChromaDB as final fallback")
            try:
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.get_or_create_collection(name="fallback_docs")
                logger.info("Minimal fallback database created successfully")
            except Exception as fallback_error:
                logger.error(f"All initialization methods failed: {fallback_error}")
                raise RuntimeError(f"Cannot initialize ChromaDB: {fallback_error}")
    
    def index_file(self, file_path: str, document_id: str, original_filename: str) -> Dict:
        """Process a single file and store it with a specific document_id."""
        logger.info(f"Indexing file: {file_path} with document_id: {document_id}")
        
        # Process the single document, passing the original filename to be stored in metadata
        chunks = self.document_processor.process_documents(
            [file_path],
            original_filename=original_filename
        )
        
        if not chunks:
            return {"status": "no_content", "message": "No content extracted from the file"}

        # Generate embeddings and store
        result = self._store_chunks(chunks, document_id)
        
        result.update({
            "files_processed": 1,
            "chunks_created": len(chunks)
        })
        
        return result

    def index_directory(self, directory_path: str, force_reindex: bool = False) -> Dict:
        """Index all documents in a directory."""
        if force_reindex:
            logger.info("Force reindexing - clearing existing collection")
            try:
                collection_name = self.collection.name
                self.chroma_client.delete_collection(name=collection_name)
                self.collection = self.chroma_client.create_collection(name=collection_name)
                logger.info(f"Successfully cleared and recreated collection '{collection_name}'.")
            except Exception as e:
                logger.error(f"Error clearing collection for re-indexing: {e}")
                return {"status": "error", "message": f"Failed to clear collection: {e}"}
            
            self.document_processor.processed_files.clear()
        
        # Get file statistics
        stats = self.document_processor.get_file_stats(directory_path)
        logger.info(f"Directory stats: {stats}")
        
        # Scan and process documents
        file_paths = self.document_processor.scan_directory(directory_path)
        
        if not file_paths:
            return {"status": "no_files", "message": "No supported files found"}
        
        # Process documents
        chunks = self.document_processor.process_documents(file_paths)
        
        if not chunks:
            return {"status": "no_content", "message": "No content extracted from files"}
        
        # This part is tricky as we don't have a single document_id for a directory.
        # For now, we will assign a generic one.
        # A better approach would be to have a document_id per file.
        directory_document_id = f"dir_{os.path.basename(directory_path)}"
        # Generate embeddings and store
        result = self._store_chunks(chunks, directory_document_id)
        
        result.update({
            "files_processed": len(file_paths),
            "chunks_created": len(chunks),
            "directory_stats": stats
        })
        
        return result
    
    def _store_chunks(self, chunks: List[Dict], document_id: str) -> Dict:
        """Generate embeddings and store chunks in vector database."""
        if not chunks:
            return {"status": "error", "message": "No chunks to store"}
        
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            # Generate embeddings in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk['content'] for chunk in batch_chunks]
                
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                
                batch_ids = []
                batch_metadatas = []
                
                for chunk in batch_chunks:
                    # Determine a unique identifier for this chunk.
                    # Priority: explicit chunk_id, then table_id (for tables), otherwise auto-generated UUID.
                    meta = chunk['metadata']
                    raw_chunk_id = (
                        meta.get('chunk_id') or
                        meta.get('table_id') or
                        f"auto_{uuid4()}"
                    )

                    # Combine with source to make the final document ID and sanitize it.
                    chunk_id = f"{meta.get('source', 'unknown')}_{raw_chunk_id}"
                    sanitized_id = chunk_id.replace('\\', '/').replace(' ', '_').replace(':', '_')
                    batch_ids.append(sanitized_id)
                    
                    # Attach/update document_id in metadata
                    metadata = chunk['metadata']
                    metadata['document_id'] = document_id
                    
                    # Simple and safe sanitization
                    sanitized_metadata = {k: str(v) for k, v in metadata.items()}
                    batch_metadatas.append(sanitized_metadata)
                
                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.info(f"Stored batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            total_docs = self.collection.count()
            logger.info(f"Successfully stored {len(chunks)} chunks. Total documents: {total_docs}")
            
            return {
                "status": "success",
                "message": f"Indexed {len(chunks)} chunks successfully",
                "total_documents": total_docs
            }
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            return {"status": "error", "message": f"Failed to store chunks: {str(e)}"}
    
    def search_similar(self, query: str, document_id: str, n_results: int = None) -> List[Dict]:
        """Search for similar documents using vector similarity."""
        n_results_to_fetch = self.max_retrieval_results
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            query_embedding = response.data[0].embedding
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results_to_fetch,
                where={"document_id": document_id},
                include=["metadatas", "documents", "distances"]
            )
            
            if not results or not results.get('ids', [[]])[0]:
                logger.info("No similar documents found in vector search.")
                return []
            
            similar_chunks = []
            for i in range(len(results['ids'][0])):
                similar_chunks.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] or {},
                    "distance": results['distances'][0][i]
                })

            similar_chunks.sort(key=lambda x: x.get('distance', 1e9))

            return similar_chunks[:self.max_retrieval_results]

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def generate_response(self, query: str, document_id: str, chat_history: List[Dict] = None) -> Dict:
        """Generate a response using RAG with optional query rewriting."""

        rewritten_query = query
        if chat_history:
            try:
                rewritten_query = self._rewrite_query(query, chat_history)
                logger.info(f"Rewritten query: '{rewritten_query}' (from '{query}')")
            except Exception as e:
                logger.warning(f"Query rewriting failed, falling back to original query: {e}")

        context_chunks = self.search_similar(query=rewritten_query, document_id=document_id)
        
        if not context_chunks:
            return {"answer": "I could not find any relevant information in the document to answer your question.", "sources": []}
        
        context = "\n\n---\n\n".join(
            f"From {chunk['metadata'].get('source', 'Unknown')} (page {chunk['metadata'].get('page', 'N/A')}):\n{chunk['content']}"
            for chunk in context_chunks
        )
        
        system_prompt = f"""
You are an intelligent assistant designed to answer questions based on a provided document.
Your task is to synthesize information from the 'DOCUMENT CONTEXT' section to answer the user's 'QUERY'.
When you form your answer, you MUST adhere to the following rules:
1.  Base your answer *exclusively* on the information found in the 'DOCUMENT CONTEXT'. Do not use any external knowledge or make assumptions.
2.  If the context does not contain the answer, state clearly: "The provided document does not contain information on this topic."
3.  Be concise and directly address the user's query.
4.  Do not repeat the user's query in your response.
5.  You MUST cite all information used from the context. Citations must be inline and in this exact format: (filename.ext, page X) for a single page or (filename.ext, pages X-Y) for a range of pages. Use the exact filename from the context. This is a strict requirement.

DOCUMENT CONTEXT:
---
{context}
---
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": f"QUERY: {query}"})

        total_chars = sum(len(m['content']) for m in messages)
        while total_chars > 400000 and len(messages) > 2:
            removed = messages.pop(1)
            total_chars -= len(removed['content'])

        try:
            logger.info("Generating response from OpenAI...")
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content

            fallback_phrases = [
                "i could not find any relevant information",
                "does not contain information"
            ]
            lower_answer = answer.lower()
            if any(p in lower_answer for p in fallback_phrases):
                return {"answer": answer, "sources": []}

            unique_sources = set()
            for chunk in context_chunks:
                metadata = chunk.get('metadata', {})
                src = metadata.get('source', 'N/A')
                unique_sources.add(src)

            sources = [{'source': src} for src in sorted(unique_sources)]

            unique_context_filenames = {Path(s['source']).name for s in sources}
            single_source_mode = len(unique_context_filenames) == 1
            the_only_filename = next(iter(unique_context_filenames)) if single_source_mode else None

            cited_filenames = set()
            page_pattern = re.compile(r'\((?:([^,]+?)\s*,\s*)?pages?\s*([\d\s,-]+)\.?\s*\)', re.IGNORECASE)
            for match in page_pattern.findall(answer):
                filename, pages = match
                
                if filename:
                    normalized_filename = Path(filename.strip()).name
                    cited_filenames.add(normalized_filename.lower())
                elif single_source_mode:
                    cited_filenames.add(the_only_filename.lower())

            filtered_sources = [
                s for s in sources if Path(s['source']).name.lower() in cited_filenames
            ]

            return {"answer": answer, "sources": filtered_sources}
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {str(e)}")
            return {"answer": "There was an error generating the response.", "sources": []}
        
    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        try:
            total_docs = self.collection.count()
            
            # Get sample of metadata to analyze content types
            if total_docs > 0:
                sample_results = self.collection.get(
                    limit=min(1000, total_docs),
                    include=['metadatas']
                )
                
                content_types = {}
                sources = set()
                
                for metadata in sample_results['metadatas']:
                    content_type = metadata.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    sources.add(metadata.get('source', 'unknown'))
                
                return {
                    "total_documents": total_docs,
                    "unique_sources": len(sources),
                    "content_types": content_types,
                    "database_path": self.persist_directory
                }
            else:
                return {
                    "total_documents": 0,
                    "unique_sources": 0,
                    "content_types": {},
                    "database_path": self.persist_directory
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_database(self) -> Dict:
        """Clear all documents from the database."""
        try:
            # Get all document IDs and delete them
            all_docs = self.collection.get()
            if all_docs and 'ids' in all_docs and all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"Cleared {len(all_docs['ids'])} documents from vector database")
            else:
                logger.info("Vector database is already empty")
            
            self.document_processor.processed_files.clear()
            logger.info("Database cleared successfully")
            return {"status": "success", "message": "Database cleared"}
        except Exception as e:
            logger.warning(f"Standard clear failed: {e}")
            try:
                # Fallback: try to delete the collection and recreate it
                collection_name = self.collection.name
                self.chroma_client.delete_collection(name=collection_name)
                self.collection = self.chroma_client.create_collection(name=collection_name)
                self.document_processor.processed_files.clear()
                logger.info("Database cleared by recreating collection")
                return {"status": "success", "message": "Database cleared (recreated collection)"}
            except Exception as e2:
                logger.error(f"Error clearing database: {str(e2)}")
                return {"status": "error", "message": str(e2)}
    
    def clear_docs_folder(self, docs_dir: str) -> Dict:
        """Deletes all files in the specified directory."""
        try:
            for file_path in Path(docs_dir).glob("**/*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info(f"All files in '{docs_dir}' have been deleted.")
            return {"status": "success", "message": f"All files in '{docs_dir}' have been deleted."}
        except Exception as e:
            logger.error(f"Error deleting files from '{docs_dir}': {e}")
            return {"status": "error", "message": str(e)}

    def reset_database(self):
        """Reset the entire database connection."""
        try:
            logger.info("Resetting database connection...")
            self._init_vector_db()
            logger.info("Database connection reset successfully")
            return {"status": "success", "message": "Database connection reset"}
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return {"status": "error", "message": str(e)} 

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rewrite_query(self, user_query: str, chat_history: List[Dict]) -> str:
        """Rewrite a follow-up question into a stand-alone search query using the last few turns."""

        relevant_history = chat_history[:]
        while relevant_history:
            history_text = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in relevant_history
            )
            if len(history_text) <= 8000:
                break
            relevant_history.pop(0)
        
        prompt = (
            "You are an assistant that reformulates follow-up questions into a self-contained, clear search query.\n"
            "Conversation so far:\n"
            f"{history_text}\n\n"
            "Follow-up question: \"" + user_query + "\"\n"
            "Rewrite the follow-up question so that it can be understood without the conversation context.\n"
            "Return ONLY the rewritten question."
        )

        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else user_query 