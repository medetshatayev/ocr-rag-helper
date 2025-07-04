import os
import logging
from typing import List, Dict, Optional, Tuple

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
        self.max_retrieval_results = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
        
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
        
        # Generate embeddings and store
        result = self._store_chunks(chunks)
        
        result.update({
            "files_processed": len(file_paths),
            "chunks_created": len(chunks),
            "directory_stats": stats
        })
        
        return result
    
    def _store_chunks(self, chunks: List[Dict]) -> Dict:
        """Generate embeddings and store chunks in vector database."""
        if not chunks:
            return {"status": "error", "message": "No chunks to store"}
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            # Generate embeddings in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk['content'] for chunk in batch_chunks]
                
                # Generate embeddings
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                
                # Prepare batch data
                for j, chunk in enumerate(batch_chunks):
                    chunk_id = f"{chunk['metadata']['source']}_{chunk['metadata'].get('chunk_id', i+j)}"
                    chunk_id = chunk_id.replace('\\', '/').replace(' ', '_')  # Clean ID
                    
                    documents.append(chunk['content'])
                    metadatas.append(chunk['metadata'])
                    ids.append(chunk_id)
                
                # Store batch in ChromaDB
                batch_ids = []
                batch_metadatas = []
                
                for j, chunk in enumerate(batch_chunks):
                    # Create clean ID
                    chunk_id = f"{chunk['metadata']['source']}_{chunk['metadata'].get('chunk_id', i+j)}"
                    chunk_id = chunk_id.replace('\\', '/').replace(' ', '_').replace(':', '_')
                    batch_ids.append(chunk_id)
                    
                    # Clean metadata - remove any problematic keys
                    clean_metadata = {}
                    for key, value in chunk['metadata'].items():
                        if isinstance(value, (str, int, float, bool)) and key != '_type':
                            clean_metadata[key] = value
                        elif isinstance(value, list):
                            clean_metadata[key] = str(value)  # Convert lists to strings
                    batch_metadatas.append(clean_metadata)
                
                try:
                    self.collection.add(
                        documents=batch_texts,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                except Exception as e:
                    # Fallback: try without embeddings (let ChromaDB generate them)
                    logger.warning(f"Failed to add with embeddings, trying without: {e}")
                    self.collection.add(
                        documents=batch_texts,
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
    
    def search_similar(self, query: str, n_results: int = None) -> List[Dict]:
        """Search for similar documents using vector similarity."""
        # The user wants to get 10 sources and then filter to the top 5
        n_results_to_fetch = 10
        
        try:
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results_to_fetch,
                include=["metadatas", "documents", "distances"] 
            )
            
            if not results or not results.get('ids', [[]])[0]:
                logger.info("No similar documents found in vector search.")
                return []
            
            # Extract and combine results with their distances (similarity scores)
            combined_results = []
            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i in range(len(ids)):
                # Ensure metadata is a dictionary
                metadata = metadatas[i] if isinstance(metadatas[i], dict) else {}
                
                combined_results.append({
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadata,
                    "distance": distances[i]
                })

            # Sort by distance (lower is better) and take top 5
            combined_results.sort(key=lambda x: x['distance'])
            top_5_results = combined_results[:5]

            # Format results for consumption
            final_sources = []
            for res in top_5_results:
                source_info = {
                    "file": res["metadata"].get("source", "Unknown"),
                    "page": res["metadata"].get("page", None),
                    "content": res["document"],
                    "similarity_score": 1 - res["distance"]  # Convert distance to similarity
                }
                final_sources.append(source_info)
                
            return final_sources

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def generate_response(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Generate response using RAG pipeline."""
        try:
            # Search for relevant documents
            relevant_docs = self.search_similar(query)
            
            if not relevant_docs:
                return {
                    "response": "I couldn't find any relevant information in the indexed documents to answer your question.",
                    "sources": [],
                    "context_used": False
                }
            
            # Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for doc in relevant_docs:
                source_file = doc['file']
                page_num = doc['page']
                file_name = os.path.basename(source_file)  # Extract just the filename
                context_parts.append(f"Document {len(context_parts) + 1} (Source: {file_name}, Page: {page_num}):\n{doc['content']}")
                
                # Add to sources
                source_info = {
                    'file': source_file,
                    'page': page_num,
                    'content_type': 'text',
                    'similarity_score': doc['similarity_score']
                }
                sources.append(source_info)
            
            formatted_context = "\n\n".join(context_parts)
            
            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions based on the provided context from documents. 

INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite the source (file name and page number when available) for your information
4. If you find table data, present it in a clear, structured format
5. Be precise and factual - don't make assumptions beyond what's in the context
6. If multiple sources provide relevant information, reference all of them"""
                }
            ]
            
            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history[-10:])  # Keep last 10 messages for context
            
            # Add current query with context
            user_message = f"""Context from indexed documents:
{formatted_context}

Question: {query}

Please answer this question based on the provided context."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Format sources for display in the UI
            sources = []
            for doc in relevant_docs:
                source_file = doc['file']
                page_num = doc['page']
                
                sources.append({
                    "file": source_file,
                    "page": page_num,
                    "content": doc['content'],
                    "similarity_score": doc.get('similarity_score', 0.0)
                })
            
            return {
                "response": assistant_response,
                "sources": sources,
                "context_used": True,
                "retrieved_chunks": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_used": False
            }
    
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
    
    def clear_database(self):
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