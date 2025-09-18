"""
RAG Engine Module
Handles vector storage, retrieval, and answer generation
"""

import os
import shutil
import time
from typing import Dict
from pathlib import Path

# Import required libraries with error handling
try:
    # Use working imports based on testing
    from langchain_community.embeddings.openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain.llms import OpenAI
    from langchain_community.chat_models import ChatOpenAI
    # Try to import RetrievalQA, but don't fail if it doesn't work
    try:
        from langchain.chains import RetrievalQA
    except ImportError:
        RetrievalQA = None
        print("Warning: RetrievalQA not available, will use alternative approach")
except ImportError:
    try:
        # Fallback to older imports
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI
        from langchain.schema import Document
    except ImportError as e:
        print(f"Warning: LangChain not available: {e}")
        OpenAIEmbeddings = None
        Chroma = None
        RetrievalQA = None
        OpenAI = None
        Document = None

try:
    import chromadb
except ImportError:
    chromadb = None

from config import config
from document_parser import DocumentParser
from logging_config import app_logger, perf_logger, user_logger, error_logger


class ArxivRAGEngine:
    """Handles RAG operations for arXiv documents"""

    def __init__(self, persist_directory: str = None):
        """
        Initialize RAG Engine with ChromaDB and OpenAI integration

        Args:
            persist_directory: Directory to persist ChromaDB data (default from config)
        """
        self.persist_directory = (
            persist_directory or config.chromadb.persist_directory
        )
        self.collection_name = config.chromadb.collection_name

        # Initialize logger
        self.logger = app_logger.getChild('rag_engine')

        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.parser = None

        # Initialize vectorstore and components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize OpenAI embeddings, LLM, and vectorstore"""
        try:
            # Check if OpenAI API key is available
            if not config.openai_api_key:
                raise ValueError(
                    "OpenAI API key is required. "
                    "Set OPENAI_API_KEY environment variable."
                )

            # Initialize OpenAI embeddings
            if OpenAIEmbeddings:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=config.openai_api_key,
                    openai_organization=config.openai_organization
                )
                self.logger.info("OpenAI embeddings initialized")
            else:
                raise ImportError("LangChain OpenAIEmbeddings not available")

            # Initialize OpenAI LLM
            if ChatOpenAI:
                # Use ChatOpenAI for chat models
                self.llm = ChatOpenAI(
                    openai_api_key=config.openai_api_key,
                    openai_organization=config.openai_organization,
                    model_name=config.rag.model_name,
                    temperature=config.rag.temperature,
                    max_tokens=config.rag.max_tokens
                )
                self.logger.info(
                    f"OpenAI Chat LLM initialized with model: {config.rag.model_name}"
                )
            elif OpenAI:
                # Fallback to old OpenAI for completion models
                self.llm = OpenAI(
                    openai_api_key=config.openai_api_key,
                    openai_organization=config.openai_organization,
                    model_name=config.rag.model_name,
                    temperature=config.rag.temperature,
                    max_tokens=config.rag.max_tokens
                )
                self.logger.info(
                    f"OpenAI LLM initialized with model: {config.rag.model_name}"
                )
            else:
                raise ImportError("LangChain OpenAI not available")

            # Initialize document parser
            self.parser = DocumentParser()
            self.logger.info("Document parser initialized")

            # Initialize or load existing vectorstore
            self._initialize_vectorstore()

        except Exception as e:
            error_msg = f"Failed to initialize RAG Engine components: {e}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise

    def _initialize_vectorstore(self):
        """Initialize or load existing ChromaDB vectorstore"""
        try:
            if not Chroma:
                raise ImportError("LangChain Chroma not available")

            if not chromadb:
                raise ImportError("ChromaDB not available")

            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

            # Create QA chain
            if RetrievalQA:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": config.rag.top_k}
                    ),
                    return_source_documents=True
                )
                self.logger.info("RetrievalQA chain initialized")
            else:
                # Fallback: create a simple retriever without QA chain
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": config.rag.top_k}
                )
                self.qa_chain = None
                self.logger.info("Simple retriever initialized (RetrievalQA not available)")

            self.logger.info(
                f"ChromaDB vectorstore initialized at: {self.persist_directory}"
            )

        except Exception as e:
            error_msg = f"Failed to initialize vectorstore: {e}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise

    def add_documents(self, papers_dir: str):
        """
        Add all PDF documents from directory to vectorstore

        Args:
            papers_dir: Directory containing PDF files
        """
        self.logger.info(f"Starting document addition from: {papers_dir}")
        user_logger.info(f"Adding documents from directory: {papers_dir}")

        papers_path = Path(papers_dir)
        if not papers_path.exists():
            error_msg = f"Papers directory does not exist: {papers_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        documents = []
        metadatas = []
        total_chunks = 0

        try:
            # Process all PDF files in the directory
            pdf_files = list(papers_path.glob("*.pdf"))

            if not pdf_files:
                self.logger.warning(f"No PDF files found in {papers_dir}")
                return

            self.logger.info(f"Found {len(pdf_files)} PDF files to process")

            for pdf_file in pdf_files:
                try:
                    self.logger.info(f"Processing PDF: {pdf_file.name}")

                    # Parse document into chunks
                    chunks = self.parser.parse_document(str(pdf_file))

                    if not chunks:
                        self.logger.warning(f"No chunks extracted from {pdf_file.name}")
                        continue

                    # Get document metadata
                    metadata = self.parser.get_document_metadata(str(pdf_file))

                    # Create documents for each chunk
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_id": i,
                                "chunk_count": len(chunks),
                                "document_id": pdf_file.stem
                            }
                        )
                        documents.append(doc)
                        metadatas.append({
                            **metadata,
                            "chunk_id": i,
                            "chunk_count": len(chunks),
                            "document_id": pdf_file.stem
                        })

                    total_chunks += len(chunks)
                    self.logger.info(f"Added {len(chunks)} chunks from {pdf_file.name}")

                except Exception as e:
                    error_msg = f"Error processing {pdf_file.name}: {e}"
                    self.logger.error(error_msg)
                    error_logger.error(error_msg, exc_info=True)
                    continue

            # Add documents to vectorstore
            if documents:
                self.logger.info(
                    f"Adding {len(documents)} total chunks to vectorstore"
                )

                # Add documents in batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    self.vectorstore.add_documents(batch_docs)
                    self.logger.info(
                        f"Added batch {i//batch_size + 1}/"
                        f"{(len(documents)-1)//batch_size + 1}"
                    )

                # Persist the vectorstore
                self.vectorstore.persist()

                self.logger.info(
                    f"Successfully added {total_chunks} chunks from "
                    f"{len(pdf_files)} documents"
                )
                user_logger.info(
                    f"Successfully added {total_chunks} chunks from "
                    f"{len(pdf_files)} documents"
                )
            else:
                self.logger.warning("No documents were added to vectorstore")

        except Exception as e:
            error_msg = f"Failed to add documents: {e}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise

    def search(self, query: str) -> Dict:
        """
        Search and answer query using RAG

        Args:
            query: Search query string

        Returns:
            Dictionary containing answer and sources
        """
        start_time = time.time()
        self.logger.info(f"Starting search for query: {query[:50]}...")
        user_logger.info(f"User search query: {query}")

        try:
            if not self.qa_chain and not hasattr(self, 'retriever'):
                return {"error": "QA chain and retriever not initialized"}

            # Perform search using QA chain or simple retriever
            if self.qa_chain:
                result = self.qa_chain({"query": query})
            else:
                # Fallback: use simple retriever and manual LLM call
                docs = self.retriever.get_relevant_documents(query)
                if not docs:
                    return {"error": "No relevant documents found"}
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Simple prompt for LLM
                prompt = f"""Based on the following context, answer the question: {query}

Context:
{context}

Answer:"""
                
                # Call LLM directly
                if hasattr(self.llm, 'invoke'):
                    # ChatOpenAI uses invoke method
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        response = response.content
                else:
                    # Old OpenAI uses call method
                    response = self.llm(prompt)
                
                # Format result to match QA chain output
                result = {
                    "result": response,
                    "source_documents": docs
                }

            response_time = time.time() - start_time

            # Log performance metrics
            perf_logger.info(f"Search completed in {response_time:.2f}s")
            perf_logger.info(f"Query: {query[:50]}...")

            if response_time > 10:
                self.logger.warning(f"Slow search detected: {response_time:.2f}s")

            # Format response
            response = {
                "answer": result["result"],
                "sources": []
            }

            # Extract source information
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                        "file": doc.metadata.get("file", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "document_id": doc.metadata.get("document_id", "Unknown"),
                        "filename": doc.metadata.get("filename", "Unknown")
                    }
                    response["sources"].append(source_info)

            self.logger.info("Search completed successfully")
            user_logger.info(f"Search completed for: {query}")

            return response

        except Exception as e:
            error_msg = f"Search failed for query '{query}': {e}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    def get_document_count(self) -> int:
        """
        Get total number of documents in vectorstore

        Returns:
            Number of documents in the vectorstore
        """
        try:
            if not self.vectorstore:
                return 0

            # Get collection count
            collection = self.vectorstore._collection
            count = collection.count()

            self.logger.info(f"Document count: {count}")
            return count

        except Exception as e:
            error_msg = f"Failed to get document count: {e}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
        return 0

    def clear_database(self):
        """Clear the vector database"""
        self.logger.info("Starting database clearing")
        user_logger.info("User cleared database")

        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                self.logger.info(f"Removed directory: {self.persist_directory}")

            # Reinitialize vectorstore
            self._initialize_vectorstore()

            self.logger.info("Database cleared successfully")
            user_logger.info("Database cleared successfully")

        except Exception as e:
            error_msg = f"Failed to clear database: {e}"
            self.logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise

    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection

        Returns:
            Dictionary with collection information
        """
        try:
            if not self.vectorstore:
                return {"error": "Vectorstore not initialized"}

            collection = self.vectorstore._collection
            count = collection.count()

            info = {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "document_count": count,
                "embedding_model": "OpenAI",
                "llm_model": config.rag.model_name,
                "top_k": config.rag.top_k
            }

            return info

        except Exception as e:
            error_msg = f"Failed to get collection info: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}
