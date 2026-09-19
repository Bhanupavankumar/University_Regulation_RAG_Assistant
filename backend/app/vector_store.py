"""Vector Database and Search Index management mirroring FAISS in Colab"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class VectorStore:
    def __init__(self, db_dir: str = "data/vector_db"):
        self.db_dir = db_dir
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype='float32')
        self.faiss_index = None
        self.use_faiss = False
        self.chroma_client = None
        self.chroma_collection = None
        
        # Check for faiss-cpu
        try:
            import faiss
            self.faiss = faiss
            self.use_faiss = True
        except ImportError:
            self.faiss = None
            self.use_faiss = False

        # Check for chromadb
        try:
            import chromadb
            chroma_path = os.path.join(self.db_dir, "chroma")
            os.makedirs(chroma_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="university_regulations",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            self.chroma_client = None
            self.chroma_collection = None

        os.makedirs(self.db_dir, exist_ok=True)
        self.load_if_exists()

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Builds index from chunks and embedding vectors."""
        self.metadata = chunks
        self.vectors = embeddings.astype('float32')
        
        if self.use_faiss and len(embeddings) > 0:
            dim = embeddings.shape[1]
            # Normalize for cosine similarity inner product
            self.faiss_index = self.faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.vectors)
        
        self.save()

    def add_documents(self, new_chunks: List[Dict[str, Any]], new_embeddings: np.ndarray):
        """Appends new chunks and vectors to existing index."""
        if len(self.metadata) == 0:
            self.build_index(new_chunks, new_embeddings)
            return

        self.metadata.extend(new_chunks)
        self.vectors = np.vstack([self.vectors, new_embeddings.astype('float32')])
        
        if self.use_faiss:
            dim = self.vectors.shape[1]
            self.faiss_index = self.faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.vectors)
        
        self.save()

    def remove_document(self, filename: str):
        """Removes all chunks associated with a filename and rebuilds index."""
        keep_indices = [i for i, c in enumerate(self.metadata) if c.get('file') != filename]
        
        if len(keep_indices) == len(self.metadata):
            return # Nothing to remove
            
        if len(keep_indices) == 0:
            self.metadata = []
            self.vectors = np.empty((0, 384), dtype='float32')
            self.faiss_index = None
        else:
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.vectors = self.vectors[keep_indices]
            if self.use_faiss:
                dim = self.vectors.shape[1]
                self.faiss_index = self.faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.vectors)
                
        self.save()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        category: Optional[str] = "all"
    ) -> List[Dict[str, Any]]:
        """
        Searches the index with query vector.
        Returns top matching chunks with similarity score in [0.0, 1.0].
        """
        if len(self.metadata) == 0 or len(self.vectors) == 0:
            return []

        # Ensure 2D query shape
        if query_vector.ndim == 1:
            q_vec = query_vector.reshape(1, -1).astype('float32')
        else:
            q_vec = query_vector.astype('float32')

        # Fast cosine similarity with NumPy or FAISS
        # Cosine similarity is dot product of L2-normalized vectors
        scores = np.dot(self.vectors, q_vec.T).flatten() # shape: (N,)
        
        # Apply category filter if specified
        valid_indices = []
        for idx, item in enumerate(self.metadata):
            if category and category.lower() != "all":
                if item.get("category", "").lower() != category.lower():
                    continue
            valid_indices.append(idx)

        if not valid_indices:
            return []

        filtered_scores = scores[valid_indices]
        # Rank top_k
        k = min(top_k, len(valid_indices))
        top_sub_indices = np.argsort(-filtered_scores)[:k]
        
        results = []
        for sub_idx in top_sub_indices:
            orig_idx = valid_indices[sub_idx]
            raw_score = float(scores[orig_idx])
            # Direct Cosine similarity in [0.0, 1.0]
            sim_score = max(0.0, min(1.0, raw_score))
            
            chunk = self.metadata[orig_idx]
            results.append({
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "file": chunk.get("file", ""),
                "page": chunk.get("page", 1),
                "category": chunk.get("category", "General"),
                "url": chunk.get("url", ""),
                "score": round(sim_score, 4)
            })

        return results

    def save(self):
        """Saves vector database matching Colab notebook format and updates Chroma collection."""
        faiss_path = os.path.join(self.db_dir, "index.faiss")
        meta_path = os.path.join(self.db_dir, "metadata.pkl")
        vec_path = os.path.join(self.db_dir, "vectors.npy")
        
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        np.save(vec_path, self.vectors)
        
        if self.use_faiss and self.faiss_index is not None:
            self.faiss.write_index(self.faiss_index, faiss_path)

        # Sync to ChromaDB persistent collection
        if self.chroma_collection is not None and len(self.metadata) > 0 and len(self.vectors) > 0:
            try:
                # Upsert all chunks into ChromaDB
                ids = [f"chunk_{i}" for i in range(len(self.metadata))]
                documents = [c.get("text", "") for c in self.metadata]
                metadatas = [{
                    "source": str(c.get("source", "")),
                    "file": str(c.get("file", "")),
                    "page": int(c.get("page", 1)),
                    "category": str(c.get("category", "General"))
                } for c in self.metadata]
                embeddings_list = self.vectors.tolist()

                self.chroma_collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings_list,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"[VectorStore] Chroma sync notice: {e}")

    def load_if_exists(self):
        """Loads saved vector database from disk if found."""
        meta_path = os.path.join(self.db_dir, "metadata.pkl")
        vec_path = os.path.join(self.db_dir, "vectors.npy")
        faiss_path = os.path.join(self.db_dir, "index.faiss")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"[VectorStore] Loaded {len(self.metadata)} chunks from {meta_path}")
            except Exception as e:
                print(f"[VectorStore] Error loading metadata: {e}")

        if os.path.exists(vec_path):
            try:
                self.vectors = np.load(vec_path)
                print(f"[VectorStore] Loaded vector matrix with shape {self.vectors.shape}")
            except Exception as e:
                print(f"[VectorStore] Error loading vectors.npy: {e}")

        if self.use_faiss and os.path.exists(faiss_path):
            try:
                self.faiss_index = self.faiss.read_index(faiss_path)
                print(f"[VectorStore] Loaded FAISS index from {faiss_path}")
            except Exception as e:
                print(f"[VectorStore] Error loading index.faiss: {e}")
