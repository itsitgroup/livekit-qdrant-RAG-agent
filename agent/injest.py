import os
import glob
import re
import uuid
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from livekit.plugins.rag import SentenceChunker

import openai

load_dotenv()

class QdrantIngestor:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_ENDPOINT"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Collection name for storing knowledge base
        self.collection_name = "knowledge_base"
        self.chunker = SentenceChunker(max_chunk_size=500, chunk_overlap=50)
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self._init_collection()

    def _init_collection(self):
        try:
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.vectors_config.params.size != 1536:
                raise ValueError("Collection exists with wrong vector size")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance="Cosine"
                )
            )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_embedding(self, text: str) -> List[float]:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def process_file(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = self._clean_text(f.read())
            
            chunks = self.chunker.chunk(text=text)
            points = []
            
            for idx, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)
                
                point_id = str(uuid.uuid4())
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": os.path.basename(file_path),
                            "chunk_index": idx
                        }
                    )
                )
            
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                print(f"Uploaded batch {i//batch_size + 1} of {len(points)//batch_size + 1}")
            
            print(f"Successfully processed {len(chunks)} chunks from {file_path}")
            return True
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
            return False

if __name__ == "__main__":
    ingestor = QdrantIngestor()
    
    text_files = glob.glob("data/itsitgroup-aboutus.txt", recursive=True)
    
    if not text_files:
        print("No text files found in data/ directory")
        exit(1)
        
    for file_path in text_files:
        try:
            ingestor.process_file(file_path)
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
    
    print(f"Completed ingestion of {len(text_files)} files")
