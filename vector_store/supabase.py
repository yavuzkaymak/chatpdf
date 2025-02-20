from config import SupaBase
from supabase.client import Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.schema.embeddings import Embeddings

class SupabaseConnection:
    def __init__(self, embedding: Embeddings, chunk_size: int = 1000):
        self.url = SupaBase.URL
        self.service_key = SupaBase.SERVICE_KEY
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.client = None
        self.vector_store = None
        self._set_client()
        self._set_vector_store()

    def _set_client(self):
        self.client = Client(self.url, self.service_key)

    def _set_vector_store(self) -> None:
        self.vector_store = SupabaseVectorStore(
                                client=self.client,
                                embedding=self.embedding,
                                chunk_size=1000,
                                query_name=SupaBase.QUERY_NAME, 
                                table_name=SupaBase.TABLE_NAME
                            )