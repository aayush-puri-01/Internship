import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.schema.document import Document
from typing import List


class QueryBasedReranker:
    def __init__(self, llm_model: str = "deepseek-r1:1.5b", embedding_model:str = "nomic-embed-text"):
        self.llm = OllamaLLM(model=llm_model)
        self.embedder = OllamaEmbeddings(model=embedding_model)

    def generate_chunk_based_queries(self, chunks):
        chunk_queries = []
        chunk_list = []
        for chunk in chunks:
            if isinstance(chunk, Document):
                content = chunk.page_content
            else:
                content = chunk

            chunk_list.append(chunk)
            
            prompt = f"""Generate a concise one line question from the content provided.
            Context: {content}
            Question:
            """

            response = self.llm.invoke(prompt)

            chunk_queries.append(response.strip())

        return chunk_queries, chunk_list
    
    def generate_embeddings(self, user_query: str, query_list: List[str]):
        user_query_embedding = self.embedder.embed_query(user_query)
        query_list_embeddings = self.embedder.embed_documents(query_list)
        return user_query_embedding, query_list_embeddings
    
    def calculate_similarity_scores(self, user_query_vector, chunk_query_vector):
        user_query_vector = np.array(user_query_vector)
        user_query_mag = np.linalg.norm(user_query_vector)
        similarity_scores = []
        for vector in chunk_query_vector:
            chunk_vector = np.array(vector)
            chunk_vector_mag = np.linalg.norm(chunk_vector)
            score = np.dot(user_query_vector, chunk_vector)
            score = score / (user_query_mag * chunk_vector_mag + 1e-5)
            similarity_scores.append(score)
        return similarity_scores
    
    def rerank_chunks(self, scores, chunks, top_k:int):
        if top_k > len(scores):
            raise ValueError("k greater than number of retrieved documents")
        else:
            sorted_indices = np.argsort(scores)[::-1][:top_k]
        reranked_chunks = [chunks[i] for i in sorted_indices]
        
        return reranked_chunks




