from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import json
from pydantic import BaseModel
from rag_pipeline.rag import RAGPipeline
from rag_pipeline.reranker import QueryBasedReranker
import traceback

class QueryRequest(BaseModel):
    user_query: str

"""
The input must be structured as a json 

{
    "user_query" : "what is attention in transformers?"
}
"""

class QueryResponse(BaseModel):
    answer: str


pipeline = RAGPipeline(
    llm_model="deepseek-r1:1.5b",
    use_reranker=True,
    one_liner=False
)

reranker = QueryBasedReranker(
    llm_model="deepseek-r1:1.5b",
    embedding_model="nomic-embed-text"
)

print("\n[INFO] RAG Pipeline and Reranker have been initialized\n")

pdf_text = pipeline.load_pdf_doc("rag_pipeline/Attention.pdf")

#Each chunk has 500 tokens and an overlap of 50 tokens is maintained
chunks = pipeline.split_document(pdf_text)

#Initializing a vector database, if already exists in the persist directory, the existing chroma vectorstore is loaded
pipeline.get_chunk_embeddings(chunks)


router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):

    try:
        if not request:
            return JSONResponse(content={"error":"Query is required"}, status_code=400)

        def rag_stream():
            output = None
            try:
                if pipeline.use_reranker == False:

                    if pipeline.one_liner == True:
                        output = pipeline.respond_one_liner("rag_pipeline/prompt_templates_oneline.txt")

                    else:
                        output = pipeline.respond_paragraph("rag_pipeline/prompt_templates.txt")  

                else: 

                    retriever = pipeline.initialize_retriever(k=5)

                    chunks = retriever.invoke(request.user_query)

                    queries_for_chunks, chunks = reranker.generate_chunk_based_queries(chunks)

                    question_embedding, chunk_queries_embeddings = reranker.generate_embeddings(request.user_query, queries_for_chunks)

                    scores = reranker.calculate_similarity_scores(question_embedding, chunk_queries_embeddings)

                    reranked_chunks = reranker.rerank_chunks(scores, chunks, top_k = 3)

                    if pipeline.one_liner == True:
                        prompt_template = pipeline.load_prompt_template(
                            "rag_pipeline/prompt_templates_oneline.txt",
                        )
                    else:
                        prompt_template = pipeline.load_prompt_template(
                            "rag_pipeline/prompt_templates.txt",
                        )           
                    output = pipeline.reranker_build_and_respond(reranked_chunks, prompt_template, user_question=request.user_query)

                yield json.dumps({"answer":output})

            except Exception as e:
                yield json.dumps({"error": "RAG Processing failed"})
            
        return StreamingResponse(rag_stream(), media_type="application/json")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error" : str(e)}, status_code=500)



