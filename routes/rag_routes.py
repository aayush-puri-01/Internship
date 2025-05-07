from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
import json
from pydantic import BaseModel
from rag_pipeline.rag import RAGPipeline
from rag_pipeline.reranker import QueryBasedReranker
import traceback
from rag_pipeline.redis_rag import CacheSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

async def stream_response(generator):
    async for chunk in generator:
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

pipeline = RAGPipeline(
    llm_model="deepseek-r1:1.5b",
    use_reranker=False,
    one_liner=True
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

cache = CacheSystem()
print(f"\nInitially Cache Status: \n")
print(cache.redis.keys())

def check_and_get_cache(question:str, format:str):
    return_tuple = cache.get_cached_response(question, response_format=format)

    if return_tuple is not None:
        logger.info("\nCache Hit! Using cached response!\n")
        return return_tuple[0], return_tuple[1]
    else:
        logger.info("\nCache Miss, running the RAG with LLM\n")
        return None, None
    


router = APIRouter()

@router.post("/query/one-liner", response_model=QueryResponse)
async def query_one_liner(request: QueryRequest, stream: bool = Query(False)):
    try:
        response_format = "single_line"

        if pipeline.use_reranker:
            logger.info("Using reranker for one-liner query")

            retriever = pipeline.initialize_retriever(k=5)
            ans, check_if_reranked = check_and_get_cache(request.user_query, response_format)

            if ans is not None and check_if_reranked == "True":
                return QueryResponse(answer=ans)

            chunks = retriever.invoke(request.user_query)

            queries_for_chunks, chunks = reranker.generate_chunk_based_queries(chunks)

            question_embedding, chunk_queries_embeddings = reranker.generate_embeddings(request.user_query, queries_for_chunks)

            scores = reranker.calculate_similarity_scores(question_embedding, chunk_queries_embeddings)
            
            reranked_chunks = reranker.rerank_chunks(scores, chunks, top_k=3)

            prompt_template = pipeline.load_prompt_template("rag_pipeline/prompt_templates_oneline.txt")

            if stream:
                generator = pipeline.reranker_build_and_respond_stream(
                    reranked_chunks, 
                    prompt_template, 
                    user_question=request.user_query)
                
                return StreamingResponse(
                    stream_response(generator), 
                    media_type="text/event-stream")
            else:
                answer = await pipeline.reranker_build_and_respond(
                    reranked_chunks, 
                    prompt_template, 
                    user_question=request.user_query)
                
                cache.cache_response(
                    request.user_query, 
                    answer, 
                    response_format=response_format, 
                    reranked=True)
                return QueryResponse(answer=answer)

        else:
            logger.info("Using basic RAG for one-liner query")

            if stream:
                generator = pipeline.respond_one_liner_stream(
                    "rag_pipeline/prompt_templates_oneline.txt", 
                    request.user_query)
                return StreamingResponse(
                    stream_response(generator), 
                    media_type="text/event-stream")

            ans, _ = check_and_get_cache(request.user_query, response_format)

            if ans is not None:
                return QueryResponse(answer=ans)
            
            answer = await pipeline.respond_one_liner(
                "rag_pipeline/prompt_templates_oneline.txt", 
                request.user_query)
            
            cache.cache_response(
                request.user_query, 
                answer, 
                response_format=response_format, 
                reranked=False)
            
            return QueryResponse(answer=answer)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/query/paragraph", response_model=QueryResponse)
async def query_paragraph(request: QueryRequest, stream: bool = Query(False)):
    try:
        response_format = "paragraph"

        if pipeline.use_reranker:
            logger.info("Using reranker for paragraph query")

            retriever = pipeline.initialize_retriever(k=5)
            ans, check_if_reranked = check_and_get_cache(request.user_query, response_format)

            if ans is not None and check_if_reranked == "True":
                return QueryResponse(answer=ans)

            chunks = retriever.invoke(request.user_query)

            queries_for_chunks, chunks = reranker.generate_chunk_based_queries(chunks)

            question_embedding, chunk_queries_embeddings = reranker.generate_embeddings(request.user_query, queries_for_chunks)

            scores = reranker.calculate_similarity_scores(question_embedding, chunk_queries_embeddings)

            reranked_chunks = reranker.rerank_chunks(scores, chunks, top_k=3)

            prompt_template = pipeline.load_prompt_template("rag_pipeline/prompt_templates.txt")

            if stream:
                generator = pipeline.reranker_build_and_respond_stream(
                    reranked_chunks, 
                    prompt_template, 
                    user_question=request.user_query)
                
                return StreamingResponse(
                    stream_response(generator), 
                    media_type="text/event-stream")
            else:
                answer = await pipeline.reranker_build_and_respond(
                    reranked_chunks, 
                    prompt_template, 
                    user_question=request.user_query)
                
                cache.cache_response(
                    request.user_query, 
                    answer, 
                    response_format=response_format, 
                    reranked=True)
                
                return QueryResponse(answer=answer)

        else:
            logger.info("Using basic RAG for paragraph query")

            if stream:
                generator = pipeline.respond_paragraph_stream(
                    "rag_pipeline/prompt_templates.txt", 
                    request.user_query)
                return StreamingResponse(
                    stream_response(generator), 
                    media_type="text/event-stream")

            ans, _ = check_and_get_cache(request.user_query, response_format)

            if ans is not None:
                return QueryResponse(answer=ans)
            
            answer = await pipeline.respond_paragraph(
                "rag_pipeline/prompt_templates.txt", 
                request.user_query)
            
            cache.cache_response(
                request.user_query, 
                answer, 
                response_format=response_format, 
                reranked=False)
            
            return QueryResponse(answer=answer)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)






# @router.post("/query", response_model=QueryResponse)
# async def query_rag(request: QueryRequest, stream:bool = Query(False)):

#     try:
#         if not request:
#             return JSONResponse(content={"error":"Query is required"}, status_code=400)

#         if pipeline.use_reranker == False:

#             if pipeline.one_liner == True and stream == True:
#                 generator = pipeline.respond_one_liner_stream("rag_pipeline/prompt_templates_oneline.txt", request.user_query)
#                 logger.info("Launching Stream Responder for one liner response\n")
#                 return StreamingResponse(
#                     stream_response(generator),
#                     media_type="text/event-stream",
#                     headers = {
#                         "Cache-Control" : "no-cache",
#                         "Connection" : "keep-alive"
#                     }
#                 )
            
#             elif pipeline.one_liner == True and stream == False:
#                 response_format = "single_line"

#                 ans, _ = check_and_get_cache(request.user_query, response_format)

#                 if ans is not None:
#                     answer = ans
#                 else:
#                     logger.info("Launching Non-Stream Responder for one liner response\n")
#                     answer = await pipeline.respond_one_liner("rag_pipeline/prompt_templates_oneline.txt", request.user_query)
#                     cache.cache_response(
#                         request.user_query,
#                         answer, 
#                         response_format=response_format, reranked=bool(pipeline.use_reranker)
#                     )
#                     logger.info("\nNew Query/Response successfully cached!\n")
#                     print(f"Updated Cache Status:\n")
#                     print(cache.redis.keys()) 
                   
#             elif pipeline.one_liner == False and stream == True:
#                 generator = pipeline.respond_paragraph_stream("rag_pipeline/prompt_templates.txt", request.user_query)
#                 logger.info("Launching Stream Responder for Paragraph response\n")
#                 return StreamingResponse(
#                     stream_response(generator),
#                     media_type="text/event-stream",
#                     headers = {
#                         "Cache-Control" : "no-cache",
#                         "Connection" : "keep-alive"
#                     }
#                 )

#             else:
#                 response_format = "paragraph"

#                 ans, _ = check_and_get_cache(request.user_query, response_format)

#                 if ans is not None:
#                     answer = ans
#                 else:
#                     logger.info("Launching Non-Stream Responder for Paragraph response\n")
#                     answer = await pipeline.respond_paragraph("rag_pipeline/prompt_templates.txt", request.user_query)
#                     print(type(pipeline.use_reranker))
#                     cache.cache_response(
#                         request.user_query, 
#                         answer, 
#                         response_format=response_format, reranked=bool(pipeline.use_reranker)
#                     )
#                     logger.info("\nNew Query/Response successfully cached!\n")
#                     print(f"Updated Cache Status: \n")
#                     print(cache.redis.keys()) 

#         else:
#             logger.info("Starting LLM based Reranker\n")
#             retriever = pipeline.initialize_retriever(k=5)

#             ans_format = "single_line" if pipeline.one_liner else "paragraph"

#             ans, check_if_reranked = check_and_get_cache(request.user_query, ans_format)

#             if ans is not None and check_if_reranked == "True":
#                 answer = ans
#             else:
#                 logger.info("Since reranked response has not been cached, reranking mechanism initiated\n")

#                 chunks = retriever.invoke(request.user_query)

#                 queries_for_chunks, chunks = reranker.generate_chunk_based_queries(chunks)

#                 question_embedding, chunk_queries_embeddings = reranker.generate_embeddings(request.user_query, queries_for_chunks)

#                 scores = reranker.calculate_similarity_scores(question_embedding, chunk_queries_embeddings)

#                 reranked_chunks = reranker.rerank_chunks(scores, chunks, top_k = 3)
#                 logger.info("Retrieved Chunks Successfully reranked")

#                 if pipeline.one_liner == True:
#                     ans_format = "single_line"
#                     prompt_template = pipeline.load_prompt_template(
#                         "rag_pipeline/prompt_templates_oneline.txt",
#                     )
#                 else:
#                     ans_format = "paragraph"
#                     prompt_template = pipeline.load_prompt_template(
#                         "rag_pipeline/prompt_templates.txt",
#                     )           

#                 if stream:
#                     generator = pipeline.reranker_build_and_respond_stream(reranked_chunks, prompt_template, user_question=request.user_query)
#                     logger.info(f"Launching Reranker enhanced Stream Responder for {ans_format} response")
#                     return StreamingResponse(
#                         stream_response(generator),
#                         media_type="text/event-stream",
#                         headers = {
#                             "Cache-Control" : "no-cache",
#                             "Connection" : "keep-alive"
#                         }
#                     )
#                 else:
#                     logger.info(f"Launching Non-Stream Responder for {ans_format} response")
#                     answer = await pipeline.reranker_build_and_respond(reranked_chunks, prompt_template, user_question=request.user_query)
#                     cache.cache_response(
#                         request.user_query, 
#                         answer, 
#                         response_format=ans_format, 
#                         reranked=bool(pipeline.use_reranker)
#                     )
#                     logger.info("\nNew Query/Response successfully cached!\n")
#                     print(f"Updated Cache Status: \n")
#                     print(cache.redis.keys()) 


#         return QueryResponse(answer=answer)

#     except Exception as e:
#         traceback.print_exc()
#         return JSONResponse(content={"error" : str(e)}, status_code=500)



