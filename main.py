from fastapi import FastAPI
from routes import rag_routes

app = FastAPI(
    title="Multimodal Assistant",
    description="Includes a RAG Pipeline and Recipe Generator powered by LLMs",
    version="1.0.0"
)

#registering the routes for rag pipeline

app.include_router(rag_routes.router,
                   prefix = "/rag",
                   tags = ["RAG Pipeline"])


