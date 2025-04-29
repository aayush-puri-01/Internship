from fastapi import FastAPI
from routes import rag_routes
from routes import recipe_routes

app = FastAPI(
    title="Multimodal Assistant",
    description="Includes a RAG Pipeline and Recipe Generator powered by LLMs",
    version="1.0.0"
)

#registering the routes for rag pipeline

app.include_router(rag_routes.router,
                   prefix = "/rag",
                   tags = ["RAG Pipeline"])
#the prefix would mean that the endpoint path would be /rag/query as provided in the routing path 


#registering the routes for recipe generation pipeline

app.include_router(recipe_routes.router,
                   prefix="/recipe",
                   tags=["Recipe Generator"])



