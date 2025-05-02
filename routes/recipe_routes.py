from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
from pydantic import BaseModel
from recipe_system.recipe import CookingAssistant
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()

class InputData(BaseModel):
    ingredients : List[str]
    prompt : str

"""
Remember that the input query will always be in the form of json and it will be structured as :

/application/json

{
    "ingredients" : ["<ing1>", "<ing2>", "<ing3>", ...],
    "dish" : "<str>"
}

"""

class ResponseSchema(BaseModel):
    title: str
    steps: List[str]

async def stream_response(generator):
    async for recipe in generator:
        yield f"data: {json.dumps({'chunk': recipe})}\n\n"
        # yield recipe

cook_help = CookingAssistant(model="deepseek-r1:1.5b")

@router.post("/query", response_model=ResponseSchema)
async def query_recipe_generator(request: InputData, stream:bool = Query(False)):
    try:
        if not request:
            logger.error(f"Error: No request found")
            return JSONResponse(content={"error" : "Query is required"}, status_code=400)
        
        try:
            if stream:
                logger.info("Processing Streaming Request \n")
                generator = cook_help._generate_recipe_stream(request)
                return StreamingResponse(
                    stream_response(generator),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            # StreamingResponse() generates a Server-Side Event (SSE) which is a HTTP message with headers and content
            else:
                logger.info("Processing Non streaming request")
                output = cook_help._generate_recipe_nonstream(request)
                logger.info(f"SSE: {output}")
                return output
        except HTTPException as e:
            logger.error(f"HTTP Exception {str(e)}")
            raise e

        except Exception as e:
                logger.error(f"Internal Processing Error, api endpoint entered")
                return JSONResponse({"error": f"Internal Processing error, {str(e)}"})
        
    except Exception as e:
        logger.info("Internal Server Error, api endpoint not entered")
        return JSONResponse(content = {"error" : str(e)}, status_code=500)