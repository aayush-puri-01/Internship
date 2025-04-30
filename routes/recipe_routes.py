from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
from pydantic import BaseModel
from recipe_system.recipe import CookingAssistant
from typing import List


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
            return JSONResponse(content={"error" : "Query is required"}, status_code=400)
        
        try:
            if stream:
                generator = cook_help._generate_recipe_stream(request)
                return StreamingResponse(
                    stream_response(generator),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            else:
                output = cook_help._generate_recipe_nonstream(request)
                print(output)
                return output
        except HTTPException as e:
            raise e

        except Exception as e:
                return JSONResponse({"error": f"Internal Processing error, {str(e)}"})
        
    except Exception as e:
        return JSONResponse(content = {"error" : str(e)}, status_code=500)