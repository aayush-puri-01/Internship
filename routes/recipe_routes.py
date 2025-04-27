from fastapi import APIRouter
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

class Response(BaseModel):
    answer : str

cook_help = CookingAssistant(model="deepseek-r1:1.5b")

@router.post("/query", response_model=Response)
async def query_recipe_generator(request: InputData):
    try:
        if not request:
            return JSONResponse(content={"error" : "Query is required"}, status_code=400)
        
        try:
            output = cook_help._generate_recipe_stream(request)
            return StreamingResponse(output, media_type="text/plain")

        except Exception as e:
            return json.dumps({"error": f"Internal Processing error, {str(e)}"})
    
    except Exception as e:
        return JSONResponse(content = {"error" : str(e)}, status_code=500)