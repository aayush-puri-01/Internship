import ollama
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
import logging
from fastapi import HTTPException
import os

# Set the base URL from the environment variable
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama.base_url = ollama_host



# ollama.base_url = "http://ollama:11434"
#this is to ensure that the ollama client which uses host by default now refers to the container service named "ollama" which hosts the ollama model by itself


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestSchema(BaseModel):
    ingredients: List[str]
    prompt: str

class ResponseSchema(BaseModel):
    title: str
    steps: List[str]

class CookingAssistant:
    def __init__(self, model:str = "deepseek-r1:1.5b"):
        self.model = model

    def get_ingredients(self) -> List[str]:
        response = ollama.chat(
            model = self.model,
            messages = [
                {
                    "role": "system",
                    "content": "You are a cooking assistant. Provide concise responses without reasoning."
                },
                {
                    "role": "user",
                    "content": "Start by asking me what Ingredients i have."
                }
            ],
        )
        print(response.message.content)
        
        ingredients_list = []
        while True:
            ans = input("\nEnter a ingredient: ")
            ingredients_list.append(ans)
            q = input("\nYou have other ingredients? Y/N :")
            if q == "N" or len(ingredients_list)>5:
                break

        return ingredients_list

    def suggest_dishes(self, ingredients_list: List[str]) -> None:
        ing_str = ""
        for item in ingredients_list:
            ing_str = ing_str + ", " + item

        response = ollama.chat(
            model = self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a cooking assistant. Provide Concise responses."
                },
                {
                    "role": "user",
                    "content": f"I have these ingredients: {ing_str}, what kind of dishes are possible"
                }
            ]
        )

        print(response.message.content)

    def get_user_prompt(self) -> str:
        return input("\nYour Prompt: What dish do you wish to make? : ")
    
    def generate_recipe(self, ingredients_list: List[str], user_prompt: str, stream: bool = False) -> Any:
        prompt_dict = {
            "ingredients" : ingredients_list,
            "prompt": user_prompt
        }

        try:
            user_input = RequestSchema(**prompt_dict)
        except ValueError as e:
            print("Invalid Input: {e}")
            return 
        
        if stream:
            return self._generate_recipe_stream(user_input)
        return self._generate_recipe_nonstream(user_input)
    
    def _generate_recipe_nonstream(self, user_input: RequestSchema) -> Any:

        ing_str = ", ".join(user_input.ingredients)

        try:
            logger.info("---LLM model starting---")
            response = ollama.chat(
                model = self.model,
                messages = [
                    {
                        "role": "system",
                        "content": """You are a helpful Cooking Assistant that needs to suggest recipes based on provided ingredients. Recommend ingredients if the ones provided are not enough for the recipe.

                        The recipe must be returned strictly in the following JSON format:
                        {
                        "title": "Recipe Title",
                        "steps": [
                            "First step",
                            "Second step",
                            "Third step"
                        ]
                        }

                        Do not add anything else. Do not include explanations or apologies.
                        Only return valid JSON matching exactly this schema.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"I have these ingredients: {ing_str}. {user_input.prompt}" 
                    }
                ],
                format = "json",
                stream = False
            )
            logger.info("---Finished Generating Response---")
            content = response.get("message", {}).get("content", "")
            logger.info(f"Raw Response: {content}")
            if not content:
                logger.error("No content in LLM response")
                raise ValueError("Missing content in LLM Response")
            recipe = ResponseSchema.model_validate_json(content)
            logger.info(f"Validated Recipe: {recipe}")
            return recipe   
    
        except ValidationError as e: 
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=500, detail = f"validation error : {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating the recipe: {str(e)}")
            raise HTTPException(status_code=500, detail = f"Error generating the recipe : {str(e)}")

        
    async def _generate_recipe_stream(self, user_input: RequestSchema) -> Any:
        ing_str = ", ".join(user_input.ingredients)

        try: 
            logger.info(f"---Ollama Model Starting----")
            stream = ollama.chat(
                model = self.model,
                messages = [
                    {
                        "role": "system",
                        "content": """You are a helpful Cooking Assistant that needs to suggest recipes based on provided ingredients. Recommend ingredients if the ones provided are not enough for the recipe. Do not provide explanations or apologies.

                        Return the recipe in this format:
                        Title: Recipe Title
                        Steps: 
                        -First step 
                        -Second step
                        -Third step
                        """
                    },
                    {
                        "role": "user",
                        "content": f"I have these ingredients: {ing_str}. {user_input.prompt}"
                    }
                ],
                # format = ResponseSchema.model_json_schema(),
                # since there is no point in validating a streaming response
                stream = True
            )

            collected_chunks = ""

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")

                # if content:
                #     collected_chunks += content
                #     logger.info(f"Recipe Chunk: {collected_chunks}")
                #     yield collected_chunks
                #     collected_chunks = ""

                # #this piece of code can indeed be improved, 
                # #error can be raised if no content exists
                # if chunk.get("done", False):
                #     logger.info("---Streaming---completed---")
                #     if collected_chunks.strip():
                #         yield collected_chunks
                #     break
                yield content

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error streaming recipe: {str(e)}")

        
    def run(self, stream: bool = False) -> Any:
        ingredients = self.get_ingredients()
        self.suggest_dishes(ingredients_list=ingredients)
        prompt = self.get_user_prompt()
        return self.generate_recipe(ingredients, prompt, stream=stream)


if __name__ == "__main__":
    assistant = CookingAssistant()
    final_response = assistant.run(stream=True)
    print(final_response)




