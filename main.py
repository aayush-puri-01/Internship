import ollama
from pydantic import BaseModel
from typing import List, Dict, Any

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
        response = ollama.chat(
        model = self.model,
        messages = [
            {
                "role": "user",
                "content": f"I have these ingredients: {user_input.ingredients}. {user_input.prompt}"
            }
        ],
        format = ResponseSchema.model_json_schema(),
        stream = False
        )
        
        print("--------Recipe--------\n")

        try:
            recipe = ResponseSchema.model_validate_json(response.message.content)
            print(recipe.model_dump_json())
            return recipe.model_dump_json()
        except ValueError:
            return {"Error": "Failed to validate response"}
        
    def _generate_recipe_stream(self, user_input: RequestSchema) -> Any:
        stream = ollama.chat(
        model = self.model,
        messages = [
            {
                "role": "user",
                "content": f"I have these ingredients: {user_input.ingredients}. {user_input.prompt}"
            }
        ],
        format = ResponseSchema.model_json_schema(),
        stream = True
        )
        
        print("--------Recipe--------\n")

        collected_chunks = ""

        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            collected_chunks += content
            print(content, end="", flush=True)

        try:
            recipe = ResponseSchema.model_validate_json(collected_chunks)
            print("\n")
            print(recipe.model_dump_json())
            return recipe.model_dump_json()
        except ValueError:
            return {"Error": "Failed to validate response"}
        
    def run(self, stream: bool = False) -> Any:
        ingredients = self.get_ingredients()
        self.suggest_dishes(ingredients_list=ingredients)
        prompt = self.get_user_prompt()
        return self.generate_recipe(ingredients, prompt, stream=stream)


if __name__ == "__main__":
    assistant = CookingAssistant()
    final_response = assistant.run(stream=True)
    print(final_response)




