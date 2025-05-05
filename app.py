import gradio as gr
import requests
import time
import re
import json

# API_URL = "http://localhost:8000/recipe/query"
API_URL = "http://fastapi:8000/recipe/query"
#this api url can be made dynamic, ==> if environment variable exists, then place "fastapi" in the hostname part of the url and if env variable doesnt exist, simply put "localhost" in the hostname part

def generate_recipe(chatbot, ingredients, dish, stream=False):

    ingredients_list = [i.strip() for i in ingredients.split(",")]
    payload = {
        "ingredients" : ingredients_list,
        "prompt": dish
    }
    user_message = f"I have these ingredients: {', '.join(ingredients_list)}.\n {payload['prompt']}"

    chatbot = chatbot + [(user_message, "")]
    yield chatbot

    try:

        if stream:
            assistant_reply = ""


            with requests.post(API_URL, json=payload, params={"stream":True}, stream=True) as r:
                r.raise_for_status()

                for chunk in r.iter_content(chunk_size = None, decode_unicode=True):
                    try:
                        assistant_reply += chunk
                        updated_chatbot = list(chatbot)
                        updated_chatbot[-1] = (user_message, assistant_reply)
                        yield updated_chatbot
                            # time.sleep(0.03)
                    except json.JSONDecodeError:
                        assistant_reply += chunk
                        updated_chatbot = list(chatbot)
                        updated_chatbot[-1] = (user_message, assistant_reply)
                        yield updated_chatbot
                                    # time.sleep(0.03)
        else:
            response = requests.post(API_URL, json=payload, params={"stream":False})
            response.raise_for_status() #Raises HTTP error if one has occurred
            data = response.json()

            assistant_reply = f"**Recipe: {data['title']}**\n\n"
            assistant_reply += "\n".join(f"{i+1}.{step}" for i, step in enumerate(data['steps']))

            chatbot[-1] = (user_message, assistant_reply)
            yield chatbot

    except Exception as e:
        error_message =  f"Error: {str(e)}"
        chatbot.append((user_message, error_message))
        yield chatbot
    
def user_input_processing(user_message, chat_history, ingredients, dish, stream_toggle):
    return generate_recipe(user_message, chat_history, ingredients, dish, stream_toggle)


def get_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Cooking Assistant (streaming chatbot)")

        chatbot = gr.Chatbot(
            label = "Recipe Assistant",
            height = 500,
            type="tuples",
            render_markdown=True
        )

        with gr.Row():
                
            ingredients_input = gr.Textbox(label="Ingredients (comma-separated)", placeholder="eg: tomato, basil, milk")
            dish_input = gr.Textbox(label="Dish Prompt", placeholder="eg: I wish to make a pizza")

        with gr.Row():

            stream_toggle = gr.Checkbox(label="Streaming Output?")
            generate_btn = gr.Button("Generate Recipe", variant = "primary")


        generate_btn.click(
            generate_recipe,
            inputs=[chatbot, ingredients_input, dish_input, stream_toggle],
            outputs=chatbot,
        )

        clear_btn = gr.Button("Clear Chat")
        clear_btn.click(lambda: [], None, chatbot, queue = False)

        return demo

final_interface = get_interface()
final_interface.launch(server_name="0.0.0.0", server_port=7860)

