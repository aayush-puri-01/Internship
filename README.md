# Cooking Assistant

A simple Python application that interacts with the Ollama API to generate recipes based on user-provided ingredients and prompts. Supports both synchronous and streaming modes for recipe generation.

## Features

- Collects ingredients from user input.
- Suggests possible dishes based on ingredients.
- Generates detailed recipes using the Ollama API.
- Supports streaming mode to display recipe output in real-time.
- Uses Pydantic for structured input/output validation.

## Requirements

- Python 3.8+
- `ollama` library (`pip install ollama`)
- `pydantic` library (`pip install pydantic`)
- Running Ollama server with the `deepseek-r1:1.5b` model

## Usage

1. Clone the repository or save the `cooking_assistant.py` file.

2. Install dependencies:

   ```bash
   pip install ollama pydantic
   ```

3. Run the script:

   ```bash
   python cooking_assistant.py
   ```

4. Follow the prompts to enter ingredients and select a dish.

5. To enable streaming mode, modify the main execution:

   ```python
   final_response = assistant.run(stream=True)
   ```

## Example

```bash
$ python cooking_assistant.py
What ingredients do you have?
Enter a ingredient: Potato
You have other ingredients? Y/N: Y
Enter a ingredient: Zucchini
You have other ingredients? Y/N: N
Possible dishes: Vegetable stir-fry, roasted vegetables
Your Prompt: What dish do you wish to make: Vegetable stir-fry
--------Recipe--------
{"title": "Vegetable Stir-Fry", "steps": ["Chop potatoes into cubes", "Slice zucchini", "Stir-fry with oil and spices"]}
```

## Notes

- Ensure the Ollama server is running with the correct model.
- Streaming mode displays recipe JSON chunks as they arrive.
- Invalid inputs or API errors are handled with clear error messages.