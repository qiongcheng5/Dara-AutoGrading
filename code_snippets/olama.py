# This is code to use deepseek by running deepseek locally in my system using ollama gave very poor results.

import requests

# Ollama server URL
OLLAMA_URL = "http://localhost:11434/api/generate"  # Change port if needed

# Function to query Ollama
def query_ollama(model, prompt, stream=False):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("response", "No response from model.")
    else:
        return f"Error {response.status_code}: {response.text}"


# Example usage
if __name__ == "__main__":
    model_name = "deepseek-r1"  # Ensure you have downloaded the model
    prompt_text = "Why is the sky blue?"

    result = query_ollama(model_name, prompt_text)
    print("Response from Ollama:\n", result)
