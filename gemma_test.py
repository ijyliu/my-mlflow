import mlflow
from openai import OpenAI
import os

# Set MLflow tracking URI (optional, defaults to ./mlruns)
# If you have a remote MLflow tracking server, set it here.
mlflow.set_tracking_uri("http://localhost:5000")

# For basic authentication (if needed)
os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow_user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

# Set an experiment name
mlflow.set_experiment("Ollama_Gemma_Experiments")

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Load gemini api key from credentials/gemini-api-key.txt
gemini_api_key_path = "credentials/gemini-api-key.txt"
with open(gemini_api_key_path, "r") as f:
    gemini_api_key = f.read().strip()

# Configure the OpenAI client to point to your Ollama endpoint
client = OpenAI(
    base_url="http://localhost:3000/v1",  # Or your custom port, e.g., "http://localhost:3000/v1"
    api_key='test',  # Use the loaded Gemini API key
)

# Now make your LLM calls using the OpenAI client
try:
    print("Making a completion request...")
    response = client.completions.create(
        model="gemma3-1b",  # Or "gemma:latest", "gemma:2b", etc. depending on your pulled model
        prompt="What is the capital of France?",
        max_tokens=50
    )
    print("Completion Response:", response.choices[0].text)

    print("\nMaking a chat completion request...")
    chat_response = client.chat.completions.create(
        model="gemma3-1b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Tell me a fun fact about cats."}
        ],
        max_tokens=50
    )
    print("Chat Response:", chat_response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")

print("\nMLflow runs should now be visible in the MLflow UI.")
