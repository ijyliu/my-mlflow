import mlflow
from packaging.version import Version
import os

# Connect to mlflow on localhost port 5000
mlflow.set_tracking_uri("http://localhost:5000")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password1234'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['MYSQL_DATABASE'] = 'mlflow_database'
os.environ['MYSQL_USER'] = 'mlflow_user'
os.environ['MYSQL_PASSWORD'] = 'mlflow'
os.environ['MYSQL_ROOT_PASSWORD'] = 'mysql'

assert Version(mlflow.__version__) >= Version("2.18.0"), (
  "This feature requires MLflow version 2.18.0 or newer. "
  "Please run '%pip install -U mlflow' in a notebook cell, "
  "and restart the kernel when the command finishes."
)

import google.genai as genai

mlflow.gemini.autolog()

# Replace "GEMINI_API_KEY" with your API key
# Load from credentials/gemini-api-key.txt
with open("credentials/gemini-api-key.txt", "r") as f:
    api_key = f.read().strip()
client = genai.Client(api_key=api_key)

# Inputs and outputs of the API request will be logged in a trace
client.models.generate_content(model="gemini-1.5-flash", contents="Hello!")
