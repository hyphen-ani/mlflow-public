Here are the commands i used


# For Creating Virtual Environment
python3 -m venv .venv {}
source .venv/bin/activate

uv venv {}
source .venv/bin/activate


# Installing Libraries
uv pip install -r requirements.txt

# Start MLFLow
mlflow ui --backend-store-ui sqlite:///mlflow.db port=7005

http://localhost:7005
