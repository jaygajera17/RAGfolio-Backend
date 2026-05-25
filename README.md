# Run

1. Create a virtual environment and activate it:
  - ```python -m venv .venv```
  -  for linux ```source .venv/bin/activate```
   -  for Windows: ```.venv\Scripts\activate```

2. Install the required dependencies:
   pip install -r requirements.txt

3.    ```uvicorn app.main:app --reload``` or 
     ```fastapi dev app/main.py --reload```