from fastapi import FastAPI
import uvicorn

# Import detection functions
from object_detection import run_object_detection
from sign_detection import run_sign_recognition

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to GuardianX Backend"}

@app.get("/detect_objects")
def detect_objects():
    run_object_detection()
    return {"status": "Object detection started"}

@app.get("/detect_signs")
def detect_signs():
    run_sign_recognition()
    return {"status": "Sign recognition started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
