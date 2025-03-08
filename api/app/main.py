from fastapi import FastAPI
from .routes import router

app = FastAPI()

# Include the router from routes.py
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Language Model API"}
