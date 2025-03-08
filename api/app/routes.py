from fastapi import APIRouter, Depends, HTTPException
from .models import TextGenerationRequest, TextGenerationResponse
from .auth import get_current_user

router = APIRouter()

# Dummy function to simulate model inference
def generate_text(prompt: str) -> str:
    # Replace this with actual model inference logic
    return f"Generated response for: {prompt}"

@router.post("/generate", response_model=TextGenerationResponse)
def generate(request: TextGenerationRequest, current_user: str = Depends(get_current_user)):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    response_text = generate_text(request.prompt)
    return TextGenerationResponse(generated_text=response_text)
