from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, T5ForConditionalGeneration,
    GPTNeoForCausalLM, ElectraModel
)
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lyon's ML Models API", version="1.0.0")

# Model configurations
MODELS = {
    'tinny-llama': 'Lyon28/Tinny-Llama',
    'pythia': 'Lyon28/Pythia', 
    'bert-tinny': 'Lyon28/Bert-Tinny',
    'albert-base': 'Lyon28/Albert-Base-V2',
    't5-small': 'Lyon28/T5-Small',
    'gpt2': 'Lyon28/GPT-2',
    'gpt-neo': 'Lyon28/GPT-Neo',
    'distilbert': 'Lyon28/Distilbert-Base-Uncased',
    'distil-gpt2': 'Lyon28/Distil_GPT-2',
    'gpt2-tinny': 'Lyon28/GPT-2-Tinny',
    'electra-small': 'Lyon28/Electra-Small'
}

# Cache untuk models yang sudah diload
model_cache = {}
tokenizer_cache = {}

# Environment settings
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

class InferenceRequest(BaseModel):
    model_name: str
    text: str
    max_length: int = 100
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    model_name: str
    input_text: str
    output_text: str
    processing_time: float

def load_model_with_retry(model_id: str, max_retries: int = 3):
    """Load model dengan retry mechanism"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model {model_id}, attempt {attempt + 1}")
            
            # Determine model type and load accordingly
            if 'gpt' in model_id.lower() or 'llama' in model_id.lower() or 'pythia' in model_id.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif 't5' in model_id.lower():
                model = T5ForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            elif 'neo' in model_id.lower():
                model = GPTNeoForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            elif 'electra' in model_id.lower():
                model = ElectraModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Default untuk BERT-like models
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Successfully loaded {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {model_id}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to load model after {max_retries} attempts: {str(e)}")

def get_model_and_tokenizer(model_name: str):
    """Get atau load model dan tokenizer"""
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
    
    model_id = MODELS[model_name]
    
    # Check cache
    if model_name in model_cache:
        return model_cache[model_name], tokenizer_cache[model_name]
    
    # Load model
    model, tokenizer = load_model_with_retry(model_id)
    
    # Cache model
    model_cache[model_name] = model
    tokenizer_cache[model_name] = tokenizer
    
    return model, tokenizer

@app.get("/")
async def root():
    return {
        "message": "Lyon's ML Models API", 
        "available_models": list(MODELS.keys()),
        "endpoints": ["/models", "/inference", "/health"]
    }

@app.get("/models")
async def list_models():
    return {"available_models": MODELS}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "loaded_models": list(model_cache.keys())}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    start_time = time.time()
    
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        
        # Tokenize input
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Generate output based on model type
        with torch.no_grad():
            if 'gpt' in request.model_name or 'llama' in request.model_name or 'pythia' in request.model_name:
                # Generative models
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=request.max_length,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            elif 't5' in request.model_name:
                # T5 model
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=request.max_length,
                    temperature=request.temperature,
                    do_sample=True
                )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # Encoder models (BERT-like) - return embeddings info
                outputs = model(**inputs)
                output_text = f"Embedding shape: {outputs.last_hidden_state.shape}, Input processed successfully"
        
        processing_time = time.time() - start_time
        
        return InferenceResponse(
            model_name=request.model_name,
            input_text=request.text,
            output_text=output_text,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/preload/{model_name}")
async def preload_model(model_name: str):
    """Preload specific model untuk warm-up"""
    try:
        model, tokenizer = get_model_and_tokenizer(model_name)
        return {"message": f"Model {model_name} preloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
