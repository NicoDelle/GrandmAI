import whisper
import torch

def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("medium")
    result = model.transcribe(file_path, language="it")
    
    del model
    torch.cuda.empty_cache()
    return result["text"]
