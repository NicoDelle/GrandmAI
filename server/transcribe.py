import whisper

def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("medium")
    result = model.transcribe(file_path, language="it")
    
    return result["text"]
