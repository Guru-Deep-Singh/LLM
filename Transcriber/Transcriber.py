import os
from dotenv import load_dotenv
from openai import OpenAI

class Transcriber:
    ALLOWED_BACKENDS = {"openai", "whisper"}
    def __init__(self, backend = "openai", modelName="whisper-1"): #"gpt-4o-mini-transcribe" can be used but more expensive
        if backend not in self.ALLOWED_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Allowed options are: {', '.join(self.ALLOWED_BACKENDS)}")
        self.modelName = modelName #with whisper backend {"base", "small", "medium", "large"}
        self.backend = backend

        print(f"Transcribe using {self.backend} backend with {self.modelName} model.")

        self.__initialize()

    def __initialize(self):
        if self.backend == "openai":
            load_dotenv(override=True)
            apiKey = os.getenv('OPENAI_API_KEY')
            if not apiKey:
                print("No API key was found")

            else:
                self.__requestHandler = OpenAI()

        elif self.backend == "whisper":
            import whisper
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__requestHandler = whisper.load_model(self.modelName, device=device)
            
        else:
            raise ValueError("Backend not supported")

    def run(self,audioFilePath):
        if not audioFilePath:
            return ""  # Gracefully exit if audio input was cleared
        if self.backend == "openai":
            with open(audioFilePath, "rb") as f:
                transcript = self.__requestHandler.audio.transcriptions.create(
                    file=f,
                    model=self.modelName
                )
                return transcript.text
        elif self.backend == "whisper":
            transcript = self.__requestHandler.transcribe(audioFilePath)
            return transcript["text"]