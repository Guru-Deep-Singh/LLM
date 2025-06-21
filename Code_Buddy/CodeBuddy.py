# Class for creating a LLM powered assistant
import os
from dotenv import load_dotenv
from openai import OpenAI

class CodeBuddy:
    ALLOWED_BACKENDS = {"ollama", "openai"}
    def __init__(self, backend="ollama", modelName= "llama3.2"):
        if backend not in self.ALLOWED_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Allowed options are: {', '.join(self.ALLOWED_BACKENDS)}")
           
        self.backend = backend
        self.modelName = modelName
        print(f"Code Buddy using {self.backend} backend and {modelName} model")
        self.__initialize()

    def __initialize(self):
        if self.backend == "openai":
            load_dotenv(override=True)
            apiKey = os.getenv('OPENAI_API_KEY')
            if not apiKey:
                print("No API key was found")

            else:
                self.__requestHandler = OpenAI()

        elif self.backend == "ollama":
            # Setting up Constants
            self._OLLAMA_API = "http://localhost:11434/api/chat"
            self._HEADERS = {"Content-Type": "application/json"}
            self.__requestHandler = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
            
    def __nonStreamCall(self,userPrompt, systemPrompt):
        response = self.__requestHandler.chat.completions.create(
            model = self.modelName,
            messages = [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
                ])
        return response.choices[0].message.content

    def __streamCall(self,userPrompt, systemPrompt):
        stream = self.__requestHandler.chat.completions.create(
                model = self.modelName,
                messages = [
                    {"role": "system", "content": systemPrompt},
                    {"role": "user", "content": userPrompt}
                ],
                stream=True
            )
            
        for chunk in stream:
            response = chunk.choices[0].delta
            if hasattr(response, "content"):
                yield response.content



    def run(self, userPrompt, stream = False, 
            systemPrompt ="You are a helpful and informed coding agent.\
            You are given a piece of code. You have to check if the code is correct or is incorrect.\
            You need to explain the code in beginner friendy way.\
            You are also allowed to give suggestions on improvement of code for runtime optimization.\
            Give your answer in Markdown." ):
        
        if len(systemPrompt.strip()) != 0 and len(userPrompt.strip()) != 0:
            if not stream:
                return self.__nonStreamCall(userPrompt, systemPrompt)

            else:
                return self.__streamCall(userPrompt, systemPrompt)

    def runChatbot(self, userPrompt, systemPrompt):
        messages = [{"role": "system", "content": systemPrompt}] + userPrompt
        # Always streaming
        return self.__requestHandler.chat.completions.create(
            model = self.modelName,
            messages = messages,
            stream=True
        )
                
            
    

        