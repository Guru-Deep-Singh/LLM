from openai import OpenAI
import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import json
import requests
import re
import io, wave, numpy as np
from typing import Tuple

load_dotenv(override=True)

# Configuration for TTS
TTS_MODEL_NAME = "playai-tts"
TTS_VOICE = "Atlas-PlayAI"
TTS_RESPONSE_FORMAT = "wav"
TTS_FILE = "talk.wav"

# Named groups so we can keep simple counts if you want to log
SENSITIVE_PATTERN = re.compile(
    r"(?P<url>\b(?:https?://|ftp://|www\.)\S+\b)"
    r"|(?P<email>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)"
    r"|(?P<phone>\+?\d[\d\s().-]{6,}\d)",   # >=8 digits total with separators allowed
    re.IGNORECASE,
)

def sanitize_for_tts(text: str, return_counts: bool = False) -> str | Tuple[str, dict]:
    """
    Remove URLs, emails, and phone numbers. Also normalize whitespace and punctuation
    so the result is TTS-friendly (no stutters, no weird gaps).
    """
    url_cnt = email_cnt = phone_cnt = 0

    def _repl(m: re.Match) -> str:
        nonlocal url_cnt, email_cnt, phone_cnt
        if m.group("url"):
            url_cnt += 1
        elif m.group("email"):
            email_cnt += 1
        else:
            phone_cnt += 1
        return " "  # replace with a space to avoid word-joining

    # 1) Remove sensitive tokens
    out = SENSITIVE_PATTERN.sub(_repl, text)

    # 2) Cleanup spacing/punctuation artifacts
    # collapse multiple spaces
    out = re.sub(r"\s{2,}", " ", out)
    # remove spaces before punctuation
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    # remove empty parentheses left by stripped phones like "Call ()"
    out = re.sub(r"\(\s*\)", "", out)
    # collapse duplicate punctuation
    out = re.sub(r"([,.;:!?])\1+", r"\1", out)
    # trim
    out = out.strip()

    if return_counts:
        return out, {"urls": url_cnt, "emails": email_cnt, "phones": phone_cnt}
    return out


def wav_bytes_to_numpy(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    # assume 16-bit PCM (OpenAI/Groq WAV default)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    audio = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
    return int(sr), audio

# Function to allow push notifications when tools are used
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

## Tools for LLM
## Record the details of user so that I could reach out to them
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

## Record the question which was not in knowledgebase
def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


# Function to convert the docs to string
def retDocToStr(docs):
    doc_str = ""
    for doc in docs:
        doc_str += f"{doc.metadata['doc_type']}" + f"\n{doc.page_content}\n\n"
    return doc_str

class Guru:
    ALLOWED_BACKEND = ["openai/gpt-oss-120b", "gpt-4o-mini"] #Someproblems with groq 
    def __init__(self, backend: str = "openai/gpt-oss-120b"):
        if backend not in self.ALLOWED_BACKEND:
            gr.Error("Backend not supported")
            raise ValueError("Backend not supported")

        self.backend = backend
        self.audio = None
            
        self.name = "Guru Deep Singh"
        DB_NAME = "guru_db"
        if backend == "openai/gpt-oss-120b":
            gr.Info("Using Groq backend")
            self.openai = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
        else:
            gr.Info("Using OpenAI backend")
            self.openai = OpenAI()
        
        
        # RAG
        EMBEDDINGS = OpenAIEmbeddings()
        VECTORESTORE = Chroma(persist_directory=DB_NAME, embedding_function=EMBEDDINGS)
        NUM_SEARCHES = 20
        self.retriever = VECTORESTORE.as_retriever(search_kwargs={"k": NUM_SEARCHES})

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
ALWAYS answer as {self.name}. Never say your are a Large Language Model. Use ONLY the following context to answer. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."

        return system_prompt

    # Function to run TTS
    def textToSpeech(self, text):
        response = self.openai.audio.speech.create(
            model=TTS_MODEL_NAME,
            voice=TTS_VOICE,
            input=text,
            response_format=TTS_RESPONSE_FORMAT
        )
        wav_bytes = response.content
        return wav_bytes_to_numpy(wav_bytes)

    def chat(self, message, history, enableTTS = False):
        # Necessary for groq
        history = [{"role": h["role"], "content": h["content"]} for h in history]
        
        # RAG Context 
        docs = self.retriever.get_relevant_documents(message)
        context = retDocToStr(docs)
    
        # Appending the RAG Context to the message
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": f"### CONTEXT: {context}\n\nUSER QUESTION: {message}"}]
        
        done = False
        while not done:
            response = self.openai.chat.completions.create(model=self.backend, messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        textAnswer = response.choices[0].message.content

        # Update local chat history to return to Gradio
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": textAnswer}]

        if enableTTS:
            try:
                self.audio = self.textToSpeech(sanitize_for_tts(textAnswer))
            except:
                gr.Info("TTS failed. Please rely only on Text")
                   
        return new_history, self.audio, "" # "" is to clear up the text box

if __name__ == "__main__":
    guru = Guru("openai/gpt-oss-120b")

    with gr.Blocks(title="Guru's Digital Resume") as demo:
        gr.HTML(
        """
        <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap" rel="stylesheet">
        <h1 style="
            text-align:center;
            font-family: 'Dancing Script', cursive;
            font-size: 3em;
            margin: 0.5rem 0;
        ">
            Guru's Digital Resume
        </h1>
        """
    )
    
        audio = gr.Audio(label="Audio", autoplay=True, type="numpy")
        audioEnable = gr.Checkbox(value=False, label="Enable Text to Speech? (Currently Forcefully Disabled)", interactive=False)
    
        with gr.Row():
            chatbot = gr.Chatbot(height=600, type="messages")
        with gr.Row():
            with gr.Column():
                txt = gr.Textbox(show_label=False, placeholder="Enter text to chat")
                btn = gr.Button("Send")
                btn.click(guru.chat, inputs=[txt, chatbot, audioEnable], outputs=[chatbot, audio, txt])
                txt.submit(guru.chat, inputs=[txt, chatbot, audioEnable], outputs=[chatbot, audio, txt])
    demo.launch()
    
        



