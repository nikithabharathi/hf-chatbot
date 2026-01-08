import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Check your .env file.")

# Hugging Face OpenAI-compatible client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",  # ✅ FREE SERVERLESS MODEL
            messages=[
                {"role": "system", "content": "You are a helpful study assistant."},
                {"role": "user", "content": user_message},
            ],
        )

        msg = response.choices[0].message
        reply = msg.get("content") if isinstance(msg, dict) else msg.content

        if isinstance(reply, list):
            reply = reply[0].get("text", "")

        if not reply:
            reply = "⚠️ Model returned empty response."

    except Exception as e:
        reply = f"Error: {str(e)}"

    return {"reply": reply}
