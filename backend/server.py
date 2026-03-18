from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_system import ask_rag_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "MedIQ API running"}


@app.post("/ask")
def ask_question(q: Question):
    return StreamingResponse(
        ask_rag_stream(q.question),
        media_type="text/plain"
    )
