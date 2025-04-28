from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

model = AutoModelForSeq2SeqLM.from_pretrained("./longt5-email-summarizer")
tokenizer = AutoTokenizer.from_pretrained("./longt5-email-summarizer")

class EmailThread(BaseModel):
    email_thread: str

@app.post("/summarize")
def summarize(data: EmailThread):
    inputs = tokenizer(data.email_thread, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_length=128)
    return {"summary": tokenizer.decode(output[0], skip_special_tokens=True)}