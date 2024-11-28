import streamlit as st
from groq import Groq
import os

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

def score_grammar(answer: str, strictness: float) -> float:
    system_message = (f"Evaluate the grammar, spelling, and command of English of the answer: '{answer}'. Provide a score between 0 and 1 based on strictness factor of {strictness} over 1."
                      "The strictness factor tells you how lenient you should be in assessing the student's grammar and spelling and assigning a score."
                      "If strictness factor is less than 0.4 over 1, return maximum mark (1.0) if the student's answer is at least 80% grammatically correct even if there are a few spelling errors."
                      "If strictness factor is higher i.e. 0.75, be less lenient and critically assess the grammar... the higher the strictness factor, the less lenient you should be."
                      "Return only the score, no text or feedback at all, just the score ONLY.")
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)