import streamlit as st
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

def score_structure(answer: str, strictness: float) -> float:
    system_message = (f"Evaluate the structure and organization of the answer: '{answer}'. Provide a score between 0 and 1 based on strictness factor of {strictness} over 1."
                      "The strictness factor tells you how lenient you should be in assessing the student's organization of points and assigning a score."
                      "If strictness factor is less than 0.7 over 1, always return maximum mark (1.0) if the student's answer is readable and understanble at least even if it's not the most organized and well put."
                      "Return only the score, no text or feedback at all, just the score ONLY.")
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)