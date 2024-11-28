import streamlit as st
from groq import Groq
import os

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

def correctness_score(question: str, answer: str, further_instructions: str, strictness: float) -> float:
    system_message = (
        f"Evaluate the answer '{answer}' given to the question '{question}'. "
        "Provide a correctness score between 0.0 (minimum score) and 1.0 (maximum score) based on what you know about the topic."
        "Do not penalize the student's answer if they don't provide extensive information about the topic, focus on the correctness of the answer provided and if it answers the question or not."
        f"VERY IMPORTANTLY: Apply a strictness factor of {strictness} over 1 to your evaluation. The strictness factor tells you how lenient you should be in assessing the student answer and assigning a score. "
        "If strictness factor is less than 0.4 over 1, return maximum mark (1.0) as long as the student answer shows the student has an idea of what the correct answer is even if the idea isn't complete, return maximum mark of 1.0."
        "If the strictness factor is greater than 0.4 over 1, the higher the strictness factor the less lenient you should be in assessing the student's answer in relation to the marking guide and or in correctness in general (the student's idea) and assign a score."
        "If strictness factor is high i.e. 0.8 over 1.0, then shabby answers that do not relate to the marking guide or have enough information to answer the question should not be awarded maximum mark but still award some marks for general ideas even if not complete."
        "Do not penalize grammatical errors or spelling mistakes, focus on the meaning of the answer."
        "Do not penalize grammatical errors or spelling mistakes, focus on the correctness of the answer."
        f"MOST IMPORTANT: Consider if these further instructions: {further_instructions} are met. If no further instruction is provided, ignore else penalize answers that don't satisfy the further instructions when assigning score:"
        "Return only the score, no text or feedback at all, just the score ONLY."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)
