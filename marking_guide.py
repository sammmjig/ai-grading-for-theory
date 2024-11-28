import streamlit as st
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

def similarity_score(question: str, marking_guide: str, answer: str, further_instructions: str, strictness: float) -> float:
    '''This function gets the similarity score between marking guide and student answer'''

    system_message = (
        f"You asked the question: {question}. Compare the student answer: ['{answer}'] to the marking guide: ['{marking_guide}']. "
        "Provide a single semantic similarity score between 0.0 (minimum score) and 1.0 (maximum score) based on general correctness coupled with how well the answer relates to the marking guide."
        "Do not penalize the student's answer if the answer means the same thing as or something close to the marking guide but uses different words."
        f"VERY IMPORTANTLY: Apply a strictness factor of {strictness} over 1 to your evaluation. The strictness factor tells you how lenient you should be in assessing the student answer and assigning a score. "
        "If strictness factor is less than 0.4 over 1 AND the answer shows the student has an idea of the correct asnwer even if the idea isn't complete or perfectly the same as marking guide, return maximum mark of 1.0."
        "If the strictness factor is greater than 0.4 over 1, the higher the strictness factor the less lenient you should be in assessing the student's answer in relation to the marking guide and or in correctness in general (the student's idea) and assign a score."
        "If strictness factor is high i.e. 0.8 over 1.0, then shabby answers that do not relate to the marking guide or have enough information to answer the question should not be awarded maximum mark but still award some marks for general ideas even if not complete."
        "Do not penalize grammatical errors or spelling mistakes, focus on the meaning of the answer."
        f"MOST IMPORTANT: Consider if these further instructions: {further_instructions} are met. If no further instruction is provided, ignore else override strictness factor and penalize answers that don't satisfy the further instructions when assigning score."
        "Return only the score, no text or feedback at all, just the score ONLY."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=1.0
    )
    return float(chat_response.choices[0].message.content)

