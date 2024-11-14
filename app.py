import streamlit as st
from groq import Groq
import os

# Streamlit secrets
GROQ_API_KEY = st.secrets["keys"]["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

def get_similarity_score_marking_guide(question: str, marking_guide: str, answer: str, strictness: float) -> float:

    system_message = (
        f"As an experienced professor, you asked the question: {question}. Compare the student answer: '{answer}' to the marking guide: '{marking_guide}'. "
        "Provide a single semantic similarity score between 0 and 1 based on correctness but focusing mainly on how well the answer matches the marking guide."
        f" Apply a strictness factor of {strictness} over 1 to your evaluation. Return only the score, no text or feedback at all, just the score ONLY."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=1.0
    )
    return float(chat_response.choices[0].message.content)

def get_correctness_score_no_marking_guide(question: str, answer: str, strictness: float) -> float:
    system_message = (
        f"As a professor, evaluate the answer '{answer}' to the question '{question}'. "
        "Provide a correctness score between 0 and 1 based on your pre-trained knowledge, considering how well it addresses the question. Your focus is on effort not perfection."
        f" Apply a strictness factor of {strictness} over 1 to your evaluation. Return only the score, no text or feedback at all, just the score ONLY."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)

def score_grammar(answer: str, strictness: float) -> float:
    system_message = (f"Evaluate the grammar, spelling, and command of English of the answer: '{answer}'. Provide a score between 0 and 1 based on strictness of {strictness} over 1."
                        "Return only the score, no text or feedback at all, just the score ONLY.")
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)

def score_structure(answer: str, strictness: float) -> float:
    # Structure scoring function with strictness
    system_message = (f"Evaluate the structure and organization of the answer: '{answer}'. Provide a score between 0 and 1 based on strictness of {strictness} over 1."
                      "Return only the score, no text or feedback at all, just the score ONLY.")
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)

def score_relevance(question: str, answer: str, strictness: float) -> float:
    # Relevance scoring function with strictness
    system_message = (
        f"Evaluate the relevance of the answer '{answer}' to the question '{question}'. "
        f"Provide a score between 0 and 1 based on how well it addresses the key points and considering a strictness factor of {strictness} over 1."
        "Return only the score, no text or feedback at all, just the score ONLY."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    return float(chat_response.choices[0].message.content)

def get_score(question: str, answer: str, marking_guide: str, max_score: int, use_marking_guide: bool,
              grammar_weight: float = 0.0, structure_weight: float = 0.0, relevance_weight: float = 0.0, strictness: float = 1.0) -> float:
    
    if use_marking_guide:
        # If using marking guide, calculate a similarity score only
        similarity_score = get_similarity_score_marking_guide(question, marking_guide, answer, strictness)
        return round(similarity_score * max_score, 2)
    
    else:
        # No marking guide: combine correctness with grammar, structure, and relevance scores
        correctness_score = get_correctness_score_no_marking_guide(question, answer, strictness)
        grammar_score = score_grammar(answer, strictness) * grammar_weight
        structure_score = score_structure(answer, strictness) * structure_weight
        relevance_score = score_relevance(question, answer, strictness) * relevance_weight
        
        # Adjust correctness score based on criteria weights
        adjusted_score = correctness_score * (1 - (grammar_weight + structure_weight + relevance_weight)) \
                         + grammar_score + structure_score + relevance_score
                         
        return round(adjusted_score * max_score, 2)


# Feedback functions

def generate_feedback_marking_guide(question: str, marking_guide: str, answer: str, score: float) -> str:
    
    system_message = (
        f"As a professor, you awarded the score of {score} to a student who answered the question: {question}. Compare the student answer '{answer}' with the marking guide '{marking_guide}'. "
        "Provide a structured breakdown including (1) The correct answer/ marking guide, (2) Strengths of the student answer/ how well the student's answer met the standard of the marking guide, "
        "(3) Parts of the correct answer the student missed. All as a metric/ explanation for the score you gave."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    feedback = chat_response.choices[0].message.content
    return f"Score: {score}\n\n" + feedback



def generate_feedback_no_marking_guide(question: str, answer: str, grammar_score: float, structure_score: float,
                                       relevance_score: float, score: float) -> str:
    system_message = (
        f"As a professor, you awarded the score of {score} to a student who provided this answer '{answer}' to the question '{question}'."
        "Provide feedback to the student to let them understand why you gave them that score and let them be aware of the strengths and weaknesses of their answer."
        "You may have considered one of the following (if the score is 0, it may be possible you didn't consider it at all so don't provide feedback on it if it is zero), so provide feedback on:\n"
        f"(1) Grammar Score: {grammar_score} - highlight areas of strength and errors, "
        f"(2) Structure Score: {structure_score} - outline effective structure or organizational issues, "
        f"(3) Relevance Score: {relevance_score} - indicate if the answer addressed key points and relevance to the question. "
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    feedback = chat_response.choices[0].message.content
    return f"Score: {score}\n\n" + feedback

# Streamlit UI
st.title("AI-Powered Grading App")

# Input fields for question, answer, and marking guide
question = st.text_area("Enter the Question", "")
answer = st.text_area("Enter the Student's Answer", "")

# Sidebar grading settings
st.sidebar.header("Grading Settings")

# Dropdown for choosing grading criteria
grading_option = st.sidebar.selectbox(
    "Select Grading Method",
    ["Use Marking Guide", "Don't Use Marking Guide"]
)

if grading_option == "Use Marking Guide":
    use_marking_guide = True
    marking_guide = st.sidebar.text_area("Enter the Marking Guide: ", "")
else:
    use_marking_guide = False
    marking_guide = ""

# Custom scoring settings
max_score = st.sidebar.number_input("Maximum Score for this Question", min_value=1, max_value=100, value=10)

if not use_marking_guide:
    grammar_weight = st.sidebar.slider("Grammar Weight", 0.0, 1.0, 0.2)
    structure_weight = st.sidebar.slider("Structure Weight", 0.0, 1.0, 0.2)
    relevance_weight = st.sidebar.slider("Relevance Weight", 0.0, 1.0, 0.3)

strictness = st.sidebar.slider("Strictness", 0.0, 1.0, 0.5)

# Submit button for grading
if st.button("Grade Answer"):
    if use_marking_guide:
        score = get_score(question, answer, marking_guide, max_score, use_marking_guide, strictness)
    else:
        score = get_score(question, answer, marking_guide, max_score, use_marking_guide, grammar_weight, structure_weight, relevance_weight, strictness)
    
    # Show feedback based on the grading method
    if use_marking_guide:
        feedback = generate_feedback_marking_guide(question, marking_guide, answer, score)
    else:
        grammar_score = score_grammar(answer, strictness) * grammar_weight
        structure_score = score_structure(answer, strictness) * structure_weight
        relevance_score = score_relevance(question, answer, strictness) * relevance_weight
        feedback = generate_feedback_no_marking_guide(question, answer, grammar_score, structure_score, relevance_score, score)
    
    st.write(f"### Final Score: {score}")
    st.markdown(f"### Feedback: \n\n{feedback}", unsafe_allow_html=True)
