import streamlit as st
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

def similarity_feedback(question: str, 
                        marking_guide: str, 
                        answer: str, 
                        further_instruction: str, 
                        strictness: str, 
                        initial_score: float, 
                        max_score: float) -> str:
    
    system_message = (
        f"A student provided this answer: '{answer}' to the question '{question}'"
        f"You scored the student {initial_score} over {max_score} based on the semantic similarity, relatedness, and closness of their answer to the marking guide '{marking_guide}'"
        f"You assigned this score with a strictness factor of {strictness} over 1. A strictness factor tells you how lenient you should be in grading, and how much you should reward 'having an idea even if it's incomplete' over perfection or perfect match with marking guide."
        f"If not none, You may have followed some extra instructions: {further_instruction} to guide your grading."
        "Now, provide a constructive personalized feedback to the student"
        "Keep your feedback short and encouraging. Only highlight the strengths and weaknessses of student's answer, don't provide the marking guide. Justify the score you assigned."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    feedback = chat_response.choices[0].message.content
    return f"**Correct answer:** {marking_guide}\n\n" +  f"**Your answer:** {answer}\n\n" + feedback



def correctness_feedback(question: str,
                        answer: str,
                        further_instruction: str,
                        strictness: float,
                        initial_score: float, 
                        max_score: float) -> str:
    
    system_message = (
        f"A student provided this answer: '{answer}' to the question '{question}'"
        f"You scored the student {initial_score} over {max_score} based on the general correctness of the answer, not penalizing lack of extensive information."
        f"You assigned this score with a strictness factor of {strictness} over 1. A strictness factor tells you how lenient you should be in grading, and how much you should reward 'having an idea even if it's incomplete' over perfection or extensive knowledge."
        f"If not none, You may have followed some extra instructions: {further_instruction} to guide your grading."
        "Now, provide a constructive personalized feedback to the student"
        "Keep your feedback short and encouraging. Only highlight the strengths and weaknessses of student's answer. Justify the score you assigned."
    )
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    feedback = chat_response.choices[0].message.content
    return f"Score: {initial_score}/{max_score}\n\n" +  f"Your answer: {answer}\n\n" + feedback

def general_feedback(question: str,
                     answer: str,
                     marking_guide: str,
                     initial_score: float,
                     final_score: float,
                     max_score: float,
                     grammar_score: float,
                     structure_score: float,
                     relevance_score: float,
                     grammar_weight: float,
                     structure_weight: float,
                     relevance_weight: float,
                     strictness: float,
                     further_instruction: str) -> str:

    
    system_message = (
        f"A student answered the question '{question}' with this response: '{answer}'."
        f"Using the marking guide '{marking_guide}', their initial score was {initial_score} out of the {max_score}."
        f"The score was adjusted based on the following components:\n"
        f"- Grammar: {grammar_score} (weight: {grammar_weight})\n"
        f"- Structure: {structure_score} (weight: {structure_weight})\n"
        f"- Relevance: {relevance_score} (weight: {relevance_weight})\n"
        "A weight of 0 means the component was not considered in the final score."
        f"These adjustments were applied with a strictness factor of {strictness}."
        f"If provided, additional instructions were followed: {further_instruction}."
        f"The final score after adjustments is {final_score} out of the {max_score}."
        "Now, provide a general, constructive, and personalized feedback summarizing how these components impacted the score."
        "Highlight the strengths and weaknesses in the student's answer and encourage improvement in specific areas. Justify the score you assigned."
    )
    
    messages = [{"role": "system", "content": system_message}]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0
    )
    
    feedback = chat_response.choices[0].message.content
    rubrics_considered = []

    # Check if each rubric weight is greater than zero and add it to the message
    if grammar_weight > 0:
        rubrics_considered.append(f"**Grammar score:** {grammar_score} (Weight: {grammar_weight})\n")
    if structure_weight > 0:
        rubrics_considered.append(f"**Structure score:** {structure_score} (Weight: {structure_weight})\n")
    if relevance_weight > 0:
        rubrics_considered.append(f"**Relevance score:** {relevance_score} (Weight: {relevance_weight})\n")

    # If no rubric was considered, indicate that only the initial score was used
    if not rubrics_considered:
        rubrics_considered = ["Only the initial score was considered."]
    else:
        rubrics_considered = [f"{', '.join(rubrics_considered)}"]

    # Combine all the rubrics message into one string
    rubrics_message = " ".join(rubrics_considered)

    return (
        f"**Initial Score:** {initial_score}/{max_score}\n\n"
        f"**Final Score:** {final_score}/{max_score}\n\n"
        f"**Rubrics Considered:**\n {rubrics_message}\n\n"
        f"**Your Answer:** {answer}\n\n"
        f"**Constructive Feedback:**\n{feedback}"
    )
