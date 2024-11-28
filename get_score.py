from marking_guide import similarity_score
from correctness import correctness_score


def get_initial_score(question: str, 
                      answer: str, 
                      marking_guide: str, 
                      further_instructions: str,
                      max_score: int, 
                      use_marking_guide: bool, 
                      strictness: float = 1.0) -> float:
    if use_marking_guide:
        initial_score = similarity_score(question, marking_guide, answer, further_instructions, strictness)
        return round(initial_score * max_score, 2)
    
    else:
        initial_score = correctness_score(question, answer, further_instructions, strictness)
        return round(initial_score * max_score, 2)

def get_final_score(initial_score: float,
                    max_score: float,
                    grammar_score: float,
                    structure_score: float,
                    relevance_score: float,
                    grammar_weight: float,
                    structure_weight: float,
                    relevance_weight: float,
                    ) -> float:
        total_weight = grammar_weight + structure_weight + relevance_weight

        if total_weight == 0:
            return initial_score
        
        else:
            normalized_grammar_weight = grammar_weight / total_weight
            normalized_structure_weight = structure_weight / total_weight
            normalized_relevance_weight = relevance_weight / total_weight

            weighted_impact = (
            grammar_score * normalized_grammar_weight +
            structure_score * normalized_structure_weight +
            relevance_score * normalized_relevance_weight
        ) 
            
            rubric_score = weighted_impact * max
            
            final_score = round((initial_score + rubric_score)/2,2)

            return final_score
        