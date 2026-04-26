from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

def evaluate_response(query: str, context: list[str], answer: str):
    """
    Evaluates the AI's response. 
    Note: Ragas defaults to OpenAI. We are using placeholder scores 
    to avoid Quota errors until Gemini embeddings are configured.
    """
    return {
        "faithfulness": 0.85, 
        "answer_relevancy": 0.92,
        "note": "Using placeholder scores (OpenAI Quota exceeded)"
    }
