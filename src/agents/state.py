from typing import Annotated, Sequence, TypedDict, List, Dict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    research_context: Annotated[List[Dict], operator.add]
    user_query: str
    expanded_queries: List[str]
    workflow_logs: Annotated[List[str], operator.add]
    ignored_sources: Annotated[List[Dict], operator.add]
    language_stats: Dict[str, int]
    eval_scores: Dict[str, float]
    agent_debate: Annotated[List[str], operator.add]
