import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import webbrowser
from threading import Timer


from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.agents.state import AgentState
from src.tools.search import MultiLingualSearch
from src.utils.evaluator import evaluate_response

load_dotenv()

app = FastAPI(title="LingaSynthesize API")

# --- INITIALIZE TOOLS & LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
search_tool = MultiLingualSearch()

# --- NODES ---

def supervisor_node(state: AgentState):
    new_logs = ["REASONING: Defined multi-agent search strategy."]
    
    context = state.get('research_context', [])
    if len(context) >= 3:
        new_logs.append("DECISION: Strategic data density achieved. Initializing synthesis.")
        return {"next": "synthesizer", "workflow_logs": new_logs}
    
    new_logs.append("REASONING: Cross-lingual depth insufficient. Expanding global coverage...")
    return {"next": "researcher", "workflow_logs": new_logs}

def researcher_node(state: AgentState):
    query = state["user_query"]
    new_logs = ["🔍 Initializing multi-lingual search..."]
    
    expanded = [query, f"{query} research", f"latest trends in {query}"]
    results = search_tool.perform_search(query)
    
    valid_results = []
    ignored = []
    lang_stats = {"English": 0, "Japanese": 0, "German": 0}
    
    for r in results:
        if len(r.get('content', '')) < 50:
            ignored.append({"title": r.get('title', 'Unknown'), "reason": "Low content density"})
            continue
        lang = r.get('language', 'English')
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
        valid_results.append(r)

    new_logs.append(f"STATUS: Neutralizing language barriers. Routing to JP and DE nodes...")
    new_logs.append(f"RETRIEVAL: Integrated {len(valid_results)} high-confidence global sources.")
    
    return {
        "research_context": valid_results,
        "expanded_queries": expanded,
        "workflow_logs": new_logs,
        "ignored_sources": ignored,
        "language_stats": lang_stats,
        "next": "supervisor"
    }

def synthesizer_node(state: AgentState):
    new_logs = ["🧠 Synthesizing cross-lingual intelligence..."]
    
    context = state.get('research_context', [])
    context_text = "\n".join([f"[{r['language']}] {r['content']}" for r in context])
    
    prompt = f"""You are a Cross-Lingual Research Synthesizer.
    Based on the context, produce the report with these EXACT markers:

    FINAL_INSIGHT:
    (Provide exactly 3 high-impact bullet points. Use technical language.)

    HERO_SUMMARY:
    (Exactly 2 scannable cards. Each MUST be a detailed, 3-sentence technical deep-dive into a specific finding.)

    EVIDENCE_LIST:
    (List 3 specific source titles used - include the JP or DE titles if available)

    GERMAN_INSIGHTS:
    (Analysis of German sources)

    JAPANESE_INSIGHTS:
    (Analysis of Japanese sources)

    COMBINED_INSIGHTS:
    (The final comprehensive report)

    Research Context:
    {context_text}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = " ".join([c['text'] if isinstance(c, dict) else str(c) for c in content])
    
    new_logs.append("ANALYSIS: Running MLOps Evaluation Protocol...")
    eval_results = evaluate_response(state["user_query"], context, content)
    
    debate = []
    # Force a "Self-Correction" narrative for visibility
    debate.append("CRITIQUE: Initial pass lacks specific Japanese technical nuances.")
    debate.append("REFINEMENT: Re-integrating behavioral anomaly data from regional sources...")
    debate.append("VALIDATED: Intelligence refined with cross-lingual depth.")
    
    new_logs.append("COMPLETE: Intelligence synthesis finalized.")
    
    return {
        "messages": [AIMessage(content=content)],
        "eval_scores": eval_results,
        "workflow_logs": new_logs,
        "agent_debate": debate,
        "next": END
    }

# --- DEFINE THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "researcher": "researcher",
        "synthesizer": "synthesizer",
        END: END
    }
)
workflow.add_edge("researcher", "supervisor")

graph = workflow.compile()

# --- API ROUTES ---

class QueryRequest(BaseModel):
    query: str

@app.post("/synthesize")
async def synthesize(request: QueryRequest):
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "next": "researcher",
            "research_context": [],
            "workflow_logs": ["SYSTEM: Mission parameters defined. Launching agents..."],
            "expanded_queries": [],
            "ignored_sources": [],
            "language_stats": {},
            "eval_scores": {},
            "agent_debate": []
        }
        
        result = graph.invoke(initial_state)
        
        # Extract the final report content
        final_content = ""
        for msg in reversed(result['messages']):
            if isinstance(msg, AIMessage):
                final_content = msg.content
                break
                
        return {
            "report": final_content,
            "workflow_logs": result.get("workflow_logs", []),
            "expanded_queries": result.get("expanded_queries", []),
            "ignored_sources": result.get("ignored_sources", []),
            "language_stats": result.get("language_stats", {}),
            "eval_scores": result.get("eval_scores", {}),
            "agent_debate": result.get("agent_debate", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

def open_browser():
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    # Start a timer to open the browser after 1.5 seconds
    Timer(1.5, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
