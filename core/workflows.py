from typing import TypedDict, List
from langgraph.graph import StateGraph
from core.llm import generate
from db.pg import similarity_search
from db.redis import cache

class State(TypedDict):
    question: str
    docs: List[str]
    answer: str

def retrieve(state: State):
    q = state["question"].strip()
    cached = cache.get(q)
    if cached:
        return {"docs": cached["docs"]}
    docs = similarity_search(q, k=4)
    cache.set(q, {"docs": docs}, ex=60*60)  # 1h TTL
    return {"docs": docs}

def generate_answer(state: State):
    docs = state.get("docs", [])
    context = "\n\n".join(docs) if docs else "(no context)"
    prompt = (
        "You answer strictly from the provided context. "
        "If the answer isn't present, say you can't find it.\n\n"
        f"Context:\n{context}\n\nQ: {state['question']}\nA:"
    )
    answer = generate(prompt)
    return {"answer": answer}

_graph = StateGraph(State)
_graph.add_node("retrieve", retrieve)
_graph.add_node("generate", generate_answer)
_graph.add_edge("retrieve", "generate")
_graph.set_entry_point("retrieve")
qa_graph = _graph.compile()
