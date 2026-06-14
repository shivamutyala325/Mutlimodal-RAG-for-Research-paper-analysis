from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.storage.retriver import Retriever
from pydantic import BaseModel


class RouteDecision(BaseModel):
    retrieve: bool


def route_node(state):
    model = ChatGroq(model=state.chat_model)
    query = state.messages[-1].content

    system = SystemMessage(content=(
        "You are a decision maker for a RAG chatbot. "
        "Decide whether retrieving context from a research paper is needed to answer the query. "
        "Return retrieve=True if the query asks about paper content, concepts, results, or methods. "
        "Return retrieve=False if the query is a greeting, small talk, or can be answered from conversation history alone."
    ))

    result = model.with_structured_output(RouteDecision).invoke([
        system,
        HumanMessage(content=f"Conversation history: {state.messages[:-1]}\nQuery: {query}")
    ])

    return {"retrieve": result.retrieve}


def retrive_node(state):
    query = state.messages[-1].content
    retriever = Retriever()
    results = retriever.retrieve(query, paper_id=state.paper_id)
    return {"context": [r["chunk"] for r in results]}


def chat_node(state):
    model = ChatGroq(model=state.chat_model)
    query = state.messages[-1].content

    system = SystemMessage(content=(
        "You are a helpful assistant for understanding research papers. "
        "Answer the user's question based on the conversation history and any provided context. "
        "If the context does not contain enough information, say so clearly."
    ))

    history = state.messages[:-1]

    if state.context:
        context_parts = []
        for i, chunk in enumerate(state.context):
            chunk_type = chunk.get("chunk_type", "text")
            content = chunk.get("content", "")
            if chunk_type == "image":
                context_parts.append(f"[Chunk {i+1}] (Image)\nDescription: {content}")
            elif chunk_type == "table":
                context_parts.append(f"[Chunk {i+1}] (Table)\n{content}")
            else:
                context_parts.append(f"[Chunk {i+1}] (Text)\n{content}")
        current = HumanMessage(content=f"Context:\n\n{''.join(context_parts)}\n\nQuestion: {query}")
    else:
        current = HumanMessage(content=query)

    response = model.invoke([system] + history + [current])
    return {"messages": [response]}