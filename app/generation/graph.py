from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from operator import add
from pydantic import BaseModel
from app.generation.nodes import chat_node, route_node, retrive_node
import os

chat_model = os.getenv("CHAT_MODEL")

class GraphState(BaseModel):
    messages: Annotated[list, add]
    context: list[dict] = []
    chat_model: str = "mistral"
    retrieve: bool = False


def route_decision(state):
    return "retrive_node" if state.retrieve else "chat_node"


class Chatbot:
    def __init__(self, chat_model="mistral"):
        self.chat_model = chat_model

    def get_chatgraph(self):
        builder = StateGraph(GraphState)
        checkpointer = MemorySaver()

        builder.add_node("route_node", route_node)
        builder.add_node("retrive_node", retrive_node)
        builder.add_node("chat_node", chat_node)

        builder.add_edge(START, "route_node")
        builder.add_conditional_edges("route_node", route_decision, {
            "retrive_node": "retrive_node",
            "chat_node": "chat_node"
        })
        builder.add_edge("retrive_node", "chat_node")
        builder.add_edge("chat_node", END)

        return builder.compile(checkpointer=checkpointer)
