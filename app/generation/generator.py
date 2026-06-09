from langchain_core.messages import HumanMessage
from app.generation.graph import Chatbot
from app.utils.logger import get_logger


logger = get_logger(__name__)
chatgraph = Chatbot().get_chatgraph()
config = {"configurable": {"thread_id": "shiva"}}


def display_retrieved_chunks(chunks):
    if not chunks:
        return
    print("\n--- Retrieved Context ---")
    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("chunk_type", "text")
        metadata = chunk.get("metadata", {})

        if chunk_type == "image":
            image_path = metadata.get("image_path", "N/A")
            caption = metadata.get("caption", "")
            print(f"\n[Chunk {i+1}] IMAGE")
            print(f"  Path    : {image_path}")
            if caption:
                print(f"  Caption : {caption}")

        elif chunk_type == "table":
            content = chunk.get("content", "")
            raw_table = content.split("Raw Table:")[-1].strip() if "Raw Table:" in content else content
            print(f"\n[Chunk {i+1}] TABLE")
            print(f"  Markdown:\n{raw_table}")

        else:
            snippet = chunk.get("content", "")[:120].replace("\n", " ")
            print(f"\n[Chunk {i+1}] TEXT")
            print(f"  {snippet}...")

    print("-------------------------")


class Chat():
    def __init__(self):
        logger.info(f"initated the chat instance with thread_id : {config['configurable']}")
        while True:
            user_query = input("\nEnter your query or type 'exit' to stop: ")
            if user_query.lower() == "exit":
                break

            result = chatgraph.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=config
            )

            display_retrieved_chunks(result.get("context", []))
            print("\nAnswer:", result["messages"][-1].content)
            print()
