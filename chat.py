"""
Terminal chat. Usage:
    python chat.py
Type 'quit' to exit.
"""
from rag import Retriever, answer, PROVIDER, LLM_MODEL

print(f"Loading models... (LLM: {PROVIDER}/{LLM_MODEL})")
retriever = Retriever()
history = []
print("Ask about treatments, hospitals, or doctors. Type 'quit' to exit.\n")

while True:
    q = input("You: ").strip()
    if q.lower() in {"quit", "exit", ""}:
        break
    text, sources, history = answer(retriever, q, history)
    print(f"\nAssistant: {text}\n")
    for i, title, url in sources:
        print(f"  [{i}] {title} - {url}")
    print()