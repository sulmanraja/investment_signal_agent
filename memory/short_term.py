"""Short-term memory — in-process conversation buffer for a single analysis run.

Uses langchain_core messages (HumanMessage / AIMessage) directly.
Compatible with LangChain 1.x+ (ConversationBufferMemory was removed in 1.0).
"""

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ShortTermMemory:
    """In-memory buffer that holds the conversation history for one run."""

    def __init__(self):
        self._messages: list[BaseMessage] = []

    def add(self, human: str, ai: str) -> None:
        self._messages.append(HumanMessage(content=human))
        self._messages.append(AIMessage(content=ai))

    def load(self) -> list[BaseMessage]:
        return list(self._messages)

    def as_string(self) -> str:
        """Return the buffer as a plain-text transcript."""
        lines = []
        for msg in self._messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


if __name__ == "__main__":
    print("=== short_term memory local test ===")
    mem = ShortTermMemory()
    mem.add("What is NVDA's revenue growth?", "Data-center revenue grew 427% YoY to $22.6B.")
    mem.add("What are the key risks?", "Export controls on H100 chips and elevated valuation.")
    print(f"Messages stored: {len(mem)}")
    for msg in mem.load():
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content[:80]}")
    print("\nAs string:\n" + mem.as_string())
