"""Thought Generator Agent (CrewAI).

Generates A/B/C/D investment thesis branches from aggregated signal context.
Each branch represents a distinct stance: Bullish / Bearish / Neutral / Contrarian.
"""

import os
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")


def build_thought_generator_agent() -> Agent:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
    return Agent(
        role="Investment Thesis Generator",
        goal=(
            "Generate four distinct investment stance branches (Bullish, Bearish, "
            "Neutral, Contrarian) grounded in the provided signal context."
        ),
        backstory=(
            "You are a senior portfolio strategist who excels at scenario analysis. "
            "Given data from SEC filings, macro indicators, GitHub trends, and news, "
            "you articulate four well-reasoned investment stances for a given ticker."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


class ThoughtGeneratorAgent:
    """CrewAI-backed thought generator."""

    def __init__(self):
        self.agent = build_thought_generator_agent()

    def run(self, ticker: str, context: str) -> str:
        task = Task(
            description=(
                f"Given the following investment signal context for {ticker}:\n\n"
                f"{context}\n\n"
                "Generate four investment stance branches:\n"
                "A) Bullish — reasons to buy\n"
                "B) Bearish — reasons to sell / avoid\n"
                "C) Neutral — hold / wait-and-see arguments\n"
                "D) Contrarian — non-consensus view with supporting logic\n\n"
                "Each branch should be 2-3 concise sentences."
            ),
            agent=self.agent,
            expected_output="Four labeled investment stances (A/B/C/D).",
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    print("=== thought_generator_agent local test ===")
    sample_context = (
        "NVDA reported record data-center revenue of $22B (+427% YoY). "
        "Management guided for continued strong AI chip demand. "
        "GitHub shows 3,400 new CUDA repos in 90 days. "
        "Fed funds rate at 5.25%. 10Y-2Y yield curve = -0.3 (inverted)."
    )
    agent = ThoughtGeneratorAgent()
    output = agent.run("NVDA", sample_context)
    print(output)
