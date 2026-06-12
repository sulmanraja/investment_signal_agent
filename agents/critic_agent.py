"""Critic Agent (CrewAI).

Scores and ranks the A/B/C/D thought branches produced by the
ThoughtGeneratorAgent using a structured rubric. Outputs a ranked list
with confidence scores.
"""

import os
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")


def build_critic_agent() -> Agent:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    return Agent(
        role="Investment Thesis Critic",
        goal=(
            "Evaluate each investment stance branch on evidence quality, "
            "logical consistency, risk awareness, and signal alignment. "
            "Return a ranked list with a 0-10 confidence score per branch."
        ),
        backstory=(
            "You are a rigorous risk manager and devil's advocate. You stress-test "
            "investment theses by scrutinising evidence quality, identifying logical "
            "gaps, and ensuring all material risks are acknowledged."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


class CriticAgent:
    """CrewAI-backed thesis critic."""

    def __init__(self):
        self.agent = build_critic_agent()

    def run(self, ticker: str, branches: str) -> str:
        task = Task(
            description=(
                f"Critically evaluate the following investment stance branches for {ticker}:\n\n"
                f"{branches}\n\n"
                "For each branch (A/B/C/D):\n"
                "1. Identify the strongest supporting evidence.\n"
                "2. Identify the key weakness or missing consideration.\n"
                "3. Assign a confidence score from 0 (very weak) to 10 (very strong).\n\n"
                "Finally, rank the branches by confidence score and state the recommended stance."
            ),
            agent=self.agent,
            expected_output="Scored and ranked investment stances with a final recommendation.",
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    print("=== critic_agent local test ===")
    sample_branches = (
        "A) Bullish: Record data-center revenue and AI tailwinds support continued growth.\n"
        "B) Bearish: Inverted yield curve and valuation stretch pose downside risk.\n"
        "C) Neutral: Wait for next earnings confirmation before adding exposure.\n"
        "D) Contrarian: Export restrictions on H100/A100 chips are underpriced by the market."
    )
    agent = CriticAgent()
    output = agent.run("NVDA", sample_branches)
    print(output)
