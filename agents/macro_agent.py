"""Macro Agent (CrewAI).

Fetches FRED macroeconomic indicators and contextualises them as
top-down signals that affect the investment environment for a ticker.
"""

import os
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
from tools.fred_macro_tool import get_latest_snapshot

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")


def build_macro_agent() -> Agent:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    return Agent(
        role="Macro Economist",
        goal=(
            "Interpret macroeconomic indicators (rates, inflation, growth, credit) "
            "to assess the top-down investment environment for a given sector or ticker."
        ),
        backstory=(
            "You are a top-down macro strategist who monitors the Federal Reserve, "
            "yield curve dynamics, inflation, and credit markets to contextualise "
            "equity valuations and sector rotation signals."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


class MacroAgent:
    """CrewAI-backed macro economist."""

    def __init__(self):
        self.agent = build_macro_agent()

    def run(self, ticker: str, sector: str = "Technology") -> str:
        try:
            snapshot = get_latest_snapshot()
            macro_text = "\n".join(f"  {k:<25s} {v}" for k, v in snapshot.items())
        except EnvironmentError:
            macro_text = "  [FRED_API_KEY not set — using stub data]\n  fed_funds_rate: 5.25\n  10y_treasury: 4.35"

        task = Task(
            description=(
                f"Given the following current macroeconomic indicators:\n\n{macro_text}\n\n"
                f"Assess the macro environment for {ticker} in the {sector} sector:\n"
                "1. Is the rate/inflation environment supportive or headwind for growth equities?\n"
                "2. What does the yield curve signal about recession risk?\n"
                "3. Overall macro stance: Supportive / Neutral / Headwind. (1-2 sentences)"
            ),
            agent=self.agent,
            expected_output="Macro environment assessment with stance for the sector/ticker.",
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    print("=== macro_agent local test ===")
    agent = MacroAgent()
    output = agent.run("NVDA", sector="Semiconductors")
    print(output)
