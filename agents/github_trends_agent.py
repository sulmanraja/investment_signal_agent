"""GitHub Trends Agent (CrewAI).

Fetches and interprets GitHub repository trends as a developer-adoption
and technology-momentum signal for investment analysis.
"""

import os
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
from tools.github_tool import fetch_github_trends

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")


def build_github_trends_agent() -> Agent:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    return Agent(
        role="GitHub Trend Analyst",
        goal=(
            "Interpret GitHub repository trends to identify technology adoption "
            "momentum as an investment signal."
        ),
        backstory=(
            "You are a tech-savvy analyst who monitors open-source activity as a "
            "leading indicator of enterprise technology adoption. You translate "
            "star counts, fork rates, and repo growth into investment insights."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


class GitHubTrendsAgent:
    """CrewAI-backed GitHub trend analyst."""

    def __init__(self):
        self.agent = build_github_trends_agent()

    def run(self, ticker: str, topic: str, language: str | None = "Python") -> str:
        repos = fetch_github_trends(topic, language=language, limit=10)
        repo_summary = "\n".join(
            f"  {r['name']} ({r['stars']:,} stars): {r['description'][:80]}"
            for r in repos
        )

        task = Task(
            description=(
                f"Analyse the following GitHub trending repositories related to '{topic}' "
                f"(relevant to {ticker}):\n\n{repo_summary}\n\n"
                "Summarise:\n"
                "1. What technology themes are gaining momentum?\n"
                "2. How does this activity signal developer/enterprise adoption?\n"
                "3. What is the investment implication for {ticker}? (1-2 sentences)"
            ),
            agent=self.agent,
            expected_output="Technology adoption analysis with investment signal for the ticker.",
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    print("=== github_trends_agent local test ===")
    agent = GitHubTrendsAgent()
    output = agent.run("NVDA", "CUDA GPU inference", language="Python")
    print(output)
