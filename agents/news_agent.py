"""News Agent (CrewAI).

Fetches and analyses recent news articles to surface sentiment and
event-driven signals for investment analysis.
"""

import os
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
from tools.newsdata_tool import fetch_news

LLM_MODEL = os.getenv("AGENT_LLM", "granite3.3:8b")


def build_news_agent() -> Agent:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    return Agent(
        role="News Sentiment Analyst",
        goal=(
            "Analyse recent news articles to extract sentiment signals, "
            "identify material events, and assess their investment implications."
        ),
        backstory=(
            "You are a news-driven quantitative analyst who monitors business and "
            "technology news to identify events that materially impact stock prices. "
            "You rate sentiment (positive/neutral/negative) and flag key catalysts."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


class NewsAgent:
    """CrewAI-backed news sentiment analyst."""

    def __init__(self):
        self.agent = build_news_agent()

    def run(self, ticker: str, query: str | None = None, limit: int = 10) -> str:
        query = query or f"{ticker} earnings revenue guidance"
        articles = fetch_news(query, limit=limit)

        article_summary = "\n".join(
            f"  [{a['published_at']}] {a['source']}: {a['title']}"
            for a in articles
        )

        task = Task(
            description=(
                f"Analyse the following recent news headlines for {ticker}:\n\n"
                f"{article_summary}\n\n"
                "Provide:\n"
                "1. Overall sentiment: Positive / Neutral / Negative.\n"
                "2. Top 3 material events or themes identified.\n"
                "3. Short-term investment implication (1 sentence)."
            ),
            agent=self.agent,
            expected_output="Sentiment classification, key events, and investment implication.",
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    print("=== news_agent local test ===")
    agent = NewsAgent()
    output = agent.run("NVDA", query="NVIDIA AI chips earnings")
    print(output)
