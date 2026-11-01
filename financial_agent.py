# importing libraries

import os # python ke operating system se env vaariable read krne ke kaam aata hai.
from dotenv import load_dotenv # dot env file se key settings read karne main help krta hai.
from agno.agent import Agent # ye agent kaa building block hai is se hum specialist worker define karte hain.
from agno.team import Team # multi- agent team ka building block. Team leaader speaclists ko cordinate karta hai.
from agno.team.mode import TeamMode # hum kis teamwork mode ko use kaar rahe hain, ye wo btata hai.
from agno.models.groq import Groq # selected groq hosted llm tak access dene ka model adapter.
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def build_financial_agents(model_id=DEFAULT_MODEL):
    web_agent = Agent( # framework builds a model backed agent configuration
        name="Financial News Researcher",
        role="Search the public web for company news and cite original links.",
        model=Groq(id=model_id),
        tools=[DuckDuckGoTools(enable_search=True, enable_news=True)],
        instructions=[
            "Search for the company or ticker named in the question.",
            "Give actual article titles, source links and publication dates when available.",
            "Say clearly when the tools return no usable results or dates.",
            "Avoid invented articles, prices, statistics or guaranteed predictions.",
        ],
        markdown=True,
    )

    stock_agent = Agent(
        name="Stock Data Analyst",
        role="Use the financial data tools to examine prices, company fundamentals and analyst estimates.",
        model=Groq(id=model_id),
        tools=[
            YFinanceTools(
                enable_stock_price=True,
                enable_company_info=True,
                enable_stock_fundamentals=True,
                enable_analyst_recommendations=True,
                enable_company_news=True,
                enable_historical_prices=True,
            )
        ],
        instructions=[
            "Use financial tools for numerical market claims, never model memory.",
            "Clearly identify ticker, reporting currency and data date when available.",
            "Explain finance terms in simple language suitable for a young beginner.",
            "Use compact Markdown tables for comparable financial numbers.",
            "Separate confirmed observations, analyst opinions and uncertainty.",
            "If data is absent or a tool fails, state the limitation explicitly.",
        ],
        markdown=True,
    )

    finance_team = Team(
        name="Financial Intelligence Research Team",
        model=Groq(id=model_id),
        members=[web_agent, stock_agent],
        mode=TeamMode.broadcast,
        instructions=[
            "Ask both specialists about the user's financial research question.",
            "Combine their separate observations into an educational research note.",
            "Show sections: Overview, Market Data, Recent News, Risks and Limitations, Sources.",
            "Use only evidence actually returned by members; do not invent values or links.",
            "Compare possible positives and negatives without recommending a trade.",
            "Report conflicts or missing evidence instead of smoothing them over.",
            "End with: Educational research only, not personalized investment advice.",
        ],
        store_member_responses=True,
        show_members_responses=True,
        markdown=True,
    )

    return web_agent, stock_agent, finance_team


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Add GROQ_API_KEY to .env before running the agent.")
    _, _, team = build_financial_agents()
    team.print_response(
        "Research NVDA: show available stock metrics, recent relevant news, risks and sources. Explain for beginners.",
        stream=True,
    )
