import os
from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from financial_agent import build_financial_agents

load_dotenv()
st.set_page_config(page_title="Analytix Camp | Financial Agent Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker, period):
    prices = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if prices.empty:
        return pd.DataFrame()
    return prices[["Close", "Volume"]].dropna(subset=["Close"])


st.title("📈 Financial Agent Lab")
st.caption("Aicademy 360 | Phidata concepts, implemented with modern Agno")
st.info("Two specialist agents + one coordinator | DuckDuckGo news + Yahoo Finance data | Groq LLM")

with st.sidebar:
    st.header("Mission Control")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    model_id = st.selectbox("Model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"], index=0)
    ticker = st.selectbox("Company ticker", ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], index=0)
    mode = st.radio("Who should answer?", ["Team: News + Finance", "Stock specialist", "News specialist"], index=0)
    period = st.selectbox("Chart history", ["1mo", "3mo", "6mo", "1y"], index=1)
    st.caption("A ticker is a short market symbol. This is an educational, read-only research tool.")
    if st.button("Clear AI result", use_container_width=True):
        st.session_state.pop("agent_result", None)
        st.rerun()

quick_prompts = {
    "Company overview": "Explain the company's business and latest available stock metrics in simple words.",
    "Price and fundamentals": "Explain recent price action, available valuation metrics and two key financial risks.",
    "News and risks": "Find recent relevant news and explain potential positive and negative implications, with sources.",
    "Analyst expectations": "Summarize available analyst recommendations, explain what they mean, and explain uncertainty.",
}

report_tab, chart_tab, evidence_tab, learn_tab = st.tabs([
    "🧠 Ask the Agents", "📊 Market Chart", "🔍 Actual Agent Activity", "🎓 How the System Works"
])

with report_tab:
    st.subheader("Ask a financial research question")
    mission = st.selectbox("Quick question", list(quick_prompts), index=0)
    custom = st.text_area("Or write your own question", placeholder="What does the P/E ratio mean for this company?", height=95)
    query = custom.strip() or quick_prompts[mission]
    st.caption(f"Research subject: {ticker} | Mode: {mode}. Responses may be delayed, missing, or inaccurate.")
    if st.button("🚀 Run Financial Research", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter your Groq API key in the sidebar or a private .env file.")
        else:
            try:
                os.environ["GROQ_API_KEY"] = api_key
                news_agent, stock_agent, finance_team = build_financial_agents(model_id)
                selected = finance_team if mode.startswith("Team") else stock_agent if mode == "Stock specialist" else news_agent
                prompt = f"Company ticker: {ticker}. Research question: {query}"
                with st.spinner("Running real agent tool calls and preparing the response..."):
                    response = selected.run(prompt)
                member_runs = list(getattr(response, "member_responses", None) or [])
                if not member_runs:
                    member_runs = [response]
                activity = []
                for run in member_runs:
                    tools = []
                    for tool in (getattr(run, "tools", None) or []):
                        tools.append({
                            "tool": str(getattr(tool, "tool_name", "Unknown tool")),
                            "arguments": str(getattr(tool, "tool_args", ""))[:1200],
                            "result_preview": str(getattr(tool, "result", ""))[:3500],
                        })
                    activity.append({
                        "agent": getattr(run, "agent_name", None) or mode,
                        "answer": str(getattr(run, "content", "") or ""),
                        "tools": tools,
                    })
                st.session_state.agent_result = {
                    "ticker": ticker,
                    "mode": mode,
                    "model": model_id,
                    "answer": str(getattr(response, "content", "") or "No answer was returned."),
                    "activity": activity,
                    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            except Exception as error:
                st.error(f"Agent run failed: {error}")
                st.info("Check your Groq key/model, network, quota and external finance/news tool availability.")
    result = st.session_state.get("agent_result")
    if result:
        st.divider()
        st.caption(f"Last successful run: {result['generated_at']} | {result['ticker']} | {result['mode']}")
        st.markdown(result["answer"])
        st.download_button("Download research note (.md)", result["answer"], f"{result['ticker']}_research_note.md", "text/markdown")
        st.warning("Educational demonstration only. Verify market data and original sources before making any investment decision.")

with chart_tab:
    st.subheader("Independent market-data chart")
    st.caption("This chart calls Yahoo Finance through Python. It is not a chart invented by the LLM or evidence that the agent used a tool.")
    if st.button("Load / Refresh Price History", use_container_width=True):
        load_prices.clear()
        st.session_state.chart_selection = (ticker, period)
    if st.session_state.get("chart_selection") != (ticker, period):
        st.info("Click Load / Refresh Price History to request actual market data.")
    else:
        try:
            with st.spinner("Reading available historical prices..."):
                prices = load_prices(ticker, period)
            if prices.empty:
                st.warning("No historical prices were returned for the selected ticker and period.")
            else:
                first = float(prices["Close"].iloc[0])
                last = float(prices["Close"].iloc[-1])
                period_change = (last / first - 1) * 100 if first else 0.0
                currency = "USD" if ticker in {"NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"} else "Check source"
                left, middle, right = st.columns(3)
                left.metric("Latest available daily close", f"{currency} {last:,.2f}")
                middle.metric("Selected-window change", f"{period_change:+.2f}%")
                right.metric("Last bar date", prices.index[-1].strftime("%Y-%m-%d"))
                st.line_chart(prices["Close"], height=340)
                with st.expander("Show last 12 historical rows"):
                    st.dataframe(prices.tail(12), use_container_width=True)
                st.caption("Daily bar may be provisional during a trading session. Values may be stale or adjusted by the provider. Verify before using them.")
        except Exception as error:
            st.warning(f"Yahoo Finance was unavailable: {error}")

with evidence_tab:
    st.subheader("Actual tool usage and member outputs")
    result = st.session_state.get("agent_result")
    if not result:
        st.info("Run a research question first. Only actual returned agent outputs are displayed here.")
    else:
        for index, item in enumerate(result["activity"], start=1):
            with st.expander(f"{index}. {item['agent']} | {len(item['tools'])} reported tool calls", expanded=True):
                st.markdown(item["answer"] or "No separate member summary was available in this response.")
                for tool_number, tool in enumerate(item["tools"], start=1):
                    st.write("Tool:", tool["tool"])
                    st.code(tool["arguments"], language="text")
                    st.text_area("Returned observation (truncated)", tool["result_preview"], height=125, key=f"tool_{index}_{tool_number}")
        st.caption("If no tool calls are listed, no tool-call evidence was exposed by this run. Do not assume external data was fetched.")

with learn_tab:
    st.subheader("How the three-agent research desk works")
    st.code(
        "USER QUESTION\n"
        "      |\n"
        "      v\n"
        "AGNO TEAM COORDINATOR\n"
        "      |\n"
        "      +--> FINANCIAL NEWS RESEARCHER --> DUCKDUCKGO\n"
        "      |\n"
        "      +--> STOCK DATA ANALYST ------> YFINANCE\n"
        "      |\n"
        "      v\n"
        "SOURCE-BASED RESEARCH SUMMARY\n"
        "      |\n"
        "      v\n"
        "HUMAN SOURCE AND RISK REVIEW",
        language="text",
    )
    st.markdown("**Phidata → Agno:** The project from the uploaded files used the legacy `phi` package. This class uses its actively maintained successor, `agno`, with `Team` and official toolkits.")
    st.markdown("**Specialists vs team:** A single specialist can research one part. Team mode broadcasts to both specialists and the leader synthesizes their results.")
    st.markdown("**Financial vocabulary:** **Price** is a market quote; **P/E** compares share price with earnings per share; **analyst recommendation** is an opinion, not a promise; **risk** means possible uncertainty or loss.")
    st.markdown("**Learning next:** AgentOS, authorization, evaluation, tracing, reliable source checking, market-data licensing, scheduled reports and human approvals.")
    st.warning("No actual orders, trades or portfolio transactions are performed by this classroom application.")

st.divider()
st.caption("Aicademy 360 | Agno + Groq + Yahoo Finance + DuckDuckGo | For supervised learning and research only")
