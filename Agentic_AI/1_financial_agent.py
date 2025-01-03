import os
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = Groq(model="gemma2-9b-it ",groq_api_key = GROQ_API_KEY)


## web search agent 
web_search_agent = Agent(
     name="web search agent",
     role = "search the web for the information.",
     model = model,
     tools = [DuckDuckGo()],
     instructions = ["Always show source"],
     show_tools_calls = True,
     markdown=True
)


## financial agent
financial_agent = Agent(
     name="financial ai agent",
     role = "get financial information.",
     model = model,
     tools = [YFinanceTools(stock_price=True, 
                            analyst_recommendations=True, 
                            stock_fundamentals=True,
                            company_news=True)],
     instructions = ["use tables to display the data"],
     show_tools_calls = True,
     markdown=True
)

## combining the above two independent agent

multi_ai_agent = Agent(
     team=[web_search_agent, financial_agent],
     model = model,
     instructions = ["Always show source","use tables to display the data"],
     show_tools_calls = True,
     markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendation and share latest news on Tesla.",stream=True)