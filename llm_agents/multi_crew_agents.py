from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

# Initialize the LLM and search tool
llm = ChatOpenAI(temperature=0.7)
search_tool = DuckDuckGoSearchRun()

# Define agents
market_analyst = Agent(
    role="Market Research Analyst",
    goal="Analyze market trends, size, and potential for the startup's industry",
    backstory="You're a seasoned market analyst with a keen eye for emerging trends and market dynamics.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
    model_config=ConfigDict(populate_by_name=True),
)

financial_analyst = Agent(
    role="Financial Analyst",
    goal="Evaluate the startup's financial health, projections, and funding needs",
    backstory="You're an experienced financial analyst specializing in startup valuations and financial modeling.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
    model_config=ConfigDict(populate_by_name=True),
)

tech_expert = Agent(
    role="Technology Expert",
    goal="Assess the startup's technology, its uniqueness, and potential scalability",
    backstory="You're a tech guru with extensive knowledge across various tech stacks and emerging technologies.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
    model_config=ConfigDict(populate_by_name=True),
)

competitor_analyst = Agent(
    role="Competitive Intelligence Specialist",
    goal="Identify and analyze direct competitors, their strategies, and market positioning",
    backstory="You're an expert in competitive analysis with a track record of uncovering hidden market players. You focus on direct competitors that offer similar products or services in the same target market.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
    model_config=ConfigDict(populate_by_name=True),
)

contrarian_analyst = Agent(
    role="Contrarian Analyst",
    goal="Challenge assumptions and provide alternative viewpoints on the startup's potential",
    backstory="You're a skeptical analyst known for identifying potential pitfalls and weaknesses in seemingly promising startups.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
    model_config=ConfigDict(populate_by_name=True),
)

investment_strategist = Agent(
    role="Investment Strategist",
    goal="Synthesize all information, including contrarian views, to provide a balanced investment recommendation",
    backstory="You're a veteran VC partner known for making well-informed, objective investment decisions by considering both positive and negative aspects.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
    model_config=ConfigDict(populate_by_name=True),
)


# Define tasks
def create_tasks(startup_name):
    return [
        Task(
            description=f"Conduct a comprehensive market analysis for {startup_name}'s industry.",
            agent=market_analyst,
            expected_output="A detailed market analysis report including market size, growth trends, and key players.",
        ),
        Task(
            description=f"Perform a detailed financial analysis of {startup_name}.",
            agent=financial_analyst,
            expected_output="A financial report including revenue projections, burn rate, and funding requirements.",
        ),
        Task(
            description=f"Evaluate the technological aspects and innovation potential of {startup_name}.",
            agent=tech_expert,
            expected_output="A technology assessment report highlighting the startup's innovations and potential scalability.",
        ),
        Task(
            description=f"Analyze the direct competitive landscape for {startup_name}, focusing on companies offering similar products or services in the same target market.",
            agent=competitor_analyst,
            expected_output="A focused competitive analysis report identifying direct competitors, their strategies, and market positioning.",
        ),
        Task(
            description=f"Provide a contrarian analysis of {startup_name}, challenging assumptions and identifying potential weaknesses or risks.",
            agent=contrarian_analyst,
            expected_output="A critical analysis report highlighting potential pitfalls, weaknesses, and alternative viewpoints on the startup's potential.",
        ),
        Task(
            description=f"Synthesize all findings, including contrarian views, and provide a balanced investment recommendation for {startup_name}.",
            agent=investment_strategist,
            expected_output="A comprehensive and objective investment recommendation that considers both positive aspects and potential risks.",
        ),
    ]


# Create the crew
def analyze_startup(startup_name):
    crew = Crew(
        agents=[
            market_analyst,
            financial_analyst,
            tech_expert,
            competitor_analyst,
            contrarian_analyst,
            investment_strategist,
        ],
        tasks=create_tasks(startup_name),
        verbose=True,
        process=Process.sequential,
    )
    return crew.kickoff()


# Example usage
startup_name = "OpenAI"
result = analyze_startup(startup_name)
print(result)
