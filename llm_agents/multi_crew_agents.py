from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import OpenAI

# Initialize the LLM and search tool
llm = OpenAI(temperature=0.7)
search_tool = DuckDuckGoSearchRun()

# Define agents
market_analyst = Agent(
    role="Market Research Analyst",
    goal="Analyze market trends, size, and potential for the startup's industry",
    backstory="You're a seasoned market analyst with a keen eye for emerging trends and market dynamics.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
)

financial_analyst = Agent(
    role="Financial Analyst",
    goal="Evaluate the startup's financial health, projections, and funding needs",
    backstory="You're an experienced financial analyst specializing in startup valuations and financial modeling.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
)

tech_expert = Agent(
    role="Technology Expert",
    goal="Assess the startup's technology, its uniqueness, and potential scalability",
    backstory="You're a tech guru with extensive knowledge across various tech stacks and emerging technologies.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
)

competitor_analyst = Agent(
    role="Competitive Intelligence Specialist",
    goal="Identify and analyze key competitors, their strategies, and market positioning",
    backstory="You're an expert in competitive analysis with a track record of uncovering hidden market players.",
    verbose=True,
    llm=llm,
    tools=[search_tool],
)

investment_strategist = Agent(
    role="Investment Strategist",
    goal="Synthesize all information to provide an investment recommendation",
    backstory="You're a veteran VC partner known for identifying unicorn startups early.",
    verbose=True,
    llm=llm,
)


# Define tasks
def create_tasks(startup_name):
    return [
        Task(
            description=f"Conduct a comprehensive market analysis for {startup_name}'s industry.",
            agent=market_analyst,
        ),
        Task(
            description=f"Perform a detailed financial analysis of {startup_name}.",
            agent=financial_analyst,
        ),
        Task(
            description=f"Evaluate the technological aspects and innovation potential of {startup_name}.",
            agent=tech_expert,
        ),
        Task(
            description=f"Analyze the competitive landscape for {startup_name}.",
            agent=competitor_analyst,
        ),
        Task(
            description=f"Synthesize all findings and provide an investment recommendation for {startup_name}.",
            agent=investment_strategist,
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
            investment_strategist,
        ],
        tasks=create_tasks(startup_name),
        verbose=2,
        process=Process.sequential,
    )
    return crew.kickoff()


# Example usage
startup_name = "LangChain"
result = analyze_startup(startup_name)
print(result)
