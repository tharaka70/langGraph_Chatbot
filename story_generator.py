import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START

# Load environment variables from .env file
load_dotenv()
# print(os.environ)

# PROMPT 1: Generate character type (e.g., wizard, warrior)
character_prompt = PromptTemplate.from_template(
    "Generate a fantasy character type for someone named {name}. Reply ONLY with the type."
)

# PROMPT 2: Generate a setting based on character type
setting_prompt = PromptTemplate.from_template(
    "Create a magical setting for a {character_type}. Reply with ONE sentence."
)

# PROMPT 3: Generate a plot using name, type, and setting
plot_prompt = PromptTemplate.from_template(
    "Write a short story about {name}, a {character_type}, exploring {setting}."
)


llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# --- Node 1: Generate Character Type ---
def generate_character(state):
    # Format the prompt with the current state (name)
    formatted_prompt = character_prompt.format(name=state["name"])
    
    # Call the LLM
    response = llm.invoke(formatted_prompt)
    
    # Update state with the result
    state["character_type"] = response.content.strip()  # e.g., "wizard"
    return state

# --- Node 2: Generate Setting ---
def generate_setting(state):
    formatted_prompt = setting_prompt.format(
        character_type=state["character_type"]
    )
    response = llm.invoke(formatted_prompt)
    state["setting"] = response.content.strip()  # e.g., "a floating castle in the sky"
    return state

# --- Node 3: Generate Plot ---
def generate_plot(state):
    print(state)
    formatted_prompt = plot_prompt.format(
        name=state["name"],
        character_type=state["character_type"],
        setting=state["setting"]
    )
    response = llm.invoke(formatted_prompt)
    state["plot"] = response.content.strip()
    return state

# Define the workflow
workflow = StateGraph(state_schema=dict)  # State is a dictionary

# Add nodes (the functions we defined)
workflow.add_node("generate_character", generate_character)
workflow.add_node("generate_setting", generate_setting)
workflow.add_node("generate_plot", generate_plot)

# Connect nodes in sequence
workflow.add_edge(START, "generate_character")
workflow.add_edge("generate_character", "generate_setting")
workflow.add_edge("generate_setting", "generate_plot")
workflow.add_edge("generate_plot", END)  # END is a special node

# Compile the graph
chain = workflow.compile()

# Start with the initial state (name="Luna")
result = chain.invoke({"name": "Harry potter"})

# Print the final story
print("First Story: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(result["plot"])

# Start with the initial state (name="Luna")
result = chain.invoke({"name": "Lightning_macquine"})

# Print the final story
print("Second Story: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(result["plot"])