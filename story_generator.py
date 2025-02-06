import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage 

# Load environment variables from .env file
load_dotenv()
# print(os.environ)

# System prompt for character generation
character_system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a fantasy story writer. Generate a character type based on the name."
)

# System prompt for setting generation
setting_system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a world-building expert. Create a setting that fits the character."
)

# System prompt for plot generation
plot_system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a storyteller. Write a short plot involving the character and setting."
)

# Character generation prompt
character_prompt = ChatPromptTemplate.from_messages([
    character_system_prompt,
    HumanMessagePromptTemplate.from_template("Generate a fantasy character type for someone named {name}. Reply ONLY with the type.")
]
)


# Setting generation prompt
setting_prompt = ChatPromptTemplate.from_messages([
    setting_system_prompt,
    HumanMessagePromptTemplate.from_template("Create a magical setting for a character named {name} who is a {character_type}. Reply with ONE sentence.")
]
)

# Plot generation prompt
plot_prompt = ChatPromptTemplate.from_messages([
    plot_system_prompt,
    HumanMessagePromptTemplate.from_template("Write a short story about {name}, a {character_type}, exploring {setting}.")
]
)


llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

def generate_character(state):
    print("Current state:", state)  # Print initial state
    print("Character prompt:", character_prompt)  # Print the prompt template
    
    try:
        messages = character_prompt.format_messages(name=state["name"])
        print("Formatted messages:", messages)  # Print formatted messages
        
        response = llm.invoke(messages)
        print("LLM response:", response)  # Print LLM response
        
        state["character_type"] = response.content.strip()
        print("Updated state:", state)  # Print final state
        return state
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {str(e)}")  # Print any errors
        raise

def generate_setting(state):
    messages = setting_prompt.format_messages(
        character_type=state["character_type"],
        name = state["name"]
    )
    response = llm.invoke(messages)
    state["setting"] = response.content.strip()
    return state

def generate_plot(state):
    messages = plot_prompt.format_messages(
        name=state["name"],
        character_type=state["character_type"],
        setting=state["setting"]
    )
    response = llm.invoke(messages)
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

# # Start with the initial state (name="Luna")
# result = chain.invoke({"name": "Lightning_macquine"})

# # Print the final story
# print("Second Story: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(result["plot"])