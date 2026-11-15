import os 
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

# GROQ_API_KEY is picked up automatically from environment
# How to set the environment variable
# export GROQ_API_KEY="<your own key>"
# Where to get the api key from
# https://console.groq.com/keys

llm = ChatGroq(model="openai/gpt-oss-120b")

# --- Why CodeState? ---
# LangGraph passes a single shared state object between every node in the graph.
# CodeState is that shared object. Every node receives it, and every node returns
# a partial dict whose keys merge back into it.
#
# Think of it as a whiteboard in a room where all the workers collaborate.
# Node A writes "code" on the whiteboard. Node B reads "code", writes "review".
# Nobody passes arguments directly to each other — they all read and write the
# same whiteboard.
#
# --- Could each node have a different TypedDict? ---
# No. LangGraph enforces one state type for the whole graph. Every node must
# accept and return values compatible with that single TypedDict. This is by
# design: the graph needs to know the full shape of state at compile time so it
# can route, checkpoint, and replay correctly.
#
# What you CAN do is make the state broad enough to serve everyone, using
# Optional fields for things only some nodes care about:
#   class CodeState(TypedDict):
#       language: str           # set by the caller
#       task: str               # set by the caller
#       code: str               # written by generate_code, read by review_code
#       review: str             # written by review_code
#       approved: bool          # written by a human-in-the-loop node
#
# --- Real-life examples of what goes in here ---
# 1. Coding agent   : task description, generated code, test results, error log,
#                     retry count, final answer
# 2. RAG pipeline   : user question, retrieved documents, drafted answer,
#                     fact-check result, final answer
# 3. Customer support bot : user message, intent, fetched order data,
#                           draft reply, escalation flag
# 4. Research agent : topic, search queries, raw results, summary, citations
#
# The pattern is always the same: earlier nodes populate fields, later nodes
# consume them. Fields a node doesn't touch are simply passed through unchanged.

class CodeState(TypedDict):
    language: str   # uncomment when nodes need to know the target language
    task: str       # uncomment when nodes need the original task description
    code: str         # written by generate_code, read by review_code
    review: str       # written by review_code

def investigate_codestate( state : CodeState ) : 
    print("\n=== ALL CODESTATE VALUES ===")
    for key, value in state.items():
        print(f"\n[{key.upper()}]: {value[:20]}")


# Define some workers.
# These are the folks doing the work with / without LLM.
def generate_code(state: CodeState) -> dict:

    # investigate_codestate(state)

    prompt = f"Write a {state['language']} program to {state['task']}."
    # print(f"prompt = {prompt}")

    response = llm.invoke( prompt )
    code = response.content
    print( f"CODE #### \n {code[:300]} \n#### FULL CODE NOT SHARED FOR BREVITY")
    return {"code": code}

def review_code(state: CodeState) -> dict:
    # investigate_codestate(state)
    review = "Reviewed the code"
    print(f"{review}")
    return {"review": review}


## Build and Compile the Graph
def build_graph() -> StateGraph:
    builder = StateGraph(CodeState)

    # Add the workers, nodes.
    builder.add_node("generator", generate_code)
    builder.add_node("reviewer", review_code)

    # Connect all the workers in a sequence, or web if you prefer
    builder.add_edge(START, "generator")
    builder.add_edge("generator", "reviewer")
    builder.add_edge("generator", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"language": "python", "task": "generate a add function that takes two integers and output the sum of that."})
