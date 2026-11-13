from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# GROQ_API_KEY is picked up automatically from environment
# How to set the environment variable 
# export GROQ_API_KEY="<your own key>"
# Where to get the api key from 
# https://console.groq.com/keys
llm = ChatGroq(model="openai/gpt-oss-120b")

code_prompt = PromptTemplate.from_template(
    "Write a {language} function to perform the {task}"
)

# LCEL — LangChain Expression Language
# The | operator is Python's bitwise OR, but LangChain overrides __or__ on all
# its Runnable components to mean "pipe output of left into input of right".
# These are not plain variables — each is an object with an invoke() method.
# Nothing executes here; this just wires the three steps into a single Runnable.
# When code_chain.invoke({...}) is called, data flows left to right:
#   1. code_prompt  : dict {"language", "task"} → fills template → prompt string
#   2. llm          : prompt string → Groq API call → ChatMessage object
#   3. StrOutputParser : ChatMessage → extracts .content → plain string
code_chain = code_prompt | llm | StrOutputParser()

if __name__ == "__main__":

    # result = code_chain.invoke(
    #     {"language": "python", "task": "give a list of first prime numbers"}
    # )
    result = code_chain.invoke(
        {"language": "python", "task": "give the list of pythogorean triplets"}
    )
    print(result)
