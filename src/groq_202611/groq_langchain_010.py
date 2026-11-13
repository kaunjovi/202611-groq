from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# GROQ_API_KEY is picked up automatically from environment
llm = ChatGroq(model="openai/gpt-oss-120b")

code_prompt = PromptTemplate.from_template(
    "Write a {language} function to perform the {task}"
)

code_chain = code_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    result = code_chain.invoke(
        {"language": "python", "task": "give a list of first prime numbers"}
    )

    print(result)
