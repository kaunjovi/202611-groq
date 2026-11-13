from groq import Groq

# Automatically uses the GROQ_API_KEY environment variable
client = Groq()

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
)

print(completion.choices[0].message.content)
