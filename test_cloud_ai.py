import os
from openai import OpenAI

print(
    "API key configured:",
    bool(os.getenv("OPENAI_API_KEY"))
)

if not os.getenv("OPENAI_API_KEY"):

    print("❌ OPENAI_API_KEY is missing.")

    exit()


client = OpenAI()

response = client.responses.create(

    model="gpt-5.6-luna",

    input="Say hello to Voice2Action in one sentence."
)

print("\nCloud AI response:")
print(response.output_text)