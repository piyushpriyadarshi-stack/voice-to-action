import requests

url = "http://localhost:11434/api/chat"

data = {
    "model": "llama3.2",
    "messages": [
        {
            "role": "user",
            "content": "Explain artificial intelligence in one simple sentence."
        }
    ],
    "stream": False
}

try:
    response = requests.post(
        url,
        json=data,
        timeout=120
    )

    print("Status:", response.status_code)

    if response.status_code == 200:
        result = response.json()

        print("\n🤖 Ollama response:")
        print(result["message"]["content"])

    else:
        print("\n❌ Ollama error:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ Could not connect to Ollama.")
    print("Make sure Ollama is running.")

except Exception as e:
    print("\n❌ Error:")
    print(e)