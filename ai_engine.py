import os
import json
import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

OPENAI_MODEL = "gpt-5.6-luna"


# =========================================================
# PROMPT
# =========================================================

def create_prompt(transcript):

    return f"""
You are Voice2Action, an AI meeting assistant.

Analyze the following meeting transcript.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "summary": "Short and accurate meeting summary",

    "key_points": [
        "Important point 1",
        "Important point 2"
    ],

    "action_items": [
        {{
            "task": "Specific task",
            "assigned_to": "Person or Not specified",
            "deadline": "Deadline or Not specified",
            "priority": "High, Medium, Low, or Not specified",
            "status": "Pending"
        }}
    ]
}}

Rules:

1. Do not invent information.
2. If a person is not mentioned, use "Not specified".
3. If a deadline is not mentioned, use "Not specified".
4. If priority is not mentioned, use "Not specified".
5. Every new task must have status "Pending".
6. Return JSON only.
"""


# =========================================================
# PARSE JSON
# =========================================================

def parse_response(text):

    try:

        return json.loads(text)

    except Exception:

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    text[start:end + 1]
                )

            except Exception:

                return None

    return None


# =========================================================
# OLLAMA
# =========================================================

def analyze_with_ollama(transcript):

    prompt = create_prompt(
        transcript
    )

    payload = {

        "model": OLLAMA_MODEL,

        "messages": [

            {
                "role": "user",
                "content": prompt
            }

        ],

        "stream": False
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        if response.status_code != 200:

            return {
                "success": False,
                "error": response.text,
                "provider": "Ollama"
            }

        data = response.json()

        text = data[
            "message"
        ][
            "content"
        ]

        result = parse_response(
            text
        )

        if result is None:

            return {
                "success": False,
                "error": "Invalid JSON from Ollama.",
                "provider": "Ollama"
            }

        return {
            "success": True,
            "data": result,
            "provider": "Ollama"
        }

    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "error": "Ollama is not running.",
            "provider": "Ollama"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "provider": "Ollama"
        }


# =========================================================
# OPENAI CLOUD
# =========================================================

def analyze_with_openai(transcript):

    if OpenAI is None:

        return {
            "success": False,
            "error": "OpenAI package is not installed.",
            "provider": "OpenAI"
        }

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        return {
            "success": False,
            "error": "OPENAI_API_KEY is not configured.",
            "provider": "OpenAI"
        }

    try:

        client = OpenAI(
            api_key=api_key
        )

        response = client.responses.create(

            model=OPENAI_MODEL,

            input=[
                {
                    "role": "user",
                    "content": create_prompt(
                        transcript
                    )
                }
            ]
        )

        text = response.output_text

        result = parse_response(
            text
        )

        if result is None:

            return {
                "success": False,
                "error": "Invalid JSON from OpenAI.",
                "provider": "OpenAI"
            }

        return {
            "success": True,
            "data": result,
            "provider": "OpenAI"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "provider": "OpenAI"
        }


# =========================================================
# HYBRID ROUTER
# =========================================================

def analyze_transcript(
    transcript,
    mode="Auto"
):

    # -----------------------------------------------------
    # LOCAL MODE
    # -----------------------------------------------------

    if mode == "Local":

        return analyze_with_ollama(
            transcript
        )


    # -----------------------------------------------------
    # CLOUD MODE
    # -----------------------------------------------------

    if mode == "Cloud":

        return analyze_with_openai(
            transcript
        )


    # -----------------------------------------------------
    # AUTO MODE
    # -----------------------------------------------------

    if mode == "Auto":

        cloud_result = analyze_with_openai(
            transcript
        )

        if cloud_result["success"]:

            return cloud_result

        local_result = analyze_with_ollama(
            transcript
        )

        if local_result["success"]:

            local_result["fallback"] = True

            local_result[
                "cloud_error"
            ] = cloud_result["error"]

            return local_result

        return {
            "success": False,

            "error":
                "Both Cloud AI and Local AI failed.",

            "cloud_error":
                cloud_result["error"],

            "local_error":
                local_result["error"]
        }