from faster_whisper import WhisperModel
import requests
from pathlib import Path
import json


# ==========================================
# SETTINGS
# ==========================================

AUDIO_FOLDER = Path("audio")
OUTPUT_FOLDER = Path("output")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"


# ==========================================
# 1. FIND AUDIO FILE
# ==========================================

def find_audio_file():

    supported_formats = [
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg"
    ]

    files = [
        file for file in AUDIO_FOLDER.iterdir()
        if file.suffix.lower() in supported_formats
    ]

    if not files:
        print("❌ No audio file found.")
        print("Put your audio file inside the audio folder.")
        return None

    print("\n🎵 Audio files found:")

    for i, file in enumerate(files):
        print(f"{i + 1}. {file.name}")

    return files[0]


# ==========================================
# 2. TRANSCRIBE AUDIO
# ==========================================

def transcribe_audio(audio_file):

    print("\n🎤 Loading Whisper model...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    print("Whisper loaded successfully.")

    print("\n🎤 Transcribing audio...")

    segments, info = model.transcribe(
        str(audio_file),
        beam_size=5
    )

    transcript = ""

    for segment in segments:
        transcript += segment.text.strip() + " "

    return transcript.strip()


# ==========================================
# 3. SEND TRANSCRIPT TO OLLAMA
# ==========================================

def analyze_with_ollama(transcript):

    print("\n🤖 Sending transcript to Ollama...")

    prompt = f"""
You are an AI meeting assistant.

Analyze the following meeting transcript.

TRANSCRIPT:
{transcript}

Provide the result in this exact structure:

SUMMARY:
Write a short summary of the meeting.

KEY POINTS:
- Point 1
- Point 2
- Point 3

ACTION ITEMS:
- Task:
  Assigned To:
  Deadline:
  Priority:

IMPORTANT:
Do not invent information.
If the person, deadline, or priority is not mentioned,
write "Not specified".
"""

    data = {
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
            json=data,
            timeout=300
        )

        if response.status_code != 200:

            print("\n❌ Ollama error:")
            print(response.text)

            return None

        result = response.json()

        return result["message"]["content"]

    except requests.exceptions.ConnectionError:

        print("\n❌ Cannot connect to Ollama.")
        print("Make sure Ollama is running.")

        return None


# ==========================================
# 4. SAVE RESULTS
# ==========================================

def save_results(transcript, analysis):

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    transcript_file = OUTPUT_FOLDER / "transcript.txt"
    analysis_file = OUTPUT_FOLDER / "analysis.txt"

    transcript_file.write_text(
        transcript,
        encoding="utf-8"
    )

    analysis_file.write_text(
        analysis,
        encoding="utf-8"
    )

    print("\n💾 Files saved:")

    print(f"📝 {transcript_file}")
    print(f"🤖 {analysis_file}")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=" * 60)
    print("        🎙️ VOICE2ACTION - STAGE 1")
    print("=" * 60)

    # Find audio

    audio_file = find_audio_file()

    if audio_file is None:
        return

    # Whisper

    transcript = transcribe_audio(audio_file)

    if not transcript:

        print("❌ No transcript generated.")
        return

    print("\n📝 TRANSCRIPT")
    print("=" * 60)

    print(transcript)

    print("=" * 60)

    # Ollama

    analysis = analyze_with_ollama(transcript)

    if analysis is None:
        return

    print("\n🤖 AI ANALYSIS")
    print("=" * 60)

    print(analysis)

    print("=" * 60)

    # Save

    save_results(
        transcript,
        analysis
    )

    print("\n✅ STAGE 1 COMPLETED!")


if __name__ == "__main__":
    main()