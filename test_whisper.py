from faster_whisper import WhisperModel
from pathlib import Path

# Find audio files automatically
audio_folder = Path("audio")

audio_files = list(audio_folder.glob("Recording.mp3"))

if not audio_files:
    print("❌ No audio file found inside the audio folder.")
    print("Put your recording inside:")
    print(audio_folder.absolute())
    exit()

print("🎵 Audio file found:")
for i, file in enumerate(audio_files):
    print(f"{i + 1}. {file.name}")

# Use the first audio file
AUDIO_FILE = str(audio_files[0])

print(f"\nUsing: {AUDIO_FILE}")

print("\nLoading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded successfully.")

print("\n🎤 Transcribing audio...")

try:
    segments, info = model.transcribe(
        AUDIO_FILE,
        beam_size=5
    )

    print("\n📝 TRANSCRIPT")
    print("=" * 50)

    for segment in segments:
        print(segment.text.strip())

    print("=" * 50)
    print("\n✅ Transcription completed!")

except Exception as e:
    print("\n❌ Could not process the audio file.")
    print("Error:", e)