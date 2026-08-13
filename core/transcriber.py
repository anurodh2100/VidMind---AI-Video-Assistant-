import whisper
import os
import requests

from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

# Sarvam sync API: keep pieces below 30 seconds
SARVAM_PIECE_SECONDS = 25


_model = None


# ============================================================
# WHISPER MODEL
# ============================================================

def load_model():

    global _model

    if _model is None:

        print(f"Loading Whisper model: {WHISPER_MODEL} ...")

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded successfully ✅")

    return _model


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(chunk_path: str) -> str:

    model = load_model()

    print("🌐 Detecting language...")

    # Load audio
    audio = whisper.load_audio(chunk_path)

    # Pad / trim to Whisper's expected duration
    audio = whisper.pad_or_trim(audio)

    # Convert audio to Mel spectrogram
    mel = whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels
    ).to(model.device)

    # Detect language
    _, probabilities = model.detect_language(mel)

    detected_language = max(
        probabilities,
        key=probabilities.get
    )

    confidence = probabilities[detected_language]

    print(
        f"🌐 Detected language: "
        f"{detected_language} "
        f"({confidence:.2%})"
    )

    return detected_language


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()

    print("🎙️ Using Whisper...")

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# ============================================================
# SARVAM API REQUEST
# ============================================================

def _send_to_sarvam(piece_path: str) -> str:

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "mode": "translate"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

    if not response.ok:

        print(
            f"\n❌ Sarvam returned "
            f"{response.status_code}"
        )

        print(
            f"Response body: "
            f"{response.text}\n"
        )

        response.raise_for_status()

    return response.json().get(
        "transcript",
        ""
    )


# ============================================================
# SARVAM TRANSCRIPTION
# ============================================================

def transcribe_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:

        raise RuntimeError(
            "SARVAM_API_KEY is not set "
            "in environment / .env"
        )

    print("🇮🇳 Using Sarvam AI...")

    audio = AudioSegment.from_wav(
        chunk_path
    )

    piece_ms = (
        SARVAM_PIECE_SECONDS * 1000
    )

    full_text = ""

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[
            start:start + piece_ms
        ]

        piece_path = (
            f"{chunk_path}_sv_{i}.wav"
        )

        piece.export(
            piece_path,
            format="wav"
        )

        try:

            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces}..."
            )

            text = _send_to_sarvam(
                piece_path
            )

            full_text += text + " "

        finally:

            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# ============================================================
# AUTOMATIC ENGINE SELECTION
# ============================================================

def transcribe_chunk(chunk_path: str) -> str:

    detected_language = detect_language(
        chunk_path
    )

    print(
        f"🔎 Language detected: "
        f"{detected_language}"
    )

    # English → Whisper
    if detected_language == "en":

        print(
            "➡️ Selecting Whisper "
            "for English audio."
        )

        return transcribe_chunk_whisper(
            chunk_path
        )

    # Hindi / other Indic language → Sarvam
    else:

        print(
            "➡️ Selecting Sarvam AI "
            "for non-English audio."
        )

        return transcribe_chunk_sarvam(
            chunk_path
        )


# ============================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================

def transcribe_all(chunks: list) -> str:

    full_transcript = []

    print(
        "\n🔀 Automatic transcription "
        "engine selection enabled."
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            f"🎵 CHUNK {i}/{len(chunks)}"
        )

        print("=" * 60)

        text = transcribe_chunk(
            chunk
        )

        full_transcript.append(text)

        print(
            f"✅ Chunk {i} completed"
        )

    print("\n" + "=" * 60)
    print("🎉 TRANSCRIPTION COMPLETED")
    print("=" * 60)

    return " ".join(
        full_transcript
    ).strip()