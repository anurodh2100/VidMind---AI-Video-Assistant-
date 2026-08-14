
#mannual model selection
import whisper
import os
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import torch

load_dotenv()

# Sarvam sync STT API accepts short audio.
# We use 25-second pieces with a 5-second safety margin.
SARVAM_PIECE_SECONDS = 25


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v3"
)


_model = None


# ============================================================
# WHISPER MODEL
# ============================================================

def load_model():

    global _model

    if _model is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🖥️ Whisper device: {device}")

        if device == "cuda":
            print(
                f"🎮 GPU: {torch.cuda.get_device_name(0)}"
            )

        print(
            f"Loading Whisper model: {WHISPER_MODEL} ..."
        )

        _model = whisper.load_model(
            WHISPER_MODEL,
            device=device
        )

        print(
            "Whisper model loaded successfully ✅"
        )

    return _model


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_chunk_whisper(
    chunk_path: str
) -> str:

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# ============================================================
# SARVAM API
# ============================================================

def _send_to_sarvam(
    piece_path: str
) -> str:

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
            SARVAM_STT_URL,
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

def transcribe_chunk_sarvam(
    chunk_path: str
) -> str:

    """
    Send a chunk to Sarvam.

    The chunk is divided into 25-second pieces
    before sending to the Sarvam sync API.
    """

    if not SARVAM_API_KEY:

        raise RuntimeError(
            "SARVAM_API_KEY is not set "
            "in environment / .env"
        )

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
                f"{i + 1}/{total_pieces} ..."
            )

            text = _send_to_sarvam(
                piece_path
            )

            full_text += text + " "

        finally:

            if os.path.exists(piece_path):

                os.remove(
                    piece_path
                )

    return full_text.strip()


# ============================================================
# MANUAL ENGINE SELECTION
# ============================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Manually select transcription engine.

    english  → Whisper
    hinglish → Sarvam
    """

    if language.lower() == "hinglish":

        print(
            "🇮🇳 Using Sarvam AI..."
        )

        return transcribe_chunk_sarvam(
            chunk_path
        )

    print(
        "🎙️ Using Whisper..."
    )

    return transcribe_chunk_whisper(
        chunk_path
    )


# ============================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    """
    Transcribe all audio chunks.

    language="english"
        → Whisper

    language="hinglish"
        → Sarvam
    """

    full_transcript = ""

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(
        f"\n🔧 Manual transcription mode"
    )

    print(
        f"Engine selected: {engine}"
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"🎵 CHUNK "
            f"{i}/{len(chunks)}"
        )

        print(
            "=" * 60
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += (
            text + " "
        )

        print(
            f"✅ Chunk {i} completed"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "🎉 TRANSCRIPTION COMPLETED"
    )

    print(
        "=" * 60
    )

    return full_transcript.strip()
