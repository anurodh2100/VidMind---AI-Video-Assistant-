import os
import time
import random
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ============================================================
# RETRY CONFIG
# ============================================================
MAX_RETRIES = 4
BASE_DELAY_SECONDS = 5   # first retry waits ~5s, then ~10s, ~20s, ~40s (exponential)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _build_ydl_opts(output_path: str) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": True,

        # A realistic browser User-Agent makes requests look less bot-like,
        # which helps avoid YouTube's 403 anti-bot responses.
        "http_headers": {
            "User-Agent": USER_AGENT,
        },

        # Retries yt-dlp itself does internally for network-level hiccups
        # (separate from our own retry loop below, which retries the whole
        # download in case yt-dlp gives up entirely).
        "retries": 3,
        "fragment_retries": 3,
    }

    # OPTIONAL: if 403s persist even with the User-Agent fix, YouTube is likely
    # rate-limiting/anon-blocking your IP. Using cookies from a browser where
    # you're logged into YouTube makes requests look authenticated instead of
    # anonymous, which resolves most persistent 403s. Uncomment ONE of these
    # (pick whichever browser you're logged into YouTube with):
    #
    # opts["cookiesfrombrowser"] = ("chrome",)
    # opts["cookiesfrombrowser"] = ("firefox",)
    # opts["cookiesfrombrowser"] = ("edge",)

    return opts


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL with retry-with-backoff.

    YouTube's anti-bot layer intermittently returns HTTP 403, especially
    right after repeated requests to the same IP (e.g. while testing).
    A single fresh attempt often succeeds again after a short cooldown,
    so we retry a few times with increasing delay before giving up.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = _build_ydl_opts(output_path)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🎬 Download attempt {attempt}/{MAX_RETRIES}...")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                filename = os.path.splitext(filename)[0] + ".wav"

                print("✅ Download succeeded.")
                return filename

        except yt_dlp.utils.DownloadError as e:
            last_error = e
            error_text = str(e)

            is_403 = "403" in error_text or "Forbidden" in error_text

            if attempt == MAX_RETRIES:
                print(f"❌ All {MAX_RETRIES} attempts failed.")
                break

            # Exponential backoff with a little random jitter so we don't
            # hammer YouTube at exact fixed intervals.
            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            delay += random.uniform(0, 2)

            reason = "HTTP 403 (likely rate-limited/bot-detection)" if is_403 else "download error"
            print(
                f"⚠️  Attempt {attempt} failed ({reason}). "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    # All retries exhausted — raise a clearer error for the pipeline/UI to show.
    raise RuntimeError(
        f"Failed to download YouTube audio after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}\n\n"
        "This is usually YouTube's anti-bot rate limiting (HTTP 403) after "
        "repeated requests. Try:\n"
        "  1. Running `pip install -U yt-dlp` (YouTube changes often break older versions)\n"
        "  2. Waiting a few minutes before retrying\n"
        "  3. Enabling cookiesfrombrowser in audio_processor.py (see comments in _build_ydl_opts)"
    )


def convert_to_wav(input_path: str) -> str:
    """Covert any audi/video file to WAV format usug pydub."""
    output_path = os.path.splitext(input_path)[0] + "_coverted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16hz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks