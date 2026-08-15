# 🎬 VidMind — AI Video Assistant

VidMind turns any YouTube video or local recording into a transcript, summary, action items, key decisions, open questions — and a chat interface to ask the video anything, powered by RAG.

---

## ✨ Features

- 📥 **Input** — paste a YouTube URL or upload a local audio/video file
- 🎙️ **Transcription** — Whisper (English) or Sarvam AI (Hinglish)
- 🏷️ **Auto title generation**
- 📋 **Meeting-style summary** (map-reduce over long transcripts)
- ✅ **Action items** extraction (task, owner, deadline)
- 🔑 **Key decisions** extraction
- ❓ **Open questions** extraction
- 💬 **Chat with your video** via a RAG pipeline (Chroma + HuggingFace embeddings)
- ⚡ Live step-by-step progress in the Streamlit UI
- 🌓 Light / Dark theme toggle

---

## 🗂️ Project Structure

```
AI_Video_Assistant/
│
├── .env                     # API keys (not committed)
├── main.py                  # CLI entry point (Phase 1: analyze, Phase 2: chat)
├── app.py                   # Streamlit UI
│
├── utils/
│   └── audio_processor.py   # YouTube download / local file conversion / chunking
│
└── core/
    ├── transcriber.py       # Whisper + Sarvam transcription
    ├── summarizer.py        # Title + summary generation
    ├── extractor.py         # Action items / decisions / questions extraction
    ├── rag_engine.py        # RAG chain for chatting with the transcript
    └── vector_store.py      # Chroma vector store + retriever
```

---

## ⚙️ Setup

### 1. Navigate to the project folder

```bash
cd AI_Video_Assistant
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit openai-whisper torch pydub yt-dlp requests python-dotenv \
    langchain langchain-mistralai langchain-text-splitters langchain-chroma \
    langchain-community
```

> `pydub` requires **ffmpeg** installed on your system and available on PATH — audio conversion/chunking will fail without it.

Keep `yt-dlp` current — YouTube changes frequently break older versions:
```bash
pip install -U yt-dlp
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
MISTRAL_API_KEY=your_mistral_key_here
SARVAM_API_KEY=your_sarvam_key_here

# Optional overrides
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v3
```

> ⚠️ Never commit your real `.env` file — make sure it's listed in `.gitignore`.

- **MISTRAL_API_KEY** — required, used by `ChatMistralAI` (`mistral-small-latest`) for summarization, extraction, and RAG chat
- **SARVAM_API_KEY** — required only if you use `language="hinglish"` (Sarvam AI speech-to-text)
- **WHISPER_MODEL** — Whisper model size for English transcription (default `small`); larger models are more accurate but slower and need more RAM/VRAM
- No OpenAI key is required by this pipeline (Whisper here runs locally via `openai-whisper`, not the OpenAI API)

---

## ▶️ Running the App (Streamlit UI)

```bash
streamlit run app.py
```

If `streamlit` isn't recognized even with the venv active:
```bash
python -m streamlit run app.py
```

Then in the browser:
1. Paste a YouTube URL or upload a file
2. Pick a language (`english` or `hinglish`)
3. Click **Analyze Video** and watch the pipeline steps complete live
4. Browse Summary / Action Items / Key Decisions / Open Questions / Transcript tabs
5. Use the **Chat** tab to ask questions about the video

---

## 🖥️ Running via CLI (without Streamlit)

```bash
python main.py
```

You'll be prompted for a URL/file path and language. Phase 1 prints the full analysis; Phase 2 drops you into an interactive chat loop (`exit`/`quit`/`q` to leave).

---

## 🧩 How It Works

1. **`process_input()`** detects whether the source is a YouTube URL or local file:
   - YouTube → downloaded via `yt-dlp`, converted to WAV
   - Local file → converted to mono 16kHz WAV via `pydub`
   - Either way, audio is chunked into 10-minute segments
2. **`transcribe_all()`** transcribes each chunk — Whisper for English, Sarvam AI (in 25-second pieces) for Hinglish
3. **`generate_title()`** and **`summarize()`** run map-reduce summarization over the transcript via Mistral
4. **`extract_action_items()` / `extract_key_decisions()` / `extract_questions()`** each run a dedicated Mistral prompt chain
5. **`build_rag_chain()`** embeds the transcript into a Chroma vector store (`all-MiniLM-L6-v2`) and builds a retrieval-augmented chain for Q&A
6. **`ask_question()`** answers using only the retrieved transcript context — it explicitly says so when the answer isn't in the transcript

---

## 🐞 Known Issues & Fixes

- **`HTTP Error 403: Forbidden` on YouTube download** — YouTube's anti-bot rate limiting, usually triggered by repeated download attempts in a short window. `audio_processor.py` now retries with exponential backoff (4 attempts) and sends a realistic browser User-Agent. If 403s persist:
  1. `pip install -U yt-dlp`
  2. Wait a few minutes before retrying
  3. Enable `cookiesfrombrowser` in `_build_ydl_opts()` (commented out by default) using a browser where you're logged into YouTube
- Folder naming: watch out for `utils/audio_processor.py` vs. any older/typo'd copy (e.g. `utilis/audio_processsor.py`) — make sure imports resolve to the intended file.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [OpenAI Whisper](https://github.com/openai/whisper) — local English transcription
- [Sarvam AI](https://www.sarvam.ai/) — Hinglish speech-to-text
- [LangChain](https://www.langchain.com/) — chains + RAG orchestration
- [Mistral AI](https://mistral.ai/) (`mistral-small-latest`) — summarization, extraction, RAG chat
- [Chroma](https://www.trychroma.com/) — vector store for RAG
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio download
- [pydub](https://github.com/jiaaro/pydub) — audio conversion/chunking (requires ffmpeg)

---

## 👤 Credits

Built by **AJ Developers**.
