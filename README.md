# LiveKit RAG Voice Agent with Qdrant

## Changes in this Fork

This repository is a forked version of the original Cartesia Voice Agent example. The main enhancements include the integration of Retrieval-Augmented Generation (RAG) using OpenAI embeddings and Qdrant vector database for semantic search capability. Specifically:

- **Document Ingestion:** Added `injest.py` script that processes text files, generates embeddings with OpenAI's `text-embedding-3-small` model, and populates the Qdrant vector database (`knowledge_base`).
- **Custom Retrieval Function:** Implemented `retrieve_info` in `main.py` to enable the agent to query the vector database for exact and semantic matches, returning concise, context-aware responses.
- **Enhanced Logging:** Improved logging throughout ingestion and retrieval processes to assist debugging and ensure visibility into internal operations.
- **Async Handling:** Ensured all blocking I/O operations (e.g., Qdrant queries, OpenAI embedding requests) run in separate threads (`asyncio.to_thread`) to maintain asynchronous efficiency.

---

## Original README

This is a demo of a LiveKit [Voice Pipeline Agent](https://docs.livekit.io/agents/voice-agent/voice-pipeline/) using [Cartesia](https://www.cartesia.ai/) and GPT-4o-mini.

The example includes a custom Next.js frontend and Python agent.

## Live Demo

[Live Demo](https://cartesia-assistant.vercel.app/)

![Screenshot of the Cartesia Voice Agent Example](.github/screenshot.png)

## Running the example

### Prerequisites

- Node.js
- Python 3.9-3.12
- LiveKit Cloud account (or OSS LiveKit server)
- Cartesia API key (for speech synthesis)
- OpenAI API key (for LLM)
- Deepgram API key (for speech-to-text)

### Frontend

Copy `.env.example` to `.env.local` and set the environment variables. Then run:

```bash
cd frontend
npm install
npm run dev
```

### Agent

Copy `.env.example` to `.env` and set the environment variables. Then run:

```bash
cd agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# This will ingest the documents into the Qdrant Vector Database (Only to Run Once)
python injest.py

python main.py dev
```
