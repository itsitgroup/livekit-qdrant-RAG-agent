import asyncio
import json
import os
import requests
import openai as openai_client
from typing import List, Any, Annotated
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, llm
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.log import logger
from livekit.plugins import deepgram, silero, cartesia, openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

load_dotenv()


class LoggingChatContext(ChatContext):
    def add_message(self, message: ChatMessage):
        super().add_message(message)
        if message.role == "user":
            logger.info(f"STT Transcription: {message.content}")
        elif message.role == "assistant":
            logger.info(f"Model Response: {message.content}")


class MyAgentFunctions(llm.FunctionContext):
    def __init__(self):
        super().__init__()
        self.client = QdrantClient(
            url=os.getenv("QDRANT_ENDPOINT"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = "knowledge_base"
        self.thinking_messages = [
            "Looking that up for you...",
            "One moment while I verify...",
            "Checking the documentation...",
        ]
        openai_client.api_key = os.getenv("OPENAI_API_KEY")

    def _get_query_embedding(self, text: str) -> List[float]:
        """Compute the embedding for the given text using the same model as ingestion."""
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    @llm.ai_callable()
    async def retrieve_info(
        self,
        query: Annotated[str, llm.TypeInfo(description="The user's query to search in knowledge base")]
    ) -> str:
        """Retrieve relevant information from Qdrant vector database for the given query."""
        try:
            logger.info(f"retrieve_info called with query: {query}")

            # Step 1: Try an exact text match first
            points, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="text", match=MatchText(text=query))]
                ),
                limit=1,
            )
            logger.info(f"Exact match returned {len(points)} points")
            if points and len(points) > 0:
                exact_text = points[0].payload.get("text", "").strip()
                if exact_text:
                    logger.info("Exact match found, returning result.")
                    return exact_text

            # Step 2: Compute embedding for the query
            query_embedding = await asyncio.to_thread(self._get_query_embedding, query)

            # Step 3: Perform a semantic search using the query embedding
            semantic_results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=3,  # Retrieve top 3 most relevant results
            )
            logger.info(f"Semantic search returned {len(semantic_results)} points")
            if not semantic_results or len(semantic_results) == 0:
                return "I couldn't find relevant information in our knowledge base."

            # Step 4: Combine retrieved results into a concise response
            retrieved_texts = [
                r.payload.get("text", "").strip() 
                for r in semantic_results 
                if r.payload.get("text")
            ]
            if not retrieved_texts:
                return "I couldn't find relevant information in our knowledge base."

            combined_response = "\n".join(retrieved_texts)
            truncated_response = combined_response[:1000]  # Limit response length
            logger.info(f"Returning combined response: {truncated_response}")
            return f"Here's what I found:\n{truncated_response}"

        except Exception as e:
            logger.error(f"Error in retrieve_info: {e}")
            return f"Error retrieving information: {str(e)}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["qdrant"] = MyAgentFunctions()
    headers = {
        "X-API-Key": os.getenv("CARTESIA_API_KEY", ""),
        "Cartesia-Version": "2024-08-01",
        "Content-Type": "application/json",
    }
    response = requests.get("https://api.cartesia.ai/voices", headers=headers)
    if response.status_code == 200:
        proc.userdata["cartesia_voices"] = response.json()
    else:
        logger.warning(f"Failed to fetch Cartesia voices: {response.status_code}")


async def entrypoint(ctx: JobContext):
    # retriever = ctx.proc.userdata["qdrant"]
    fnc_ctx = MyAgentFunctions()

    system_prompt = """You are a voice assistant created by LiveKit. Use these rules:
                    1. Respond conversationally using natural speech patterns
                    2. If you need to look up information, say you're checking
                    3. When using retrieved information, mention the source
                    4. Keep responses under 3 sentences"""
    initial_ctx = LoggingChatContext(
        messages=[
            ChatMessage(role="system", content=system_prompt)
        ]
    )
    cartesia_voices: List[dict[str, Any]] = ctx.proc.userdata["cartesia_voices"]

    tts = cartesia.TTS(
        model="sonic-2",
    )
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=tts,
        fnc_ctx=fnc_ctx,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=("You are a voice assistant. Answer questions using the knowledge base when appropriate. "
                  "If you don't know an answer about Its IT Group, you can call the retrieve_info function to search for it. "
                  "Always try to to keep the answers concise and under 3 sentences. "
                  "If any Question comes regarding Its IT Group, search the knowledge base.")
        )
    )

    is_user_speaking = False
    is_agent_speaking = False

    @ctx.room.on("participant_attributes_changed")
    def on_participant_attributes_changed(
        changed_attributes: dict[str, str], participant: rtc.Participant
    ):
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD:
            return

        if "voice" in changed_attributes:
            voice_id = participant.attributes.get("voice")
            logger.info(
                f"participant {participant.identity} requested voice change: {voice_id}"
            )
            if not voice_id:
                return

            voice_data = next(
                (voice for voice in cartesia_voices if voice["id"] == voice_id), None
            )
            if not voice_data:
                logger.warning(f"Voice {voice_id} not found")
                return
            if "embedding" in voice_data:
                language = "en"
                if "language" in voice_data and voice_data["language"] != "en":
                    language = voice_data["language"]
                tts._opts.voice = voice_data["embedding"]
                tts._opts.language = language
                if not (is_agent_speaking or is_user_speaking):
                    asyncio.create_task(
                        agent.say("How do I sound now?", allow_interruptions=True)
                    )

    await ctx.connect()

    @agent.on("agent_started_speaking")
    def agent_started_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = True

    @agent.on("agent_stopped_speaking")
    def agent_stopped_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = False

    @agent.on("user_started_speaking")
    def user_started_speaking():
        nonlocal is_user_speaking
        is_user_speaking = True

    @agent.on("user_stopped_speaking")
    def user_stopped_speaking():
        nonlocal is_user_speaking
        is_user_speaking = False

    voices = []
    for voice in cartesia_voices:
        voices.append(
            {
                "id": voice["id"],
                "name": voice["name"],
            }
        )
    voices.sort(key=lambda x: x["name"])
    await ctx.room.local_participant.set_attributes({"voices": json.dumps(voices)})

    agent.start(ctx.room)
    await agent.say("Hi there! I'm your LiveKit assistant. Ask me anything about our platform!", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
