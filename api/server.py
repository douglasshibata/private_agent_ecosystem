import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from private_agent_ecosystem.graph import PrivateAgentEcosystemGraph, AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("private_agent_ecosystem.api")

app = FastAPI(
    title="Private AI Agent Ecosystem API",
    description="100% Private, Air-Gapped Multi-Agent Orchestration Engine",
    version="1.0.0"
)

# Enable CORS for local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., example="Qual o valor total do contrato e qual a multa por atraso de pagamento?")
    file_path: Optional[str] = Field(None, example="test_contrato.pdf", description="Caminho do arquivo local para análise")
    provider: Optional[str] = Field(None, example="ollama", description="Provedor local (ollama ou vllm)")


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    qdrant_status: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Dependency-aware local health monitor.
    Verifies connection status to local Qdrant and returns current active LLM configurations.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    # Check Qdrant status using a fast connection test
    qdrant_status = "unreachable"
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, timeout=1.0)
        client.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "offline-fallback"

    return {
        "status": "healthy",
        "llm_provider": provider,
        "qdrant_status": qdrant_status
    }


async def sse_event_generator(query: str, file_path: Optional[str], provider: Optional[str]):
    """
    Asynchronous event stream generator for Server-Sent Events (SSE).
    Streams agent states, internal thoughts, tool logs, and the final response token-by-token.
    """
    try:
        graph = PrivateAgentEcosystemGraph(provider=provider)
        state = AgentState(user_query=query, file_path=file_path)

        yield f"event: start\ndata: {json.dumps({'message': 'Iniciando fluxo cooperativo de Agentes de IA Privados...'})}\n\n"
        await asyncio.sleep(0.1)

        # Step 1: Document Analyst
        yield f"event: agent_change\ndata: {json.dumps({'agent': 'DocumentAnalyst', 'status': 'executing'})}\n\n"
        state = graph.run_agent_step(state)

        # Stream the Document Analyst's internal reasoning
        analyst_thought = state.agent_thoughts[-1]["thought"] if state.agent_thoughts else ""
        yield f"event: thought\ndata: {json.dumps({'agent': 'DocumentAnalyst', 'thought': analyst_thought})}\n\n"
        await asyncio.sleep(0.5)

        # Stream document parsing tool logs
        doc_chars = len(state.document_content) if state.document_content else 0
        yield f"event: tool\ndata: {json.dumps({'tool': 'read_document_tool', 'status': 'completed', 'chars_read': doc_chars})}\n\n"
        await asyncio.sleep(0.2)

        # Step 2: Internal Researcher
        yield f"event: agent_change\ndata: {json.dumps({'agent': 'InternalResearcher', 'status': 'executing'})}\n\n"
        state = graph.run_agent_step(state)

        researcher_thought = state.agent_thoughts[-1]["thought"] if len(state.agent_thoughts) > 1 else ""
        yield f"event: thought\ndata: {json.dumps({'agent': 'InternalResearcher', 'thought': researcher_thought})}\n\n"
        await asyncio.sleep(0.5)

        # Stream vector database search logs
        yield f"event: tool\ndata: {json.dumps({'tool': 'qdrant_search_tool', 'status': 'completed'})}\n\n"
        await asyncio.sleep(0.2)

        # Step 3: Code & Synthesis Agent
        yield f"event: agent_change\ndata: {json.dumps({'agent': 'CodeSynthesisAgent', 'status': 'executing'})}\n\n"
        state = graph.run_agent_step(state)

        synthesis_thought = state.agent_thoughts[-1]["thought"] if len(state.agent_thoughts) > 2 else ""
        yield f"event: thought\ndata: {json.dumps({'agent': 'CodeSynthesisAgent', 'thought': synthesis_thought})}\n\n"
        await asyncio.sleep(0.5)

        # Stream python sandbox execution logs
        yield f"event: tool\ndata: {json.dumps({'tool': 'secure_python_executor_tool', 'status': 'completed', 'result': state.code_results})}\n\n"
        await asyncio.sleep(0.2)

        # Step 4: Streaming Final Output (Token-by-Token response simulation)
        yield f"event: agent_change\ndata: {json.dumps({'agent': 'SynthesisEngine', 'status': 'streaming_final_output'})}\n\n"

        final_synthesis = state.final_synthesis or "Relatório Final Indisponível."

        # Split final text into chunks/words to simulate premium streaming experience
        words = final_synthesis.split(" ")
        for i, word in enumerate(words):
            # yield token
            token_payload = {"token": word + " ", "index": i}
            yield f"event: token\ndata: {json.dumps(token_payload)}\n\n"
            # small delay for visualization
            await asyncio.sleep(0.02)

        # Finished
        yield f"event: done\ndata: {json.dumps({'status': 'success', 'full_response': final_synthesis})}\n\n"

    except Exception as e:
        logger.error(f"Error in SSE streaming: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/agent/chat")
async def chat_agents_stream(request: ChatRequest):
    """
    Main entrypoint for local execution of cooperative Agents.
    Streams back complete real-time execution graphs (thoughts, tools, and token tokens) using Server-Sent Events (SSE).
    """
    logger.info(f"Received chat request: '{request.query}' with file: '{request.file_path}'")
    return StreamingResponse(
        sse_event_generator(request.query, request.file_path, request.provider),
        media_type="text/event-stream"
    )
