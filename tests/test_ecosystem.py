import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Force the LLM factory to use Mock mode for offline testing
os.environ["USE_MOCK_LLM"] = "true"
os.environ["QDRANT_URL"] = "http://localhost:6333"

# Ensure the correct import path by appending the repository root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient
from private_agent_ecosystem.config.llm import LocalLLMFactory, MockLLM, MockEmbeddings
from private_agent_ecosystem.tools.local_tools import LocalToolsSuite
from private_agent_ecosystem.graph import PrivateAgentEcosystemGraph, AgentState
from private_agent_ecosystem.api.server import app


class TestPrivateAgentEcosystem(unittest.TestCase):
    """
    Comprehensive test suite for the Private AI Agent Ecosystem.
    """

    def setUp(self):
        self.client = TestClient(app)
        self.tools = LocalToolsSuite()

    def test_llm_factory_mock_fallback(self):
        """
        Verifies that LocalLLMFactory returns MockLLM and MockEmbeddings
        when USE_MOCK_LLM is enabled or forced.
        """
        llm = LocalLLMFactory.get_llm(force_mock=True)
        self.assertIsInstance(llm, MockLLM)

        embeddings = LocalLLMFactory.get_embeddings(force_mock=True)
        self.assertIsInstance(embeddings, MockEmbeddings)

    def test_file_reader_tool(self):
        """
        Verifies the file reader tool under various file conditions.
        """
        # 1. Non-existent file
        result = self.tools.read_document_tool("non_existent_file.pdf")
        self.assertIn("Erro", result)

        # 2. Plain text / MD file
        test_file = "test_document.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Cláusula de Confidencialidade: Dados estritamente privados.")

        try:
            content = self.tools.read_document_tool(test_file)
            self.assertEqual(content, "Cláusula de Confidencialidade: Dados estritamente privados.")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_secure_python_executor_tool(self):
        """
        Verifies the secure executor tool outputs correctly and filters security risks.
        """
        # 1. Safe calculation
        safe_code = "print(150 * 1000)"
        res = self.tools.secure_python_executor_tool(safe_code)
        self.assertIn("150000", res)
        self.assertIn("Executado com sucesso", res)

        # 2. Block malicious script
        malicious_code = "import os; os.system('rm -rf /')"
        bad_res = self.tools.secure_python_executor_tool(malicious_code)
        self.assertIn("Erro de Segurança", bad_res)

    def test_qdrant_search_tool_fallback(self):
        """
        Verifies that when Qdrant is unreachable or offline, the tool
        fails gracefully to the keyword-matching in-memory fallback.
        """
        # Forced offline
        self.tools._qdrant_connected = False
        res = self.tools.qdrant_search_tool(query="regras de multa contratual")
        self.assertIn("Resultado Fallback", res)
        self.assertIn("multa de 10%", res)

    def test_graph_state_transitions(self):
        """
        Verifies state flow transitions sequentially:
        DocumentAnalyst -> InternalResearcher -> CodeSynthesisAgent -> COMPLETE
        """
        graph = PrivateAgentEcosystemGraph(force_mock=True)
        state = AgentState(user_query="Analisar reajuste do contrato")

        # Initial step
        self.assertEqual(state.current_agent, "DocumentAnalyst")

        # Step 1: DocumentAnalyst -> InternalResearcher
        state = graph.run_agent_step(state)
        self.assertEqual(state.current_agent, "InternalResearcher")
        self.assertIsNotNone(state.document_content)

        # Step 2: InternalResearcher -> CodeSynthesisAgent
        state = graph.run_agent_step(state)
        self.assertEqual(state.current_agent, "CodeSynthesisAgent")
        self.assertIsNotNone(state.research_results)

        # Step 3: CodeSynthesisAgent -> COMPLETE
        state = graph.run_agent_step(state)
        self.assertEqual(state.current_agent, "COMPLETE")
        self.assertIsNotNone(state.final_synthesis)

    def test_api_health_endpoint(self):
        """
        Verifies the FastAPI health monitor returns structured state.
        """
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("llm_provider", data)
        self.assertIn("qdrant_status", data)

    def test_api_chat_streaming_endpoint(self):
        """
        Verifies that the `/api/agent/chat` endpoint handles requests and
        streams Server-Sent Events (SSE) properly.
        """
        payload = {
            "query": "Qual o valor total?",
            "file_path": "mock_contrato.pdf",
            "provider": "ollama"
        }

        response = self.client.post("/api/agent/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])

        # Read streaming lines
        lines = []
        for line in response.iter_lines():
            if line:
                if isinstance(line, bytes):
                    lines.append(line.decode("utf-8"))
                else:
                    lines.append(line)

        self.assertTrue(any("event: start" in line for line in lines))
        self.assertTrue(any("event: agent_change" in line for line in lines))
        self.assertTrue(any("event: thought" in line for line in lines))
        self.assertTrue(any("event: tool" in line for line in lines))
        self.assertTrue(any("event: token" in line for line in lines))
        self.assertTrue(any("event: done" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
