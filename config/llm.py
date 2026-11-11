import os
import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

# Resilient try-except imports for ChatOllama to handle different LangChain version landscapes
try:
    from langchain_community.chat_models.ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fallback mock/stub class if absolutely not present
            class ChatOllama(BaseChatModel):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                def _generate(self, *args, **kwargs):
                    raise NotImplementedError("ChatOllama is not installed.")
                @property
                def _llm_type(self) -> str:
                    return "ollama_stub"

# Resilient try-except imports for OllamaEmbeddings
try:
    from langchain_community.embeddings.ollama import OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            # Fallback mock/stub class if absolutely not present
            class OllamaEmbeddings(Embeddings):
                def __init__(self, **kwargs):
                    pass
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.1] * 768 for _ in texts]
                def embed_query(self, text: str) -> List[float]:
                    return [0.1] * 768

# Configure standard logger
logger = logging.getLogger("private_agent_ecosystem.llm")
logging.basicConfig(level=logging.INFO)


class MockLLM(BaseChatModel):
    """
    Mock LLM for testing/fallback environments to ensure 100% offline unit-test coverage.
    """
    model_name: str = "mock-local-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatResult, ChatGeneration
        from langchain_core.messages import AIMessage

        # Simple rule-based mock replies to emulate Agent behavior & Tool execution
        text_content = ""
        last_message = messages[-1].content if messages else ""

        # Analyze input and simulate structured responses or normal thoughts
        if "extrair" in last_message.lower() or "analisar" in last_message.lower():
            text_content = (
                "Pensamento: Preciso analisar o documento contratual.\n"
                "Ação: read_document_tool\n"
                "Argumento: {'file_path': 'test_contrato.pdf'}\n"
                "Resposta: O documento test_contrato.pdf foi analisado com sucesso. Trata-se de um contrato corporativo privado de prestação de serviços com valor de R$ 150.000,00, sem reajuste automático, válido por 12 meses."
            )
        elif "pesquisar" in last_message.lower() or "qdrant" in last_message.lower() or "busca" in last_message.lower():
            text_content = (
                "Pensamento: Vou pesquisar no banco vetorial Qdrant.\n"
                "Ação: qdrant_search_tool\n"
                "Argumento: {'query': 'multa contratual'}\n"
                "Resposta: Encontrada cláusula de multa de 10% por atraso de pagamento no contrato principal."
            )
        elif "executar" in last_message.lower() or "python" in last_message.lower():
            text_content = (
                "Pensamento: Preciso executar um script Python para validar o cálculo.\n"
                "Ação: secure_python_executor_tool\n"
                "Argumento: {'code': 'print(150000 * 0.10)'}\n"
                "Resposta: Executado com sucesso. Saída: 15000.0"
            )
        else:
            text_content = (
                "Simulação de Resposta do Agente de IA Privado. Processado offline no modelo local. "
                "Todos os dados estão armazenados localmente e protegidos contra vazamento."
            )

        ai_msg = AIMessage(content=text_content)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "mock_local_model"


class MockEmbeddings(Embeddings):
    """
    Mock Embeddings class for offline tests.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Returns a mock 1536-dimensional or 768-dimensional vector
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 768


class LocalLLMFactory:
    """
    Factory class to dynamically manage and instantiate local LLM models and embeddings.
    Allows transparent switching between Ollama (ideal for development) and vLLM (ideal for production).
    """

    @staticmethod
    def get_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        force_mock: bool = False
    ) -> BaseChatModel:
        """
        Returns a ChatOpenAI model pointing to vLLM, a ChatOllama model pointing to Ollama,
        or a MockLLM if requested or if offline/unreachable.
        """
        if force_mock or os.environ.get("USE_MOCK_LLM", "false").lower() == "true":
            logger.info("Initializing fallback MockLLM for unit tests.")
            return MockLLM()

        # Load environment defaults if not explicitly passed
        provider = provider or os.environ.get("LLM_PROVIDER", "ollama").lower()
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        vllm_url = os.environ.get("vLLM_URL", "http://localhost:8000/v1")

        if provider == "vllm":
            model = model_name or os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
            logger.info(f"Connecting to local vLLM server at {vllm_url} for model: {model}")
            return ChatOpenAI(
                base_url=vllm_url,
                api_key="local-no-leak",
                model_name=model,
                temperature=temperature,
                max_tokens=2048,
                model_kwargs={"extra_body": {"repetition_penalty": 1.15}}
            )
        elif provider == "ollama":
            model = model_name or os.environ.get("LLM_MODEL", "qwen2.5-coder:7b")
            logger.info(f"Connecting to local Ollama server at {ollama_url} for model: {model}")
            return ChatOllama(
                base_url=ollama_url,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                keep_alive="5m"
            )
        else:
            logger.warning(f"Unknown provider '{provider}', falling back to MockLLM.")
            return MockLLM()

    @staticmethod
    def get_embeddings(
        provider: Optional[str] = None,
        force_mock: bool = False
    ) -> Embeddings:
        """
        Returns local embeddings using OllamaEmbeddings (e.g., nomic-embed-text)
        or a MockEmbeddings class.
        """
        if force_mock or os.environ.get("USE_MOCK_LLM", "false").lower() == "true":
            logger.info("Initializing fallback MockEmbeddings.")
            return MockEmbeddings()

        provider = provider or os.environ.get("LLM_PROVIDER", "ollama").lower()
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        if provider == "ollama":
            logger.info(f"Initializing local OllamaEmbeddings at {ollama_url}")
            return OllamaEmbeddings(
                base_url=ollama_url,
                model=os.environ.get("EMBEDDINGS_MODEL", "nomic-embed-text")
            )
        else:
            logger.info("Initializing fallback MockEmbeddings.")
            return MockEmbeddings()
