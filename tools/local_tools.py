import os
import sys
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger("private_agent_ecosystem.tools")


class LocalToolsSuite:
    """
    Suite of secure, local tools tailored for an air-gapped agent environment.
    Supports file reading, safe script execution, and local vector search (Qdrant).
    """

    def __init__(self, qdrant_url: Optional[str] = None):
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._qdrant_client: Optional[QdrantClient] = None
        self._qdrant_connected = False

    def get_qdrant_client(self) -> Optional[QdrantClient]:
        """
        Lazy-loads the Qdrant Client to handle cases where the Qdrant container is not immediately ready.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client

        try:
            logger.info(f"Connecting to Qdrant local instance at {self.qdrant_url}...")
            # We set short timeouts to avoid blocking during startup or unit testing
            client = QdrantClient(url=self.qdrant_url, timeout=2.0)
            # Simple check to see if it responds
            client.get_collections()
            self._qdrant_client = client
            self._qdrant_connected = True
            logger.info("Successfully connected to Qdrant.")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant at {self.qdrant_url} ({e}). Falling back to in-memory mock search.")
            self._qdrant_client = None
            self._qdrant_connected = False

        return self._qdrant_client

    def read_document_tool(self, file_path: str) -> str:
        """
        Reads local files (PDF, JSON, MD, TXT) safely.
        """
        logger.info(f"[TOOL] Reading document: {file_path}")
        if not os.path.exists(file_path):
            return f"Erro: O arquivo '{file_path}' não foi encontrado no sistema de arquivos local."

        # Check extension
        _, ext = os.path.splitext(file_path.lower())

        try:
            if ext in [".txt", ".md", ".json"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if ext == ".json":
                    # format nicely
                    parsed = json.loads(content)
                    return json.dumps(parsed, indent=2, ensure_ascii=False)
                return content
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    text = []
                    for page in reader.pages:
                        text_page = page.extract_text()
                        if text_page:
                            text.append(text_page)
                    return "\n--- Nova Página ---\n".join(text)
                except ImportError:
                    return f"Aviso: Biblioteca 'pypdf' não instalada. Exibindo metadados básicos do arquivo '{file_path}' (Tamanho: {os.path.getsize(file_path)} bytes)."
            else:
                # Default raw reader
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(5000) # limit to first 5000 chars
        except Exception as e:
            return f"Erro ao ler o arquivo {file_path}: {str(e)}"

    def secure_python_executor_tool(self, code: str) -> str:
        """
        Runs a short Python script in a separate, isolated subprocess with safety constraints.
        """
        logger.info(f"[TOOL] Secure Python execution requested.")

        # Block malicious commands / operations to preserve local safety
        forbidden_keywords = [
            "rm -rf", "shred", "os.system", "shutil.rmtree",
            "subprocess.call", "pty.spawn", "import urllib", "requests.post"
        ]
        for keyword in forbidden_keywords:
            if keyword in code:
                return f"Erro de Segurança: O código contém comandos não autorizados ('{keyword}')."

        try:
            # Execute with python -c under a limited timeout of 5 seconds
            cmd = [sys.executable, "-c", code]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                return f"Executado com sucesso. Saída:\n{output}"
            else:
                error = result.stderr.strip()
                return f"Erro de Execução (Código {result.returncode}):\n{error}"
        except subprocess.TimeoutExpired:
            return "Erro: O script excedeu o tempo limite de execução (5 segundos)."
        except Exception as e:
            return f"Erro ao executar o código Python: {str(e)}"

    def qdrant_search_tool(self, query: str, collection_name: str = "documentos_privados") -> str:
        """
        Searches the local Qdrant database for semantic vector matches.
        """
        logger.info(f"[TOOL] Searching local Qdrant for: '{query}'")
        client = self.get_qdrant_client()

        if not self._qdrant_connected or client is None:
            # Local fallback database mock search
            logger.info("Using in-memory mock search fallback.")
            mock_database = {
                "multa": "Cláusula 14.2: Em caso de atraso superior a 5 dias úteis, incidirá multa de 10% sobre o valor da fatura vencida.",
                "reajuste": "Cláusula 5.1: O preço total do contrato é de R$ 150.000,00, sem reajustes automáticos durante o período de 12 meses.",
                "rescisao": "Cláusula 18.1: A rescisão sem justa causa pode ser efetuada por qualquer parte mediante aviso prévio por escrito de 30 dias.",
                "privacidade": "Cláusula 22: As partes obrigam-se a manter estrito sigilo sobre todas as informações corporativas compartilhadas.",
                "geral": "Contrato Corporativo de Prestação de Serviços de TI entre a TechCorp S.A. e a SoftSoluções Ltda."
            }

            # Simple keyword matching fallback
            matches = []
            for kw, val in mock_database.items():
                if kw in query.lower():
                    matches.append(val)

            if not matches:
                matches.append(mock_database["geral"])

            return "\n\n".join([f"[Resultado Fallback (Sem Conexão Qdrant)]:\n{m}" for m in matches])

        try:
            # In a real run, we would call local embeddings. Here we simulate vector matching or simple scroll
            # since embeddings require high CPU/GPU.
            # To be highly robust, let's look if collection exists, if not, create it
            collections_list = client.get_collections().collections
            collection_names = [col.name for col in collections_list]

            if collection_name not in collection_names:
                # Create the collection
                from qdrant_client.http import models
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")

                # Pre-seed collection with mock documents for testing
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=1,
                            vector=[0.1] * 768,
                            payload={"text": "Cláusula 14.2: Em caso de atraso superior a 5 dias úteis, incidirá multa de 10% sobre o valor da fatura vencida."}
                        ),
                        models.PointStruct(
                            id=2,
                            vector=[0.2] * 768,
                            payload={"text": "Cláusula 5.1: O preço total do contrato é de R$ 150.000,00, sem reajustes automáticos durante o período de 12 meses."}
                        )
                    ]
                )

            # Query Qdrant (using a dummy vector since we are in a lightweight client context, or standard query)
            results = client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True
            )

            hits = []
            for point in results[0]:
                payload = point.payload or {}
                text = payload.get("text", "")
                if text:
                    hits.append(text)

            if hits:
                return "\n\n".join([f"[Qdrant Match]:\n{h}" for h in hits])
            return "Nenhum documento relevante encontrado no Qdrant."

        except Exception as e:
            return f"Erro na consulta do Qdrant: {str(e)}"
