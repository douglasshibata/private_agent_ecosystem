# 🛡️ Ecossistema Corporativo de Agentes de IA 100% Privado
> **Arquitetura Air-Gapped, Orquestração Multi-Agente Cooperativa e Execução Local de Alta Performance**

Este projeto implementa a arquitetura completa e a estrutura de um ecossistema corporativo de Agentes de IA estritamente privado, rodando localmente sem qualquer dependência de APIs externas (zero vazamento de dados), baseado nas diretrizes de alta performance do **Agent Starter Kit**.

---

## 🎯 1. Arquitetura de Inferência Local e Fluxo Multi-Agente

O sistema orquestra múltiplos agentes especializados em um fluxo de grafo sequencial cooperativo, onde as saídas e pensamentos de um agente servem de insumo para o próximo, culminando na geração de relatórios consolidados em tempo real com streaming via Server-Sent Events (SSE).

### 📊 Diagrama ASCII da Arquitetura Isolada (Air-Gapped)

```text
       ┌─────────────────────────────────────────────────────────────┐
       │                 AMBIENTE PRIVADO / AIR-GAPPED               │
       │                                                             │
       │   ┌──────────────┐             ┌─────────────────────────┐  │
       │   │  User Input  │             │   Arize Phoenix /       │  │
       │   └──────┬───────┘             │   Langfuse Observability│  │
       │          │ (HTTP/JSON)         └───────────▲─────────────┘  │
       │          ▼                                 │ (OTel Traces)  │
       │   ┌──────────────┐      Tool Call      ┌───┴─────────────┐  │
       │   │ FastAPI Core │ ──────────────────> │   Local Tools   │  │
       │   │   Backend    │ <────────────────── │  (Python/File)  │  │
       │   └──────┬───────┘      Tool Output    └─────────────────┘  │
       │          │                                                  │
       │          ▼                                                  │
       │   ┌──────────────┐      Embedding      ┌─────────────────┐  │
       │   │    Agent     │ ──────────────────> │  Local Qdrant   │  │
       │   │ Orchestrator │ <────────────────── │  Vector Database│  │
       │   │ (Graph Flow) │   Query Retrieval   └─────────────────┘  │
       │   └──────┬───────┘                                          │
       │          │                                                  │
       │          │ (OpenAI-Compatible Protocol)                     │
       │          ▼                                                  │
       │   ┌──────────────────────────────────────────────────────┐  │
       │   │           LOCAL INFERENCE ENGINE SELECTOR            │  │
       │   │                                                      │  │
       │   │   [OLLAMA (Dev / Prototipagem)]                      │  │
       │   │   Port: 11434 (llama3.1:8b / qwen2.5-coder:7b)       │  │
       │   │                         OR                           │  │
       │   │   [vLLM (Produção de Alta Performance)]              │  │
       │   │   Port: 8000 (PagedAttention, AWQ, GPTQ)             │  │
       │   └──────────────────────────────────────────────────────┘  │
       │          │                                                  │
       │          ▼ (SSE Stream / Token-by-Token)                    │
       │   ┌──────────────┐                                          │
       │   │ Output Stream│                                          │
       │   └─────────────────────────────────────────────────────────┘
```

### 🔄 Alternância Transparente entre Ollama e vLLM

A arquitetura de backend aproveita o protocolo padronizado da **OpenAI-Compatible API**, permitindo que a aplicação mude de motor de inferência local apenas alterando duas variáveis de ambiente:

1. **Ollama (Ideal para Desenvolvimento Local):**
   - **Endpoint:** `http://localhost:11434/v1`
   - **Vantagem:** Inicialização instantânea, gerenciamento simplificado de modelos, ideal para máquinas locais ou laptops com ou sem GPU de nível empresarial.
2. **vLLM (Ideal para Produção e Cluster Corporativo):**
   - **Endpoint:** `http://localhost:8000/v1`
   - **Vantagem:** Altíssimo throughput impulsionado pelo algoritmo de **PagedAttention**, suporte a lote contínuo (continuous batching), excelente para clusters de produção servindo dezenas de usuários em paralelo com latência mínima.

---

## 📦 2. Estrutura de Diretórios do Ecossistema

O projeto está estruturado de forma estritamente modular e fortemente tipada:

```text
private_agent_ecosystem/
├── api/
│   ├── __init__.py
│   └── server.py             # Servidor FastAPI com endpoints de SSE streaming e /health
├── config/
│   ├── __init__.py
│   └── llm.py                # Abstração de provedor de LLM local (Ollama vs vLLM) e mocks
├── tools/
│   ├── __init__.py
│   └── local_tools.py        # Ferramentas locais (Leitura de PDF/JSON, Execução Python, Qdrant)
├── tests/
│   ├── __init__.py
│   └── test_ecosystem.py     # Suite completa de testes unitários e de integração
├── README.md                 # Guia de arquitetura, hardware e operação (Este Arquivo)
├── Dockerfile                # Configuração de build Docker para o backend Python
├── docker-compose.yml        # Orquestração local do Qdrant, Ollama, vLLM e Phoenix
├── download_models.sh        # Script automatizado de bootstrap e carregamento dos modelos
└── requirements.txt          # Dependências do ecossistema local
```

---

## 🛠️ 3. Configuração de Infraestrutura e Execução (`docker-compose.yml`)

### Serviços Disponibilizados

O arquivo `docker-compose.yml` unifica toda a infraestrutura offline necessária:
- **`private_qdrant`:** Armazenamento vetorial rápido com volumes persistentes.
- **`private_ollama` / `private_vllm`:** Motores de inferência com pass-through de GPU NVIDIA habilitado por padrão.
- **`private_phoenix`:** Tracing e observabilidade local com visualização detalhada de spans e execuções de ferramentas offline via Arize Phoenix.
- **`backend`:** Aplicação FastAPI compilada localmente e integrada aos demais componentes.

### 🚀 Executando o Ambiente Local

1. **Garantir os Requisitos Básicos:**
   - Docker e Docker Compose instalados.
   - NVIDIA Container Toolkit configurado (para aceleração por GPU de hardware local).

2. **Iniciar o Docker Compose:**
   ```bash
   docker-compose up -d --build
   ```

3. **Bootstrap Automatizado de Modelos (Ollama):**
   Execute o script para baixar sequencialmente os modelos de alta performance recomendados:
   ```bash
   ./download_models.sh
   ```

---

## 🧬 4. Fluxo de Trabalho dos Agentes (Cooperative Graph)

O orquestrador em `graph.py` coordena um fluxo de estado em 4 etapas principais:

1. **Agente Analista de Documentos (`DocumentAnalyst`):**
   - Lê os arquivos no sistema de arquivos local usando `read_document_tool`.
   - Extrai as cláusulas contratuais essenciais e fatos brutos.
2. **Agente Pesquisador Interno (`InternalResearcher`):**
   - Utiliza a consulta vetorial `qdrant_search_tool` para cruzar o documento atual com regras de conformidade e históricos salvos localmente.
3. **Agente de Código e Síntese (`CodeSynthesisAgent`):**
   - Executa scripts de cálculo matemático (ex: multas, taxas) utilizando a ferramenta altamente segura `secure_python_executor_tool` (com bloqueios de comandos de IO e sanitização de strings).
   - Une todos os achados em um relatório final formatado em Markdown profissional.

### 🛠️ Function/Tool Calling Confiável em Modelos Abertos

Modelos open-source locais de menor escala (como o Qwen 2.5 7B ou Llama 3.1 8B) às vezes enfrentam dificuldades em manter o formato padrão de JSON Schema em quantizações muito agressivas.

Para resolver isso de forma **100% confiável**, implementamos uma **estratégia de dupla camada (Dual-Layer)** no ecossistema:
1. **Camada de Binding Nativo:** Tentativa de execução do binding nativo fornecido pelo LangChain.
2. **Camada de Parse via String/Prompts Otimizados:** Caso o modelo falhe na geração do JSON Schema puro, o grafo interpreta a sintaxe otimizada por tags no prompt (`Ação: <tool_name>` e `Argumento: <args>`). Isso garante que os agentes nunca falhem em realizar suas chamadas de ferramenta corporativas, mesmo rodando em hardware humilde ou CPUs puras.

---

## 📊 5. Guia Operacional e Otimização de Hardware

### Dimensionamento de VRAM para Modelos Open-Source

A tabela abaixo serve como guia executivo para dimensionar a memória de GPU (VRAM) necessária com base nas quantizações comuns de mercado (AWQ, GPTQ, GGUF, FP16):

| Modelo de IA | Tamanho Original (FP16) | Quantização Recomendada | VRAM Requerida (Inference Only) | Hardware Mínimo Recomendado |
|:---|:---:|:---:|:---:|:---|
| **Llama 3.1 (8B)** | 16 GB | Q4_K_M GGUF / AWQ | **~5.5 GB** | NVIDIA RTX 3060 / 4060 (Laptop ou Desktop) |
| **Qwen 2.5 Coder (7B)** | 14 GB | Q8_0 GGUF / AWQ | **~8.5 GB** | NVIDIA RTX 3070 / 4070 (8GB - 12GB VRAM) |
| **DeepSeek-R1 (1.5B)** | 3 GB | Q8_0 | **~2 GB** | CPU Pura ou qualquer GPU integrada |
| **DeepSeek-R1 (8B)** | 16 GB | Q4_K_M | **~6 GB** | NVIDIA RTX 4060 (8GB VRAM) |
| **Qwen 2.5 Coder (32B)**| 64 GB | Int4 AWQ / GPTQ | **~19.5 GB** | NVIDIA RTX 3090 / 4090 / A6000 (24GB VRAM) |

### ⚙️ Ajustes Finos (Tuning) para Evitar Out-Of-Memory (OOM)

No arquivo de configuração `config/llm.py` e nos parâmetros de inferência, aplicamos as seguintes heurísticas de otimização:

1. **Ajuste de Context Window (`num_ctx`):**
   - Definido por padrão como **4096** tokens. Embora modelos modernos suportem até 128k, usar janelas de contexto gigantescas em hardware local escala o uso de VRAM de forma quadrática no mecanismo de atenção clássico. Para análises de contratos comuns, 4096 tokens equilibram perfeitamente o espaço do documento e a memória disponível.
2. **Temperature & Top_P para Redução de Alucinações:**
   - **`temperature=0.0`:** Mantém a geração estritamente determinística, ideal para agentes de auditoria matemática e conformidade contratual.
   - **`top_p=0.9`:** Filtra o vocabulário de probabilidade acumulada apenas para os tokens de altíssima confiança jurídica e lógica.
3. **Continuous Batching & PagedAttention no vLLM:**
   - Através do comando `--max-model-len 4096` e uso do driver nativo de GPU, o vLLM otimiza o cache de chaves/valores (KV Cache) impedindo a fragmentação e economizando até **35% de VRAM** em requisições paralelas.

---

## 🔒 6. Operação Totalmente Offline / Air-Gapped

Para implantar este ecossistema em servidores isolados fisicamente da internet:

1. **Fase de Preparação (Internet Ativa):**
   - Baixe as imagens do Docker necessárias:
     ```bash
     docker pull qdrant/qdrant:v1.9.0
     docker pull ollama/ollama:latest
     docker pull vllm/vllm-openai:latest
     docker pull arizephoenix/phoenix:latest
     ```
   - Baixe os modelos locais de LLM e Embeddings usando o script `./download_models.sh`.
   - Exporte as imagens do Docker em arquivos `.tar`:
     ```bash
     docker save -o qdrant.tar qdrant/qdrant:v1.9.0
     docker save -o ollama.tar ollama/ollama:latest
     ```

2. **Fase de Instalação (Sem Internet / Air-Gapped):**
   - Transfira os arquivos `.tar` e os volumes persistentes do Ollama/vLLM (onde os modelos foram armazenados) para o servidor isolado usando mídia física ou rede interna segura.
   - Carregue as imagens no Docker local do servidor isolado:
     ```bash
     docker load -i qdrant.tar
     docker load -i ollama.tar
     ```
   - Execute o comando de subida:
     ```bash
     docker-compose up -d
     ```
   O ecossistema rodará de forma independente sem enviar sequer um único pacote para fora da sua intranet de segurança máxima corporativa.

---

## 🧪 7. Verificação e Testes

Para rodar a suite automatizada de testes locais (que verifica a funcionalidade de LLMs, ferramentas, grafos de transição de estados e o servidor de streaming FastAPI em modo 100% mockado e seguro):

```bash
python3 -m pytest private_agent_ecosystem/tests/
```

Todas as chamadas estarão seguras e validadas!
