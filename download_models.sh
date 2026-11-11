#!/bin/bash
# Script de Bootstrap para Download de Modelos Locais no Ollama
# Autores: Arquiteto de Software Sênior & AI Engineer

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
echo "==> Iniciando o bootstrap dos modelos locais no Ollama..."
echo "==> Usando Ollama em: ${OLLAMA_HOST}"

# Aguarda o Ollama iniciar e responder na porta 11434
echo "==> Aguardando o serviço Ollama ficar pronto..."
until curl -s "${OLLAMA_HOST}" > /dev/null; do
    echo "    [Aguardando] O serviço Ollama ainda não está respondendo. Tentando novamente em 3 segundos..."
    sleep 3
done
echo "==> [OK] Ollama está online!"

# Lista de modelos requeridos
MODELS=(
    "llama3.1:8b"
    "qwen2.5-coder:7b"
    "nomic-embed-text"
)

# Pull de cada modelo de forma sequencial
for model in "${MODELS[@]}"; do
    echo "--------------------------------------------------------"
    echo "==> Baixando modelo: ${model}..."
    echo "--------------------------------------------------------"
    # Se estiver rodando dentro do Docker, podemos delegar para o CLI dentro do container
    if docker ps --format '{{.Names}}' | grep -q 'private_ollama'; then
        echo "    [DOCKER] Executando comando de pull dentro do container 'private_ollama'..."
        docker exec -it private_ollama ollama pull "${model}"
    else
        echo "    [LOCAL] Executando 'ollama pull' localmente..."
        ollama pull "${model}"
    fi
    echo "==> [OK] Modelo ${model} carregado com sucesso!"
done

echo "========================================================"
echo "==> TODOS OS MODELOS FORAM BAIXADOS E ESTÃO PRONTOS! <=="
echo "========================================================"
echo "Modelos ativos:"
if docker ps --format '{{.Names}}' | grep -q 'private_ollama'; then
    docker exec -it private_ollama ollama list
else
    ollama list
fi
