# typed-lm

Monorepo Rust para **inferência determinística** e **treino de adapters** de
modelos estilo Llama/Qwen2, parte do ecossistema **Sciencekit**. Em vez de geração
autorregressiva, o servidor classifica respostas em uma *single forward pass*:
cada pergunta é respondida a partir dos *logits* de um modelo local executado com
[Candle](https://github.com/huggingface/candle). O treinador produz adapters
LoRA/QLoRA e artefatos quantizados (FP8/FP4) que o servidor consome diretamente.

A API HTTP é compatível com o formato **Jev (TypeSafe AI)**: o cliente envia
`state` (fatos do caso) e `questions` (`noul`/`choice`/`score` com instruções e
critérios) e recebe respostas tipadas — sem texto livre.

## Workspace

| Crate | Papel | Tipo |
|---|---|---|
| `typed-lm-common` | Contrato Jev, labels, rendering de prompt, detecção de checkpoint, device/dtype, quantização | lib |
| `typed-lm-serve` | Servidor Actix compatível com Jev (binário sem subcomando) | bin |
| `typed-lm-trainer` | Fine-tuning LoRA/QLoRA e quantização pós-treino (subcomandos `train`/`quantize`) | bin + lib |

```bash
cargo build --workspace
cargo run -p typed-lm-serve -- --help
cargo run -p typed-lm-trainer -- --help
```

## Rotas (`typed-lm-serve`)

### `POST /v1/systemone`

Avalia uma ou mais perguntas (`noul`, `choice` e `score` podem ser combinados na
mesma requisição). O campo `model` deve ser o nome servido (ver
`--served-model-name`) ou qualquer alias com o prefixo `jev-`. A validação é
estrutural: `questions` não pode ser vazio, `choice` exige ao menos um critério e
`score` exige de 2 a 10 níveis.

Request (ver `examples/request_mixed.json`):

```json
{
  "model": "jev-latest",
  "state": "Order #7710 arrived with a smashed box and a cracked vase inside. Delivery was 3 days ago and the customer asks what to do next.",
  "questions": {
    "refund_eligible": {
      "type": "noul",
      "instructions": "The customer is eligible for a full refund under the store policy."
    },
    "responsible_department": {
      "type": "choice",
      "instructions": "Which department should handle this case?",
      "criteria": {
        "billing": "Double charges and payment errors",
        "logistics": "Damaged, lost, or late shipments",
        "product_support": "Defective-item troubleshooting, replacements, and setup help"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this case?",
      "criteria": ["Routine", "Urgent", "Emergency"]
    }
  }
}
```

Response (formato; valores variam por modelo e contexto):

```json
{
  "model": "jev-latest",
  "answers": {
    "refund_eligible": { "type": "noul", "noul": 0.87 },
    "responsible_department": {
      "type": "choice",
      "choice": "logistics",
      "probabilities": { "billing": 0.05, "logistics": 0.9, "product_support": 0.05 },
      "confidence": 0.85
    },
    "urgency": {
      "type": "score",
      "score": 1.2,
      "legend": { "0": "Routine", "1": "Urgent", "2": "Emergency" },
      "probabilities": { "0": 0.2, "1": 0.4, "2": 0.4 },
      "confidence": 0.2
    }
  },
  "usage": { "input_tokens": 512, "output_tokens": 4 }
}
```

Semântica por tipo:

- `noul`: probabilidade da resposta afirmativa no campo `noul` (0.0 a 1.0).
- `choice`: rótulo vencedor em `choice`, distribuição em `probabilities` e
  `confidence`.
- `score`: valor esperado sobre os níveis em `score`, legenda índice→nome em
  `legend`, distribuição em `probabilities` e `confidence`.

Erros seguem o envelope `{"error": {"message": "..."}}`: corpo inválido ou
pergunta fora do contrato retorna `422`, modelo desconhecido `404` e falha de
inferência `500`.

### `GET /v1/models`

Lista o modelo servido mais o alias `jev-latest`:

```json
{
  "object": "list",
  "data": [{ "id": "jev-latest", "object": "model", "owned_by": "typed-lm" }],
  "models": [{ "name": "jev-latest", "description": "Jev-compatible model served from context '...'", "release_date": "unknown" }]
}
```

### `GET /health` e `GET /health/live`

`/health` retorna `{"status": "ok", "startup_seconds": 12.3}` (tempo de carga do
modelo no startup); `/health/live` retorna `{"status": "ok"}` e não depende do
modelo.

## Quickstart do servidor

Pré-requisitos: Rust estável.

```bash
cargo build
cargo run -p typed-lm-serve
```

O servidor escuta em `0.0.0.0:8080` por padrão (`http://127.0.0.1:8080`). No
primeiro startup ele baixa os pesos do modelo padrão para o cache local do
Hugging Face.

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_noul.json

# Ancorado nos fatos fictícios da loja GreenLeaf (resources/memory.md):
cargo run -p typed-lm-serve -- --context-path resources/memory.md
```

## Configuração do servidor

Cada opção do `serve` também pode vir de variável de ambiente
(flag CLI > env > default). Verificado em `cargo run -p typed-lm-serve -- --help`:

| CLI Flag | Environment Variable | Default |
|---|---|---|
| `--host` | `HOST` | `0.0.0.0` |
| `--port` | `PORT` | `8080` |
| `--model-id` | `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--model-revision` | `MODEL_REVISION` | `main` |
| `--weights-file` | `WEIGHTS_FILE` | (auto-detected) |
| `--tokenizer-file` | `TOKENIZER_FILE` | (next to the weights) |
| `--config-file` | `CONFIG_FILE` | (next to the weights) |
| `--context-path` | `CONTEXT_PATH` | (missing = empty context) |
| `--served-model-name` | `SERVED_MODEL_NAME` | `jev-latest` |
| `--model-dtype` | `MODEL_DTYPE` | `auto` (F32 on CPU, F16 on CUDA/Metal) |
| `--session-cache-entries` | `SESSION_CACHE_ENTRIES` | `16` |
| `--session-cache-tokens` | `SESSION_CACHE_TOKENS` | `32768` |
| `--hf-token` | `HF_TOKEN` | (missing) |

### Session prefix cache

A parte cara de uma requisição é o forward pass sobre o prefixo do *state*
(contexto de sistema + `state`). O servidor tokeniza `system + state` uma vez e
retém o KV-cache resultante em um cache LRU limitado, indexado por um hash
canônico do state. Um state que reaparece entre requisições pula esse forward
pass. O cache é limitado pelo número de entradas (`--session-cache-entries`) e
pelo total de tokens cacheados (`--session-cache-tokens`); as entradas menos
recentes são descartadas primeiro. Zerar qualquer limite desabilita o cache de
sessão. O cache retido **nunca** é mutado: cada requisição clona antes de usar.

### Modelos restritos e `HF_TOKEN`

Se você trocar para um modelo de acesso restrito via `--model-id`, aceite os
termos de uso na página do modelo e exporte um token com permissão de leitura
antes de subir o servidor:

```bash
HF_TOKEN=hf_seu_token cargo run -p typed-lm-serve -- --model-id recogna-nlp/bode-1b-instruct
```

Sem o token, o download falha com `401` — comportamento esperado, não bug.

### Modelos, layouts e arquiteturas suportados

O checkpoint é detectado automaticamente:

- **Layouts**: safetensors (único ou *sharded*), GGUF (denso ou GGML-quantizado,
  ex.: `Q4_K_M`), PyTorch `.pth`/`.bin` e NumPy `.npz`.
- **Arquiteturas**: Llama e Qwen2.
- **Weight kinds**: precisão completa (`BF16`/`F16`/`F32`), GGML-quantizado
  (GGUF), e **FP8 (`F8_E4M3`/`F8_E5M2`) e FP4 (MXFP4)** — estes últimos são
  **dequantizados no load** para denso F32, já que o Candle não traz kernel de
  matmul nesses tipos. `GPTQ`/`AWQ` seguem rejeitados com erro descritivo.

## Treino e quantização (`typed-lm-trainer`)

O treinador aplica LoRA/QLoRA sobre um checkpoint-base congelado, otimizando a
**cross-entropy na posição de decisão** (o último token do prompt, restrito aos
candidatos) — a mesma posição que o servidor lê no inference — com KL de
calibração opcional. O dataset é **Jev-native**: cada pergunta do contrato às
respostas ganha um campo `answer`.

### Formato do dataset

Um arquivo `.jsonl` (um registro por linha) ou `.json` (objeto único ou array).
Um diretório é varrido recursivamente (`discovery`).

```json
{
  "state": "charged twice",
  "questions": {
    "refund": { "type": "noul", "instructions": "Refund?", "answer": "yes" },
    "dept": {
      "type": "choice", "instructions": "Dept?",
      "criteria": { "billing": "Payments", "technical": "Bugs" },
      "answer": "technical"
    },
    "urg": {
      "type": "score", "instructions": "Urgent?",
      "criteria": ["Routine", "Urgent", "Emergency"],
      "answer": "Urgent"
    }
  }
}
```

A `answer` é semântica (`yes`/`no`, nome da opção, nome do nível) e é mapeada
para o rótulo de planilha (`A`, `B`, …) que o servidor pontua.

### Treinar um adapter

```bash
cargo run -p typed-lm-trainer -- train \
  --model-id /caminho/local/do/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train \
  --method lora --lora-rank 16 --lora-alpha 32 \
  --epochs 3 --batch-size 4 --learning-rate 1e-4 \
  --max-sequence-length 1024
```

Flags principais do `train`: `--method lora|qlora`, `--lora-rank`, `--lora-alpha`,
`--lora-dropout`, `--epochs`, `--batch-size`, `--gradient-accumulation-steps`,
`--learning-rate`, `--warmup-steps`, `--weight-decay`, `--maximum-gradient-norm`,
`--max-sequence-length`, `--minimum-improvement`, `--early-stop-patience`,
`--quantization none|fp8|fp4`, `--quantization-mode post-training|training`.
Veja `cargo run -p typed-lm-trainer -- train --help`.

Ao final são gravados `adapter.safetensors` e `adapter_config.json` no
`--output-directory`.

### Quantizar (PTQ)

```bash
cargo run -p typed-lm-trainer -- quantize \
  --model-id /caminho/local/do/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 \
  --output-directory output/quantized
```

O merge LoRA→base é feito (quando `--adapter-directory` é informado) e a
quantização pós-treino (PTQ) gera `model.safetensors` + `quantization_config.json`.
`FP4` é gravado como nibbles empacotados em `U8` + expoentes (`*_scale`), pois o
safetensors/Candle não converte `F4`; o loader dequantiza ambos para denso F32.

### Paridade CPU/CUDA

`PrecisionPolicy { master: F32, compute: F32(CPU)/BF16(GPU), reduction: F32 }`
mantém peso-mestre e otimizador em **F32** e usa BF16 apenas como precisão de
compute em GPU, preservando a qualidade do adapter entre dispositivos. Paridade
**funcional**, não de velocidade: treino real roda em CUDA; CI/testes rodam em
CPU com modelos dummy ou casos `#[ignore]`.

## Aceleração do servidor

### CPU

Para inferência em CPU o modo recomendado é um checkpoint GGUF `Q4_K_M`
(aproximadamente metade do custo por requisição do caminho denso `F32`) combinado
com a feature `mkl` (Intel MKL BLAS) e a atenção *flash* fundida da CPU (usada
automaticamente, mantendo GQA agrupado). `--model-dtype` fica `auto` (F32 em CPU).

```bash
cargo build --release -p typed-lm-serve --features mkl
cargo run --release -p typed-lm-serve --features mkl -- \
  --model-id Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --weights-file qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --context-path resources/memory.md
```

`.cargo/config.toml` já define `target-cpu=native`; compile e execute na mesma
máquina (remova em cross-compile).

#### Ganhos medidos em CPU

`reports_latency_breakdown` (release, Qwen2.5-1.5B denso, `F32`):

| Prefix | Stage | Baseline | + CPU flash | + MKL |
|---|---|---|---|---|
| 64 | prefill | 3.13 s | 2.34 s | **0.52 s** |
| 256 | prefill | 8.65 s | 4.97 s | **1.69 s** |
| 1024 | prefill | 28.23 s | 20.66 s | **13.13 s** |
| 64 | 5 batched suffixes | 1.44 s | 1.33 s | **0.25 s** |
| 256 | 5 batched suffixes | 2.47 s | 1.98 s | **0.35 s** |
| 1024 | 5 batched suffixes | 4.74 s | 4.59 s | **2.57 s** |
| 64 | single next token | 655 ms | 699 ms | **159 ms** |
| 256 | single next token | 811 ms | 347 ms | **175 ms** |
| 1024 | single next token | 815 ms | 545 ms | **300 ms** |

### GPU (CUDA) via devcontainer

O host precisa do driver NVIDIA e do NVIDIA container toolkit
(`nvidia-ctk runtime configure --runtime=docker`). O devcontainer instala o
toolkit CUDA (`nvcc`) via feature `nvidia-cuda` e reserva a GPU no
`docker-compose.yml`.

```bash
cargo build --release -p typed-lm-serve --features cuda
cargo run --release -p typed-lm-serve --features cuda -- \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --context-path resources/memory.md
```

`auto` seleciona pesos `F16` em CUDA. Verifique a GPU com `nvidia-smi` dentro do
container.

#### Latência medida em GPU

`reports_latency_breakdown` (release, Qwen2.5-1.5B, `F16`, RTX 3070):

| Prefix | prefill | 5 batched suffixes | single next token |
|---|---|---|---|
| 64 | 14 ms | 36 ms | 52 ms |
| 256 | 31 ms | 81 ms | 65 ms |
| 1024 | 154 ms | 379 ms | 64 ms |

## Testes

```bash
cargo test --workspace
cargo test --workspace -- --ignored --nocapture
cargo clippy --workspace --all-targets
cargo fmt --check
```

- `cargo test --workspace`: testes unitários de calibração de probabilidades,
  labels, rendering de prompt, evicção do session cache, atenção flash da CPU
  contra referência matmul/softmax, quantização FP8/FP4, dataset/collate, LoRA e
  loop de treino (com dummies) e integração de API via `actix_web::test` com
  `MockEvaluator` — sem download de pesos.
- `cargo test --workspace -- --ignored`: testes *live*, marcados com `#[ignore]`;
  baixam pesos reais uma vez (equivalência com o upstream, ganho do cache de
  sessão, benchmark de latência, carregamento de artefatos FP8/FP4). Não rodar no CI.
- `cargo clippy --workspace --all-targets` / `cargo fmt --check`: lint e
  formatação. Não use `--all-features` no Linux (a feature Metal exige macOS).

## Arquitetura

- **HTTP (`typed-lm-serve/src/api/`)**: `actix-web` com `web::scope("/v1")`
  (`POST /v1/systemone`, `GET /v1/models`) e `GET /health`, `GET /health/live`.
  Os *handlers* recebem um `SharedState` (avaliador + nome do modelo + nome do
  contexto + tempo de startup) via `web::Data`.
- **Contrato compartilhado (`typed-lm-common`)**: os DTOs de requisição, a
  aritmética de labels e o rendering do prompt vivem na *common*, de modo que
  servidor e treinador produzam prompts byte-idênticos e concordem sobre a
  posição de decisão.
- **Scoring (Candle)**: `CandleEvaluator` prefila o contexto de sistema fixo em
  um KV-cache **uma vez no startup**. Cada requisição faz um *single shared
  prefill* dos tokens comuns às perguntas, **broadcast** do cache na dimensão de
  batch de atenção, e avalia cada sufixo de pergunta em **um único forward
  pass batelado**. O prefixo de state (`system + state`) é retido num LRU
  limitado. Lê o *logit* de cada token de rótulo (`A`, `B`, …) na última posição
  e calibra a distribuição (softmax binária para `noul`, temperatura para
  `choice`/`score`).
- **Forward paralelo (`typed-lm-serve/src/infrastructure/parallel_llama.rs`)**:
  implementação Llama vendorizada e broadcastável. Valida contra o upstream
  `Llama`/`Qwen2` por testes de equivalência `#[ignore]`.
- **Treino (`typed-lm-trainer/src/`)**: `dataset` (discovery/record/loader/collate),
  `model` (precision/LoRA/forward diferenciável/weight_loading),
  `training` (loss/optimizer/checkpoint/loop) e `quantization` (export).
- **Context (`ContextProvider`)**: atualmente `FileContextProvider` (ex.:
  `--context-path resources/memory.md`); a interface permite trocar a fonte por
  retrieval (RAG) no futuro sem mudar handlers, avaliador ou API.
