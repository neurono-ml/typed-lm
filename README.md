# manaca-jev-like

Servidor de inferência determinística em Rust, parte do ecossistema **Sciencekit**.
Ele substitui a geração de texto autorregressiva por classificação em
*single forward pass*: cada pergunta é respondida a partir dos *logits* de um
modelo estilo Llama (ex.: Manacá-1B) executado localmente com
[Candle](https://github.com/huggingface/candle).

A API HTTP é compatível com o formato **Jev (TypeSafe AI)**: o cliente envia o
`state` (os fatos do caso) e as `questions` (noul/choice/score com instruções e
critérios) e recebe respostas tipadas — sem texto livre.

## Rotas

### `POST /v1/systemone`

Avalia uma ou mais perguntas (`noul`, `choice` e `score` podem ser combinadas
no mesmo pedido). O campo `model` deve ser o nome servido (ver
`--served-model-name`) ou qualquer alias com prefixo `jev-`. A validação é
estrutural: `questions` não pode ser vazio, `choice` exige ao menos um
critério e `score` exige de 2 a 10 níveis.

Pedido (ver `examples/request_mixed.json`):

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

Resposta (formato; valores variam conforme modelo e contexto):

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
pergunta fora do contrato retorna `422`, modelo desconhecido retorna `404` e
falha de inferência retorna `500`.

### `GET /v1/models`

Lista o modelo servido mais o alias `jev-latest`:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

```json
{
  "object": "list",
  "data": [{ "id": "jev-latest", "object": "model", "owned_by": "manaca" }],
  "models": [{ "name": "jev-latest", "description": "Jev-compatible model served from context '...'", "release_date": "unknown" }]
}
```

### `GET /health` e `GET /health/live`

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/health/live
```

`/health` retorna `{"status": "ok", "startup_seconds": 12.3}` (tempo de carga
do modelo na subida); `/health/live` retorna `{"status": "ok"}` e não depende
do modelo.

## Quickstart

Pré-requisitos: Rust estável (o binário chama-se `manaca-typed`, ver
`Cargo.toml`).

```bash
cargo build
cargo run -- serve
```

O servidor escuta em `0.0.0.0:8080` por padrão (acesse via
`http://127.0.0.1:8080`). Na primeira subida ele baixa os pesos do modelo
padrão para o cache local do Hugging Face.

Pedidos de exemplo (servidor rodando em outro terminal):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_noul.json

curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json

curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_context.json
```

Os exemplos usam os fatos fictícios da loja GreenLeaf descritos em
`examples/README.md`. Sem contexto (`--context-path` ausente) o avaliador
recebe um contexto vazio; para ancorar as respostas nos fatos de exemplo:

```bash
cargo run -- serve --context-path resources/memory.md
```

## Configuração

Toda opção do `serve` também pode vir de variável de ambiente
(flag CLI > env > default). Verificado contra `cargo run -- serve --help`:

| Flag CLI | Variável de ambiente | Default |
|---|---|---|
| `--host` | `HOST` | `0.0.0.0` |
| `--port` | `PORT` | `8080` |
| `--model-id` | `MODEL_ID` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `--context-path` | `CONTEXT_PATH` | (ausente = contexto vazio) |
| `--served-model-name` | `SERVED_MODEL_NAME` | `jev-latest` |
| `--hf-token` | `HF_TOKEN` | (ausente) |

Exemplo com ambiente:

```bash
PORT=9090 MODEL_ID=recogna-nlp/bode-1b-instruct cargo run -- serve
```

### Modelos com acesso restrito (gated) e `HF_TOKEN`

O modelo padrão (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) é público e não exige
autenticação. Se você trocar para um modelo com acesso restrito via
`--model-id` (por exemplo `recogna-nlp/bode-1b-instruct`), aceite as condições
de uso na página do modelo no Hugging Face e exporte um token com permissão
de leitura antes de subir o servidor:

```bash
HF_TOKEN=hf_seu_token_aqui cargo run -- serve --model-id recogna-nlp/bode-1b-instruct
```

Sem o token, o download dos pesos falha com erro `401`
(`failed to download ... status code 401`) — comportamento esperado, não um
bug. Detalhes e mais exemplos em `examples/README.md`.

## Testes

```bash
cargo test
cargo test -- --ignored
cargo clippy
cargo fmt
```

- `cargo test`: unitários (calibração de probabilidades, rótulos, prompt) e
  integração da API com `actix_web::test` via avaliador simulado
  (`MockEvaluator`), sem baixar pesos.
- `cargo test -- --ignored`: teste *live*, marcado com `#[ignore]`; baixa os
  pesos reais uma vez e valida respostas ancoradas em
  `resources/memory.md`. Não roda no CI.
- `cargo clippy` / `cargo fmt`: lint e formatação.

## Arquitetura

- **HTTP (`src/api/`)**: `actix-web` com `web::scope("/v1")` (`POST
  /v1/systemone`, `GET /v1/models`) e as rotas `GET /health` e
  `GET /health/live`. Os *handlers* recebem um `SharedState` (avaliador +
  nome do modelo + nome do contexto + tempo de subida) via `web::Data`.
- **Orquestração (Rig)**: o avaliador monta o histórico de conversa com tipos
  do `rig-core` — o contexto carregado como mensagem `system` e o
  `state` + texto da pergunta como mensagem `user` — e renderiza o prompt
  único avaliado pelo modelo. O Rig não expõe logprobs: ele organiza, não
  pontua.
- **Scoring (Candle)**: `CandleEvaluator` executa um *forward pass* por
  pergunta, lê o *logit* do token de cada rótulo de resposta (`A`, `B`, …) na
  última posição e calibra a distribuição (softmax binário para `noul`,
  softmax com temperatura para `choice`/`score`).
- **Contexto (`ContextProvider`)**: hoje implementado como
  `FileContextProvider` (ex.: `--context-path resources/memory.md`); a
  interface permite trocar a fonte por recuperação (RAG) no futuro sem
  alterar *handlers*, avaliador ou API.
