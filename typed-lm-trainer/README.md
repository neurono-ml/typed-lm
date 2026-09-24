# typed-lm-trainer

Fine-tuning **LoRA/QLoRA**, treino de parâmetros completos (**full** e
**from-scratch**) e quantização pós-treino (**PTQ FP8/FP4**) para os modelos
servidos pelo `typed-lm-serve`. O treinador otimiza a **cross-entropy na posição
de decisão** — o último token do prompt, restrito aos candidatos — que é
exatamente a posição que o servidor lê no inference, garantindo que o adapter
ajuste o comportamento que a API efetivamente usa.

## Subcomandos

```bash
cargo run -p typed-lm-trainer -- train --help
cargo run -p typed-lm-trainer -- quantize --help
```

## Formato do dataset (Jev-native)

Cada registro espelha o contrato de request do Jev (`state` + mapa de
`questions`) e adiciona um campo `answer` a cada pergunta. A `answer` é
**semântica**:

- `noul` → `"yes"` / `"no"` (ou os rótulos declarados em `criteria`);
- `choice` → o nome da opção;
- `score` → o nome do nível.

O treinador mapeia a `answer` para o rótulo de planilha (`A`, `B`, …) que o
servidor pontua.

Arquivos aceitos: `.jsonl` (um registro por linha) ou `.json` (objeto único ou
array). Um diretório é varrido recursivamente.

```jsonl
{"state": "charged twice", "questions": {"refund": {"type": "noul", "instructions": "Refund?", "answer": "yes"}, "dept": {"type": "choice", "instructions": "Dept?", "criteria": {"billing": "Payments", "technical": "Bugs"}, "answer": "technical"}}}
{"state": "package arrived broken", "questions": {"urg": {"type": "score", "instructions": "Urgent?", "criteria": ["Routine", "Urgent", "Emergency"], "answer": "Urgent"}}}
```

Um exemplo em `resources/dataset.jsonl`.

## Métodos de treino

`--method` seleciona a parcela treinada e o artefato produzido:

| `--method` | Parâmetros treináveis | Saída em `--output-directory` |
|---|---|---|
| `lora` | Adapters LoRA sobre um checkpoint denso congelado | `adapter.safetensors` + `adapter_config.json` |
| `qlora` | Adapters LoRA sobre um checkpoint-base quantizado (dequantizado no load) | `adapter.safetensors` + `adapter_config.json` |
| `full` | Todos os parâmetros, partindo de um checkpoint existente | `model.safetensors` + `config.json` + `tokenizer.json` |
| `from-scratch` | Todos os parâmetros, partindo de pesos inicializados aleatoriamente | `model.safetensors` + `config.json` + `tokenizer.json` |

`full` e `from-scratch` usam `save_full_checkpoint` e gravam um checkpoint
completo (pesos densos + `config.json` + `tokenizer.json`). Esse artefato é
**servível diretamente** pelo `typed-lm-serve`, sem etapa de merge de adapter.
Os métodos `full`/`from-scratch` exigem a geometria do modelo: informe-a via
checkpoint (`--model-id` com `config.json`) ou pelas flags de geometria abaixo.

## Treinar um adapter

```bash
cargo run -p typed-lm-trainer -- train \
  --model-id /caminho/local/do/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train \
  --method lora \
  --lora-rank 16 --lora-alpha 32 \
  --epochs 3 --batch-size 4 --learning-rate 1e-4
```

Flags principais:

| Flag | Descrição | Default |
|---|---|---|
| `--model-id` | Checkpoint-base local (diretório) | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--dataset` | Arquivo ou diretório do dataset | (obrigatório) |
| `--output-directory` | Destino do adapter ou do checkpoint completo | `output/train` |
| `--method` | `lora`, `qlora`, `full` ou `from-scratch` | `lora` |
| `--lora-rank` / `--lora-alpha` | Rank e alpha do LoRA (escala `alpha/rank`) | `16` / `32` |
| `--lora-dropout` | Dropout do adapter | `0` |
| `--epochs` | Épocas | `3` |
| `--batch-size` | Batch por passo (itens de mesmo state são agrupados) | `4` |
| `--gradient-accumulation-steps` | Micro-batches acumulados | `1` |
| `--learning-rate` | LR de pico (warmup + cosine) | `1e-4` |
| `--warmup-steps` | Passos de warmup | `10` |
| `--weight-decay` | Weight decay do AdamW | `0` |
| `--maximum-gradient-norm` | Clip por norma global | `1` |
| `--max-sequence-length` | Prompt máximo; itens maiores são ignorados | `1024` |
| `--minimum-improvement` | Melhora mínima que reseta a paciência | `0` |
| `--early-stop-patience` | Épocas sem melhora antes de parar (`0` desativa) | `0` |
| `--quantization` | `none`, `fp8` ou `fp4` | `none` |
| `--quantization-mode` | `post-training` ou `training` | `post-training` |
| `--device` | `auto` (CUDA > Metal > CPU), `cpu` ou `cuda` | `auto` |
| `--seed` | Semente de inicialização determinística (`from-scratch`) | `42` |
| `--tokenizer-file` | `tokenizer.json` usado por `from-scratch` (sem checkpoint não há tokenizer) | (nenhum) |
| `--configuration-file` | TOML opcional; flags explícitas da CLI têm precedência | (nenhum) |

Para `lora`/`qlora` a saída no `--output-directory` é `adapter.safetensors` +
`adapter_config.json`; para `full`/`from-scratch` é um checkpoint completo
(`model.safetensors` + `config.json` + `tokenizer.json`).

### Geometria do modelo (`full` / `from-scratch`)

Usadas quando a geometria não vem de um `config.json` de checkpoint. Todas são
opcionais na CLI (default `none`) e podem vir do arquivo TOML:

| Flag | Descrição |
|---|---|
| `--architecture` | Família (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `gemma3`) |
| `--hidden-size` | Dimensão oculta |
| `--intermediate-size` | Dimensão intermediária do feed-forward |
| `--num-hidden-layers` | Número de blocos transformer |
| `--num-attention-heads` | Número de cabeças de query |
| `--num-key-value-heads` | Número de cabeças key/value (GQA) |
| `--vocab-size` | Tamanho do vocabulário |
| `--max-position-embeddings` | Comprimento máximo de sequência |
| `--rope-theta` | Frequência-base do rotary embedding |
| `--rms-norm-eps` | Épsilon da normalização RMS |
| `--tie-word-embeddings` | Embeddings de entrada e saída compartilham peso |

A referência completa do TOML (seções `[run]`, `[model]`, `[initialization]`,
`[dataset]`, `[tokenizer]`) está em
[`docs/configuration-file.md`](../docs/configuration-file.md).

## Treinar do zero (`from-scratch`)

```bash
cargo run -p typed-lm-trainer -- train \
  --dataset resources/dataset.jsonl \
  --output-directory output/scratch \
  --method from-scratch \
  --architecture qwen2 \
  --hidden-size 512 --intermediate-size 2048 \
  --num-hidden-layers 8 --num-attention-heads 8 --num-key-value-heads 2 \
  --vocab-size 151936 --max-position-embeddings 1024 \
  --tokenizer-file /caminho/para/tokenizer.json \
  --seed 42 --device auto
```

Sem `--model-id`, o formato do prompt (rótulo de planilha) é fixado pela
`--architecture`; sem checkpoint, o tokenizer precisa ser informado em
`--tokenizer-file` (ou em `[tokenizer] file` no TOML).

## Quantizar (PTQ)

```bash
cargo run -p typed-lm-trainer -- quantize \
  --model-id /caminho/local/do/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 \
  --output-directory output/quantized
```

- `--adapter-directory` é opcional: quando informado, o adapter é mergeado ao
  peso-base antes da quantização.
- Saída: `model.safetensors` + `quantization_config.json`.
- `fp8` usa tensores `F8_E4M3` com escala por canal (`*_scale`).
- `fp4` (MXFP4) grava nibbles E2M1 empacotados em `U8` + expoentes `F8E8M0`
  (`*_scale`), pois o safetensors/Candle não converte `F4`. O loader dequantiza
  ambos os formatos para denso F32 no load.

## Paridade CPU/CUDA

`PrecisionPolicy { master: F32, compute: F32(CPU)/BF16(GPU), reduction: F32 }`:
peso-mestre e otimizador em F32 em ambos os dispositivos, BF16 só como compute em
GPU, reduções sempre em F32. A qualidade do adapter independe do device;
paridade é funcional, não de velocidade.

## Testes

```bash
cargo test -p typed-lm-trainer
cargo test -p typed-lm-trainer -- --ignored   # casos com pesos reais
```

Os testes E2E (overfit dummy, export FP8/FP4) rodam em CPU com um checkpoint
minúsculo em `tests/support/`, sem download. O teste live de GPU
(`tests/live_gpu_e2e.rs`) baixa um checkpoint real, treina LoRA em CUDA e exporta
FP8:

```bash
cargo test -p typed-lm-trainer --features cuda --test live_gpu_e2e -- --ignored --nocapture
```

Documentação detalhada (em inglês) em [`docs/training.md`](../docs/training.md).
