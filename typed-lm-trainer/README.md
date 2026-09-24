# typed-lm-trainer

Fine-tuning **LoRA/QLoRA** e quantização pós-treino (**PTQ FP8/FP4**) para os
modelos servidos pelo `typed-lm-serve`. O treinador otimiza a **cross-entropy na
posição de decisão** — o último token do prompt, restrito aos candidatos — que é
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
| `--output-directory` | Destino do adapter | `output/train` |
| `--method` | `lora` ou `qlora` | `lora` |
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

Saída no `--output-directory`: `adapter.safetensors` + `adapter_config.json`.

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
minúsculo em `tests/support/`, sem download.
