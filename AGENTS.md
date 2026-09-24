# Diretrizes para Agentes de IA (AGENTS.md)

## 1. Contexto do Projeto
Este projeto é um servidor de inferência determinística focado em MLOps (parte do ecossistema Sciencekit). Ele substitui a geração de texto autorregressiva de LLMs tradicionais por uma arquitetura de classificação de passagem única (*single forward pass*), utilizando modelos estilo Llama (ex: Manacá-1B) e Qwen2 (ex: Qwen2.5-1.5B-Instruct, modelo padrão) através do framework Hugging Face `candle`.

O objetivo é fornecer uma API de latência ultrabaixa para roteamento semântico, estritamente compatível com a especificação da API do **Jev** (TypeSafe AI).

## 2. Arquitetura e Stack Tecnológico
*   **Linguagem:** Rust (Edition 2021).
*   **Servidor Web:** `actix-web` com concorrência assíncrona gerenciada pelo `tokio`.
*   **Motor de Inferência:** `candle-core`, `candle-nn`, `candle-transformers` (arquiteturas `Llama` e `Qwen2`).
*   **Layout de Módulos:**
    *   `src/api/` — DTOs, erros, rotas e *handlers* do Actix.
    *   `src/domain/` — lógica pura (calibração, labels, templates de prompt) e a *trait* `Evaluator`.
    *   `src/infrastructure/` — Candle: carregamento de checkpoint, tokenizador, forward paralelo vendorizado e o avaliador real.
    *   `src/config/`, `src/bootstrap/` — CLI e inicialização/servidor.
*   **Gerenciamento de Estado:** O modelo, o tokenizador e o `base_cache` (KV-Cache do prompt de sistema) devem ser carregados uma única vez na inicialização da aplicação e compartilhados com os *workers* do `actix-web` através de `actix_web::web::Data`. O cache mutável por requisição deve ser sempre um clone do cache base.
*   **Formatos suportados:** safetensors (único ou *sharded*), GGUF (denso e GGML-quantizado, ex.: `Q4_K_M`), PyTorch `.pth`/`.bin` e NumPy `.npz`. `FP8`/`GPTQ`/`AWQ` são rejeitados com mensagem clara.
*   **CPU:** atenção *flash* fundida do `candle-nn` é usada automaticamente na CPU (mantém GQA agrupado); `--features mkl` habilita BLAS Intel MKL. GGUF `Q4_K_M` é o modo CPU recomendado.
*   **GPU:** `--features cuda` (F16 automático). O devcontainer instala o toolkit CUDA via *feature* `nvidia-cuda` e reserva a GPU no `docker-compose.yml`; o host precisa apenas do driver e do NVIDIA container toolkit.
*   **Cache de sessão:** o prefixo `system + state` é retido num cache LRU (por hash canônico do state, limitado por nº de entradas e total de tokens). O valor guardado é sempre um clone; o cache retido nunca é mutado.

## 3. Contrato de API (Compatibilidade Jev)
A aplicação não gera texto livre. Ela expõe rotas que retornam tipos estruturados baseados na extração direta de *logits*. A rota principal é `POST /v1/systemone`, que aceita um payload JSON com `model`, `state` e `questions` (mapa de perguntas).

Cada pergunta é tipada em um dos formatos abaixo e pode ser combinada na mesma requisição:

*   **`noul`:** decisão booleana → `{"type": "noul", "noul": 0.0 a 1.0}`.
*   **`choice`:** seleciona a melhor opção de um conjunto restrito → `{"type": "choice", "choice": "String", "probabilities": {...}, "confidence": 0.0 a 1.0}`.
*   **`score`:** pontuação contínua mapeada a partir de posições do vocabulário → `{"type": "score", "score": f32, "legend": {...}, "probabilities": {...}, "confidence": 0.0 a 1.0}`.

Complementam a API: `GET /v1/models`, `GET /health` e `GET /health/live`.

## 4. Regras de Código (Inquebráveis)
1.  **Código Sempre em Inglês:** Todo código (variáveis, funções, structs, módulos, comentários e mensagens de commit) deve ser escrito em inglês. Documentação voltada ao usuário (`README.md`, `AGENTS.md`, respostas no chat) pode ser em português.
2.  **Proibição Absoluta de Abreviações:** O agente **jamais** deve utilizar abreviações em variáveis, funções, structs ou módulos.
    *   *Errado:* `calc_prob`, `ctx`, `req`, `init_kv`.
    *   *Correto:* `calculate_probability`, `context`, `request`, `initialize_key_value_cache`.
3.  **Isolamento de Cache:** O `cache_base` nunca deve ser mutado durante uma requisição de usuário. A rota deve clonar o cache, executar o *forward pass* a partir do comprimento da sequência do sistema (`system_sequence_length`) e descartar o clone ao fim do escopo. O cache de sessão (LRU por hash do state) também só guarda e entrega *clones*; o `cache_base` do contexto e cada prefixo retido permanecem imutáveis.
4.  **Tratamento de Erros:** Não utilize `unwrap()` ou `expect()` no código de produção. Mapeie os erros do Candle e do Actix para uma struct de erro customizada que retorne um `HttpResponse::InternalServerError` padronizado.
5.  **Proibição Total de `unwrap()`/`expect()` (inclusive em testes):** Nenhum arquivo em `src/` (incluindo `#[cfg(test)]`, mocks, helpers e exemplos internos) pode conter `.unwrap()`, `.expect(`, `.unwrap_err()` ou `.expect_err()`. Métodos que não causam panic (`unwrap_or`, `unwrap_or_else`, `unwrap_or_default`) são permitidos. Em testes, funções devem retornar `anyhow::Result<()>` (ou `Result<_, EvaluationError>`) e propagar com `?`; casos de erro devem ser verificados com `assert!(result.is_err())` + `let Err(error) = result else { return Ok(()); }`, e valores `Option` com `ok_or_else(|| anyhow::anyhow!(...))?` ou `unwrap_or`/`unwrap_or_default`. Valide com `grep -rn "unwrap()\|\.expect(\|unwrap_err" src/ --include="*.rs"` retornando vazio.

## 5. Diretrizes de Testes (Obrigatório)
Nenhum código, rota ou função deve ser gerado sem o respectivo teste automatizado. O agente deve assumir a metodologia TDD (Test-Driven Development) nas suas respostas.

1.  **Testes Unitários:** Para a lógica matemática de extração e calibração de probabilidades (Softmax restrito) a partir de tensores simulados (*dummy tensors*). Inclua também testes de equivalência numérica para os caminhos vendorizados: a atenção *flash* da CPU contra uma referência matmul/softmax (com GQA e deslocamento causal) e a tokenização estruturada contra o prompt monolítico.
2.  **Testes de Integração de API:** Utilize o `actix_web::test` para criar instâncias locais do serviço `App` e garantir que os payloads JSON de entrada e saída correspondam exatamente ao contrato do Jev.
3.  **Mocks Injetáveis:** Para evitar o download de modelos pesados durante os testes de CI/CD, crie uma abstração (*trait*) para o avaliador. A camada do Actix deve ser testável através de uma estrutura de modelo simulado (*mock*) que devolva *logits* previsíveis.
4.  **Testes Live (`#[ignore]`):** Casos que exigem pesos reais (equivalência com o upstream, ganho do cache de sessão, benchmark de latência) ficam marcados com `#[ignore]` e nunca rodam no CI.

## 6. Fluxo de Trabalho do Agente
Quando instruído a criar uma nova funcionalidade:
1. Comece desenhando os tipos de dados (Structs de *Request* e *Response*).
2. Escreva o teste de integração do Actix para a rota (que inicialmente falhará).
3. Implemente a lógica de manipulação de tensores (com testes unitários).
4. Conecte tudo no *handler* do Actix até que os testes passem.