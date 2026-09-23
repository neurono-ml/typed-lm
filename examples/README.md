# Exemplos de uso da API

Exemplos de corpo de requisição para `POST /v1/systemone`.
As respostas dependem dos fatos de `resources/memory.md`
(loja fictícia GreenLeaf).

## Subir o servidor

```bash
cargo run -- serve
```

O servidor sobe por padrão em `http://127.0.0.1:8080`
sem contexto (`--context-path` ausente). Passe
`--context-path resources/memory.md` (ou `CONTEXT_PATH`)
para ancorar as respostas nos fatos de exemplo.

## Modelos com acesso restrito (gated)

O modelo padrão (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) é público
e não precisa de autenticação. Se você trocar para um modelo
com acesso restrito via `--model-id` (por exemplo
`recogna-nlp/bode-1b-instruct`), é preciso aceitar as condições
de uso na página do modelo no Hugging Face e exportar um token
com permissão de leitura antes de subir o servidor:

```bash
HF_TOKEN=hf_seu_token_aqui cargo run -- serve --model-id recogna-nlp/bode-1b-instruct
```

Sem o `HF_TOKEN`, o download dos pesos falha com erro `401`
(`failed to download ... status code 401`) — esse é o
comportamento esperado, não um bug.

## Variáveis de ambiente

Toda opção do `serve` também pode vir de ambiente
(flag CLI > env > default): `HOST`, `PORT`,
`MODEL_ID`, `CONTEXT_PATH`,
`SERVED_MODEL_NAME` (e `HF_TOKEN` para `--hf-token`).

```bash
PORT=9090 MODEL_ID=recogna-nlp/bode-1b-instruct cargo run -- serve
```

## Pedidos `curl`

Booleano simples (elegibilidade de reembolso de cobrança duplicada):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_noul.json
```

Booleano + roteamento + urgência (item danificado no transporte):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json
```

Perguntas complexas ancoradas na memória (purificador de ar
de uso médico com defeito: elegibilidade, departamento
responsável, urgência e regra de vale-presente):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_context.json
```

## Teste live de integração (requer pesos reais)

O teste live carrega o modelo real e valida respostas ancoradas
na memória. Ele é marcado com `#[ignore]` e **não** roda no CI.
Para executá-lo (baixa os pesos uma vez para o cache local):

```bash
cargo test -- --ignored
```
