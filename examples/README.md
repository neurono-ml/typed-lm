# Exemplos de uso da API

Exemplos de corpo de requisição para `POST /v1/systemone`.
As respostas dependem dos fatos de `resources/memory.md`
(loja fictícia GreenLeaf).

## Subir o servidor

```bash
cargo run -- serve
```

O servidor sobe por padrão em `http://127.0.0.1:8080`
com o contexto `resources/memory.md`.

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
