# Diretrizes para Agentes de IA (AGENTS.md)

## 1. Contexto do Projeto
Este projeto é um servidor de inferência determinística focado em MLOps (parte do ecossistema Sciencekit). Ele substitui a geração de texto autorregressiva de LLMs tradicionais por uma arquitetura de classificação de passagem única (*single forward pass*), utilizando modelos estilo Llama (ex: Manacá-1B) através do framework Hugging Face `candle`.

O objetivo é fornecer uma API de latência ultrabaixa para roteamento semântico, estritamente compatível com a especificação da API do **Jev** (TypeSafe AI).

## 2. Arquitetura e Stack Tecnológico
*   **Linguagem:** Rust (Edition 2021).
*   **Servidor Web:** `actix-web` com concorrência assíncrona gerenciada pelo `tokio`.
*   **Motor de Inferência:** `candle-core`, `candle-nn`, `candle-transformers` (foco na arquitetura `Llama`).
*   **Gerenciamento de Estado:** O modelo, o tokenizador e o `base_cache` (KV-Cache do prompt de sistema) devem ser carregados uma única vez na inicialização da aplicação e compartilhados com os *workers* do `actix-web` através de `actix_web::web::Data`. O cache mutável por requisição deve ser sempre um clone do cache base.

## 3. Contrato de API (Compatibilidade Jev)
A aplicação não gera texto livre. Ela expõe rotas que retornam tipos estruturados baseados na extração direta de *logits*. Toda rota deve aceitar um payload JSON com o `estado` (os dados a serem avaliados) e o `schema` (as regras).

O agente deve implementar as seguintes rotas base:

*   **POST `/v1/noul`:** Retorna uma decisão booleana.
    *   *Resposta esperada:* `{"decisao": true|false, "confianca": 0.0 a 1.0}`
*   **POST `/v1/choice`:** Seleciona a melhor opção de um conjunto restrito.
    *   *Resposta esperada:* `{"escolha": "String", "confianca": 0.0 a 1.0}`
*   **POST `/v1/score`:** Retorna uma pontuação contínua mapeada a partir de posições específicas do vocabulário.
    *   *Resposta esperada:* `{"pontuacao": f32, "confianca": 0.0 a 1.0}`

## 4. Regras de Código (Inquebráveis)
1.  **Código Sempre em Inglês:** Todo código (variáveis, funções, structs, módulos, comentários e mensagens de commit) deve ser escrito em inglês. Documentação voltada ao usuário (`README.md`, `AGENTS.md`, respostas no chat) pode ser em português.
2.  **Proibição Absoluta de Abreviações:** O agente **jamais** deve utilizar abreviações em variáveis, funções, structs ou módulos.
    *   *Errado:* `calc_prob`, `ctx`, `req`, `init_kv`.
    *   *Correto:* `calculate_probability`, `context`, `request`, `initialize_key_value_cache`.
3.  **Isolamento de Cache:** O `cache_base` nunca deve ser mutado durante uma requisição de usuário. A rota deve clonar o cache, executar o *forward pass* a partir do comprimento da sequência do sistema (`system_sequence_length`) e descartar o clone ao fim do escopo.
4.  **Tratamento de Erros:** Não utilize `unwrap()` ou `expect()` no código de produção. Mapeie os erros do Candle e do Actix para uma struct de erro customizada que retorne um `HttpResponse::InternalServerError` padronizado.

## 5. Diretrizes de Testes (Obrigatório)
Nenhum código, rota ou função deve ser gerado sem o respectivo teste automatizado. O agente deve assumir a metodologia TDD (Test-Driven Development) nas suas respostas.

1.  **Testes Unitários:** Para a lógica matemática de extração e calibração de probabilidades (Softmax restrito) a partir de tensores simulados (*dummy tensors*).
2.  **Testes de Integração de API:** Utilize o `actix_web::test` para criar instâncias locais do serviço `App` e garantir que os payloads JSON de entrada e saída correspondam exatamente ao contrato do Jev.
3.  **Mocks Injetáveis:** Para evitar o download de modelos pesados durante os testes de CI/CD, crie uma abstração (*trait*) para o avaliador. A camada do Actix deve ser testável através de uma estrutura de modelo simulado (*mock*) que devolva *logits* previsíveis.

## 6. Fluxo de Trabalho do Agente
Quando instruído a criar uma nova funcionalidade:
1. Comece desenhando os tipos de dados (Structs de *Request* e *Response*).
2. Escreva o teste de integração do Actix para a rota (que inicialmente falhará).
3. Implemente a lógica de manipulação de tensores (com testes unitários).
4. Conecte tudo no *handler* do Actix até que os testes passem.