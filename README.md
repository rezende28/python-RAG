# Python RAG — Busca Semântica em Documentos

Projeto autoral desenvolvido em Python para explorar o conceito de RAG (Retrieval-Augmented Generation) por meio da recuperação semântica de informações em documentos de texto.

## Sobre o projeto

A aplicação lê documentos no formato `.txt`, divide seus conteúdos em trechos (*chunks*) e utiliza um modelo de embeddings para transformar esses trechos em representações numéricas.

A partir de uma pergunta digitada pelo usuário, o sistema calcula a similaridade semântica entre a pergunta e os trechos armazenados, retornando os três resultados mais relevantes e identificando seus respectivos arquivos de origem.

## Tecnologias utilizadas

- Python
- Sentence Transformers
- Modelo `paraphrase-multilingual-MiniLM-L12-v2`

## Como funciona

1. Lê os arquivos `.txt` armazenados na pasta `documentos`.
2. Divide o conteúdo em trechos.
3. Gera embeddings para os trechos utilizando Sentence Transformers.
4. Recebe uma pergunta digitada pelo usuário.
5. Compara a pergunta com os documentos por similaridade semântica.
6. Exibe os três trechos mais relevantes, suas pontuações de similaridade e os arquivos de origem.

## Estrutura do projeto

```text
python-rag/
├── documentos/
│   ├── documento1.txt
│   └── documento2.txt
├── main.py
└── README.md
```

## Como executar

1. Clone este repositório.
2. Instale as dependências:

```bash
pip install sentence-transformers
```

3. Coloque arquivos `.txt` na pasta `documentos`.
4. Execute:

```bash
python main.py
```

5. Digite uma pergunta no terminal para consultar os documentos. Para encerrar, digite `sair`.

## Objetivo

Projeto de aprendizado prático para compreender o funcionamento de embeddings, busca semântica e recuperação de informações como parte do conceito de RAG.

**Observação:** este projeto realiza a etapa de recuperação de informações. Ainda não possui integração com um modelo de linguagem para gerar respostas a partir dos trechos recuperados.