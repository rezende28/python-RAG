
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# 1. Localiza a pasta dos documentos
pasta_projeto = Path(__file__).parent
pasta_documentos = pasta_projeto / "documentos"


# 2. Divide os documentos em chunks
def dividir_em_chunks(texto):
    return [
        linha.strip()
        for linha in texto.splitlines()
        if linha.strip()
    ]


# 3. Carrega os documentos e guarda a origem de cada chunk
chunks = []
fontes = []

for arquivo in pasta_documentos.glob("*.txt"):
    conteudo = arquivo.read_text(encoding="utf-8")

    trechos = dividir_em_chunks(conteudo)

    for trecho in trechos:
        chunks.append(trecho)
        fontes.append(arquivo.name)


if not chunks:
    raise ValueError("Nenhum documento TXT encontrado na pasta documentos.")


# 4. Carrega o modelo de embeddings
print("Carregando o modelo de embeddings...")
modelo = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# 5. Transforma todos os chunks em vetores numéricos
print("Gerando embeddings dos documentos...")
embeddings_documentos = modelo.encode(
    chunks,
    convert_to_tensor=True,
    normalize_embeddings=True
)


# 6. Recebe perguntas e procura os trechos mais semelhantes
print("\nRAG pronto para pesquisar!")
print("Digite 'sair' para encerrar.")

while True:
    pergunta = input("\nFaça sua pergunta: ").strip()

    if pergunta.lower() == "sair":
        print("Até a próxima!")
        break

    if not pergunta:
        print("Digite uma pergunta.")
        continue

    # Transforma a pergunta em embedding
    embedding_pergunta = modelo.encode(
        pergunta,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Calcula a similaridade entre a pergunta e cada chunk
    pontuacoes = util.cos_sim(
        embedding_pergunta,
        embeddings_documentos
    )[0]

    # Ordena os trechos do mais semelhante para o menos semelhante
    quantidade = min(3, len(chunks))
    melhores_indices = pontuacoes.argsort(
        descending=True
    )[:quantidade]

    print("\nTrechos recuperados:")

    for indice in melhores_indices:
        posicao = indice.item()
        similaridade = pontuacoes[posicao].item()

        print(f"\nDocumento: {fontes[posicao]}")
        print(f"Similaridade: {similaridade:.3f}")
        print(f"Trecho: {chunks[posicao]}")
