from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# 1. Localiza a pasta dos documentos do arquivo
pasta_projeto = Path(__file__).parent
pasta_documentos = pasta_projeto / "documentos"


# 2. Chunking inteligente: divide o texto por blocos/parágrafos (linhas em branco)
def dividir_em_chunks(texto):
    # O split("\n\n") separa o texto em parágrafos ou blocos de regras
    blocos = texto.split("\n\n")
    return [bloco.strip() for bloco in blocos if bloco.strip()]


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


# 4. Carrega o modelo de embeddings multilíngue
print("Carregando o modelo de embeddings...")
modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# 5. Transforma todos os blocos de texto em vetores numéricos
print("Gerando embeddings dos documentos do arquivo...")
embeddings_documentos = modelo.encode(
    chunks, convert_to_tensor=True, normalize_embeddings=True
)


# 6. Loop de atendimento interativo
print("\nAssistente de Arquivo RAG pronto para consultas!")
print("Digite 'sair' para encerrar.")

while True:
    pergunta = input("\nFaça sua pergunta sobre prazos ou solicitações: ").strip()

    if pergunta.lower() == "sair":
        print("Até a próxima!")
        break

    if not pergunta:
        print("Digite uma pergunta válida.")
        continue

    # Transforma a pergunta do usuário em embedding
    embedding_pergunta = modelo.encode(
        pergunta, convert_to_tensor=True, normalize_embeddings=True
    )

    # Calcula a similaridade entre a pergunta e cada bloco de texto
    pontuacoes = util.cos_sim(embedding_pergunta, embeddings_documentos)[0]

    # Pega os 3 melhores trechos
    quantidade = min(3, len(chunks))
    melhores_indices = pontuacoes.argsort(descending=True)[:quantidade]

    print("\n--- Trechos mais relevantes encontrados ---")

    for indice in melhores_indices:
        posicao = indice.item()
        similaridade = pontuacoes[posicao].item()

        # Filtro opcional: só mostra se a similaridade for razoável (ex: acima de 0.3)
        if similaridade < 0.3:
            continue

        print(f"\nDocumento: {fontes[posicao]}")
        print(f"Similaridade: {similaridade:.3f}")
        print(f"Trecho:\n{chunks[posicao]}")
        print("-" * 40)