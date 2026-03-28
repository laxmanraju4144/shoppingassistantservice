# Shopping Assistant Service

An AI-powered shopping assistant that analyzes a room image and recommends matching products using OpenAI GPT-4o and a PostgreSQL vector store (pgvector).

---

## How It Works

1. **Vision** — Sends the room image to GPT-4o, which describes the interior design style.
2. **Vector Search** — Embeds the room description + user prompt using OpenAI Embeddings and runs a similarity search against the product catalog stored in PostgreSQL (pgvector).
3. **RAG Response** — GPT-4o combines the retrieved products and the room description to generate a natural language recommendation with product IDs.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Kubernetes cluster | MicroK8s or EKS |
| OpenAI API Key | `sk-...` from [platform.openai.com](https://platform.openai.com) |
| LangChain API Key | `ls-...` from [smith.langchain.com](https://smith.langchain.com) (optional, for tracing) |
| PostgreSQL with pgvector | The in-cluster `postgres` service (upgraded to pgvector image) |

---

## Environment Variables

The service reads all credentials from environment variables injected via a Kubernetes Secret.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o and embeddings |
| `LANGCHAIN_API_KEY` | LangChain/LangSmith API key (optional, for tracing) |
| `DATABASE_URL` | PostgreSQL connection string with pgvector |
| `COLLECTION_NAME` | Table name for product embeddings (e.g. `products`) |

---

## Step-by-Step Setup

### Step 1 — Deploy the vectordb

`vectordb` is a dedicated PostgreSQL + pgvector instance just for the shopping assistant. It is completely separate from the `postgres` instance used by authservice.

```bash
kubectl apply -f GitOps/base/vectordb/deployment.yaml
kubectl apply -f GitOps/base/vectordb/service.yaml
```

> The init ConfigMap automatically enables the `vector` extension and creates the LangChain embedding tables on first startup.

### Step 2 — Create the Kubernetes Secrets

Create the postgres password secret (skip if already exists):

```bash
kubectl create secret generic postgres-secret \
  --from-literal=password=Manoj7100
```

Create the shopping assistant secret with your API keys:

```bash
kubectl create secret generic shopping-assistant-secrets \
  --from-literal=OPENAI_API_KEY=sk-your-openai-key-here \
  --from-literal=LANGCHAIN_API_KEY=ls-your-langchain-key-here \
  --from-literal=DATABASE_URL="postgresql+psycopg://authuser:Manoj7100@vectordb:5432/shoppingdb" \
  --from-literal=COLLECTION_NAME=products
```

### Step 3 — Load Product Embeddings

Before the service can recommend products, you need to embed the product catalog into PostgreSQL.

Connect to the postgres pod:

```bash
kubectl exec -it $(kubectl get pod -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- psql -U authuser -d shoppingdb
```

Insert your products with embeddings using a one-time Python script or via the LangChain `PGVector.from_documents()` API. Example:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
import os

products = [
    Document(page_content="Aviator Sunglasses - Retro-style gold sunglasses", metadata={"id": "OLJCESPC7Z", "name": "Sunglasses", "categories": "accessories"}),
    # add more products ...
]

vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="products",
    connection=os.environ["DATABASE_URL"],
)
vectorstore.add_documents(products)
```

### Step 4 — Deploy the Shopping Assistant

Apply the updated deployment:

```bash
kubectl apply -f GitOps/base/shoppingassistantservice/deployment.yaml
kubectl apply -f GitOps/base/shoppingassistantservice/service.yaml
```

Verify the pod is running:

```bash
kubectl get pods -l app=shoppingassistantservice
kubectl logs -l app=shoppingassistantservice
```

### Step 5 — Test the Service

Port-forward to test locally:

```bash
kubectl port-forward svc/shoppingassistantservice 8080:8080
```

Send a test request:

```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need a lamp for my living room",
    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Living_room_MiD.jpg/1280px-Living_room_MiD.jpg"
  }'
```

Expected response:

```json
{
  "content": "Your room has a modern minimalist style... I recommend the Bamboo Glass Jar for its clean lines... [OLJCESPC7Z], [L9ECAV7KIM], [2ZYFJ3GM2N]"
}
```

---

## Architecture

```
User (image + prompt)
        |
        v
  Flask POST /
        |
        v
  Step 1: GPT-4o Vision
  "Describe the room style"
        |
        v room_description
  Step 2: OpenAI Embeddings
  + pgvector similarity search
  (vectordb:5432 / shoppingdb)
        |
        v matching products
  Step 3: GPT-4o RAG
  "Recommend from these products"
        |
        v
  {"content": "recommendation + [id1],[id2],[id3]"}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Pod stuck in `0/1 Running` | Check `kubectl logs` — likely missing secret key |
| `relation "products" does not exist` | Product embeddings not loaded — run Step 3 |
| `vector extension not found` | Postgres image is not pgvector — re-apply deployment.yaml |
| `AuthenticationError: OpenAI` | Wrong or missing `OPENAI_API_KEY` in the secret |
| Empty recommendations | Products not embedded yet, or embedding model mismatch |
