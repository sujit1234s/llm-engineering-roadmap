import logging
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from google import genai
from google.genai import errors, types

# 1. Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"
MIN_SIMILARITY_THRESHOLD = 0.55  # Guardrail: reject unrelated matches

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("SemanticSearch")


# 2. Structured Schema for Knowledge Base Documents
class KnowledgeDoc(BaseModel):
    doc_id: str
    category: str
    content: str
    embedding: list[float] = Field(default_factory=list)


# 3. Resilient Embedding Function (Carrying forward Day 5 resilience)
def is_transient_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in {429, 500, 502, 503, 504}


@retry(
    retry=retry_if_exception(is_transient_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    reraise=True,
)
def get_embedding(text: str, task_type: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,  # Compact, fast 768-d vectors
        ),
    )
    return response.embeddings[0].values


# 4. Vector Math: Cosine Similarity from Scratch
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# 5. Build & Index Company Knowledge Base
RAW_POLICIES = [
    KnowledgeDoc(
        doc_id="POL-01",
        category="Returns",
        content="Customers may return delivered electronics within 30 days of receipt for a full refund.",
    ),
    KnowledgeDoc(
        doc_id="POL-02",
        category="Shipping",
        content="Orders currently in transit cannot be refunded until physical delivery is completed.",
    ),
    KnowledgeDoc(
        doc_id="POL-03",
        category="Hardware Support",
        content="If your laptop fails to charge, inspect the AC power adapter cable and test an alternate wall outlet.",
    ),
    KnowledgeDoc(
        doc_id="POL-04",
        category="Security",
        content="Multi-factor authentication (MFA) can be reset by verifying your registered mobile number via OTP.",
    ),
]

print("⏳ Indexing knowledge base documents into 768-dimensional vectors...")
for doc in RAW_POLICIES:
    doc.embedding = get_embedding(doc.content, task_type="RETRIEVAL_DOCUMENT")
print(f"✅ Indexed {len(RAW_POLICIES)} documents (Vector dims: {len(RAW_POLICIES[0].embedding)})\n")


# 6. Semantic Search Engine with Confidence Guardrail
def semantic_search(query: str, top_k: int = 2) -> list[tuple[float, KnowledgeDoc]]:
    query_vec = get_embedding(query, task_type="RETRIEVAL_QUERY")

    scored_docs: list[tuple[float, KnowledgeDoc]] = []
    for doc in RAW_POLICIES:
        score = cosine_similarity(query_vec, doc.embedding)
        scored_docs.append((score, doc))

    # Sort highest similarity first
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return scored_docs[:top_k]


# 7. Test Queries (Notice zero exact keyword overlap in Queries 1 & 2!)
test_queries = [
    "My notebook battery isn't filling up when plugged in.",        # Matches POL-03 semantically!
    "Can I cancel and get my money back while the package is on the road?",  # Matches POL-02!
    "What is the recipe for masala dosa?",                          # Off-topic -> blocked by threshold!
]

for q in test_queries:
    print(f"{'='*65}\n🔍 QUERY: '{q}'\n{'='*65}")
    results = semantic_search(q, top_k=2)

    top_score, top_doc = results[0]
    if top_score < MIN_SIMILARITY_THRESHOLD:
        print(f"🛡️ GUARDRAIL BLOCKED: Top match score ({top_score:.4f}) below threshold ({MIN_SIMILARITY_THRESHOLD}). No relevant policy found.\n")
        continue

    for rank, (score, doc) in enumerate(results, start=1):
        status = "✅ MATCH" if score >= MIN_SIMILARITY_THRESHOLD else "⚠️ WEAK"
        print(f"  #{rank} [{status} | Score: {score:.4f}] ({doc.doc_id} - {doc.category}): {doc.content}")
    print()