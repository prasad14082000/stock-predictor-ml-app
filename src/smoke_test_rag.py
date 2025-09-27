from langchain_core.documents import Document
from src.rag.chroma_db import load_chroma
from src.rag.query_engine import run_query
from src.rag.schema import QueryRequest, OptionForecast

db = load_chroma("data/research/index")

req = QueryRequest(
    ticker="TITAN",
    question="What are the key drivers for margins in H2 and any management guidance?",
    top_k=5,
    use_mmr=True,
    forecasts=[
        OptionForecast(
            ticker="TITAN", expiry="2025-12-31", strike=3500.0,
            option_type="CALL", model_price=125.4, iv=0.29, spot=3320.0, source="black-scholes-v1"
        )
    ]
)

resp = run_query(db, req)
print(resp.thesis.model_dump_json(indent=2))
for c in resp.retrieved:
    print(c.source, c.page, c.source_type)
