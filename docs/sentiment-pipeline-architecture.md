# Sentiment Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SENTIMENT ANALYSIS PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   DATA SOURCE    │     │    INGESTION     │     │    ENRICHMENT    │
│                  │     │                  │     │                  │
│   Amazon S3      │────▶│  ReviewParser    │────▶│  ReviewEnricher  │
│                  │     │                  │     │                  │
│  ┌────────────┐  │     │  • Parse JSON    │     │  • Batch Process │
│  │  Google    │  │     │  • Normalize     │     │  • LLM Analysis  │
│  │  Reviews   │  │     │  • Date Extract  │     │                  │
│  │  (Scraped) │  │     │  • Insert to DB  │     │                  │
│  └────────────┘  │     │                  │     │                  │
└──────────────────┘     └────────┬─────────┘     └────────┬─────────┘
                                  │                        │
                                  │                        │
                                  ▼                        ▼
                         ┌──────────────────────────────────────────┐
                         │              SQLite DATABASE              │
                         │                                          │
                         │  ┌─────────────┐    ┌─────────────────┐  │
                         │  │  reviews    │    │  enrichments    │  │
                         │  │             │    │                 │  │
                         │  │ • review_id │◀──▶│ • review_id     │  │
                         │  │ • location  │    │ • topics[]      │  │
                         │  │ • rating    │    │ • sentiment     │  │
                         │  │ • text      │    │ • score         │  │
                         │  │ • date      │    │ • entities[]    │  │
                         │  └─────────────┘    └─────────────────┘  │
                         │                                          │
                         │  ┌─────────────┐    ┌─────────────────┐  │
                         │  │ locations   │    │ insights_cache  │  │
                         │  └─────────────┘    └─────────────────┘  │
                         └──────────────────────────────────────────┘
                                           │
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
         ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
         │   INSIGHTS GEN   │   │   CHAT ENGINE    │   │   FILTER ENGINE  │
         │                  │   │                  │   │                  │
         │ • Top Topics     │   │ • Semantic Search│   │ • Rating Filter  │
         │ • Key Drivers    │   │ • RAG via KB     │   │ • Date Range     │
         │ • Anomalies      │   │ • LLM Response   │   │ • Topic Filter   │
         │ • Summaries      │   │                  │   │ • Sentiment      │
         └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │              FastAPI SERVER               │
                         │                                          │
                         │  /api/dashboard/summary                  │
                         │  /api/dashboard/trends                   │
                         │  /api/dashboard/topics                   │
                         │  /api/dashboard/highlight                │
                         │  /api/chat                               │
                         │  /api/reviews                            │
                         │  /api/insights/{location}                │
                         └──────────────────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │              WEB DASHBOARD                │
                         │                                          │
                         │  • Sentiment Charts                      │
                         │  • Topic Analysis                        │
                         │  • Rating Trends                         │
                         │  • Interactive Chat                      │
                         │  • Location Map                          │
                         └──────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AWS BEDROCK INTEGRATION                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   BedrockClient                                                                  │
│   ├── Model: Claude Sonnet                                                     │
│   │                                                                              │
│   ├── enrich_reviews_batch()  ──────────────────────────────────────────────┐    │
│   │   • Batch processing (20 reviews/call)                                  │    │
│   │   • Topic extraction                                                    │    │
│   │   • Sentiment analysis (-1 to +1 score)                                 │    │
│   │   • Entity extraction                                                   │    │
│   │                                                                         │    │
│   ├── retrieve()  ──────────────────────────────────────────────────────────┤    │
│   │   • Knowledge Base RAG                                                  │    │
│   │   • Vector search                                                       │    │
│   │   • Location filtering                                                  │    │
│   │                                                                         │    │
│   └── invoke()  ────────────────────────────────────────────────────────────┘    │
│       • General LLM calls                                                        │
│       • Insight generation                                                       │
│       • Chat responses                                                           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW SUMMARY                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

  1. INGEST    S3 (scraped Google reviews) → ReviewParser → reviews table
  2. ENRICH    reviews → BedrockClient (batch) → enrichments table
  3. ANALYZE   enrichments → InsightGenerator → insights_cache
  4. SERVE     Database → FastAPI → Web Dashboard
  5. INTERACT  User Query → ChatEngine → Bedrock KB → Response


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TOPIC CATEGORIES                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

  • wait_times          • pricing_fees        • cleanliness
  • staff_behavior      • reservation_issues  • preferred_program
  • vehicle_condition   • customer_service    • tolls_epass
  • location_access     • insurance
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| ReviewParser | `src/ingestion/parser.py` | Parse JSON, normalize, insert reviews |
| ReviewEnricher | `src/ingestion/enricher.py` | Batch LLM enrichment orchestration |
| BedrockClient | `src/utils/bedrock.py` | AWS Bedrock API wrapper |
| Database | `src/storage/db.py` | SQLite operations |
| InsightGenerator | `src/monitor/insights.py` | Generate cached insights |
| ChatEngine | `src/explore/chat.py` | RAG-powered chat |
| FilterEngine | `src/explore/filters.py` | Review filtering |
| FastAPI App | `src/api/app.py` | REST API endpoints |

## Pipeline Execution

```bash
# Run full pipeline
python src/run_ingestion.py

# Start API server
python src/api/app.py
```
