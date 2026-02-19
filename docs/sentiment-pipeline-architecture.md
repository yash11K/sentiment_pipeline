# Sentiment Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SENTIMENT ANALYSIS PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   DATA SOURCE    │     │    INGESTION     │     │    ENRICHMENT    │
│                  │     │                  │     │                  │
│   Amazon S3      │────▶│  S3ReviewSource  │────▶│  ReviewEnricher  │
│                  │     │  ReviewParser    │     │                  │
│  Path Format:    │     │                  │     │  • Batch Process │
│  {source}/       │     │  • List S3 files │     │  • LLM Analysis  │
│  {brand}/        │     │  • Parse JSON    │     │  • Topics        │
│  {LOCODE}_       │     │  • Normalize     │     │  • Sentiment     │
│  {YYYY-MM-DD}_   │     │  • Track status  │     │  • Key Phrases   │
│  {LAT}_{LON}     │     │  • Brand detect  │     │  • Urgency Level │
│  .json           │     │  • Competitor    │     │  • Actionable    │
│                  │     │    classify      │     │                  │
│  Examples:       │     │  • Insert to DB  │     │                  │
│  google/avis/    │     │                  │     │                  │
│  ATL_2026-02-12_ │     │                  │     │                  │
│  33.64_-84.42    │     │                  │     │                  │
│  .json           │     │                  │     │                  │
└──────────────────┘     └────────┬─────────┘     └────────┬─────────┘
                                  │                        │
                                  ▼                        ▼
                         ┌──────────────────────────────────────────┐
                         │               DATABASE                   │
                         │                                          │
                         │  ┌─────────────┐    ┌─────────────────┐  │
                         │  │  reviews    │    │  enrichments    │  │
                         │  │             │    │                 │  │
                         │  │ • review_id │◀──▶│ • review_id     │  │
                         │  │ • location  │    │ • topics[]      │  │
                         │  │ • source    │    │ • sentiment     │  │
                         │  │ • brand     │    │ • score         │  │
                         │  │ • is_comp.  │    │ • entities[]    │  │
                         │  │ • rating    │    │ • key_phrases[] │  │
                         │  │ • text      │    │ • urgency_level │  │
                         │  │ • date      │    │ • actionable    │  │
                         │  ┌─────────────┐    │ • suggested_    │  │
                         │  │ ingestion_  │    │   action        │  │
                         │  │ files       │    └─────────────────┘  │
                         │  │             │                         │
                         │  │ • s3_key    │    ┌─────────────────┐  │
                         │  │ • status    │    │ insights_cache  │  │
                         │  │ • counts    │    └─────────────────┘  │
                         │  └─────────────┘                         │
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
                         │              FastAPI SERVER              │
                         │                                          │
                         │  /api/dashboard/summary                  │
                         │  /api/dashboard/trends                   │
                         │  /api/dashboard/topics                   │
                         │  /api/dashboard/highlight                │
                         │  /api/chat                               │
                         │  /api/reviews                            │
                         │  /api/insights/{location}                │
                         │                                          │
                         │  COMPETITIVE ANALYSIS APIs:              │
                         │  /api/competitive/summary                │
                         │  /api/competitive/topics                 │
                         │  /api/competitive/trends                 │
                         │  /api/competitive/gap-analysis           │
                         │  /api/competitive/market-position        │
                         │                                          │
                         │  INGESTION APIs:                         │
                         │  /api/ingestion/pending                  │
                         │  /api/ingestion/process                  │
                         │  /api/ingestion/history                  │
                         │  /api/ingestion/status/{s3_key}          │
                         └──────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AWS BEDROCK INTEGRATION                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   BedrockClient                                                                  │
│   ├── Model: Claude Sonnet                                                       │
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
│                              DATA FLOW SUMMARY                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

  1. INGEST    S3 (scraped Google reviews) → ReviewParser → reviews table
  2. ENRICH    reviews → BedrockClient (batch) → enrichments table
  3. ANALYZE   enrichments → InsightGenerator → insights_cache
  4. SERVE     Database → FastAPI → API (frontend externally managed)
  5. INTERACT  User Query → ChatEngine → Bedrock KB → Response


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TOPIC CATEGORIES                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

  • wait_times          • pricing_fees        • cleanliness
  • staff_behavior      • reservation_issues  • preferred_program
  • vehicle_condition   • customer_service    • tolls_epass
  • location_access     • insurance           • shuttle_service
  • upgrade_downgrade   • damage_claims
```

## S3 Ingestion API

### File Path Convention
Files in S3 follow the structure: `{source}/{brand}/{LOCODE}_{YYYY-MM-DD}_{LAT}_{LON}.json`

| Component | Description | Example |
|-----------|-------------|---------|
| source | Review platform | google, tripadvisor, yelp |
| brand | Car rental brand | avis, hertz, budget, enterprise |
| LOCODE | 3-letter airport/location code | ATL, LAX, JFK |
| YYYY-MM-DD | Scrape date | 2026-02-12 |
| LAT_LON | GPS coordinates | 33.6407_-84.4277 |

Examples:
- `google/avis/ATL_2026-02-12_33.6407_-84.4277.json`
- `google/hertz/ATL_hertz_2026-02-12_33.6407_-84.4277.json`
- `google/budget/LAX_2026-01-15_33.9425_-118.4081.json`

### Brand Classification

Our brands (is_competitor=false): Avis, Budget, Payless, Apex, Maggiore
All other brands are classified as competitors (is_competitor=true).

### Competitive Analysis API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/competitive/summary` | GET | Market overview: our brands vs competitors |
| `/api/competitive/topics` | GET | Topic distribution comparison across brands |
| `/api/competitive/trends` | GET | Rating trends over time per brand |
| `/api/competitive/gap-analysis` | GET | Where competitors outperform us and vice versa |
| `/api/competitive/market-position` | GET | Market share and rating rank per brand |

All competitive endpoints accept optional `location_id` query parameter to scope to a specific market.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingestion/pending` | GET | List unprocessed S3 files |
| `/api/ingestion/process` | POST | Process selected files |
| `/api/ingestion/history` | GET | View ingestion history |
| `/api/ingestion/status/{s3_key}` | GET | Check file status |

### Enhanced Enrichment Metadata

The enrichment process now extracts additional metadata:

| Field | Type | Description |
|-------|------|-------------|
| topics | array | Categorized topics from predefined list |
| sentiment | string | positive/negative/neutral |
| sentiment_score | float | -1.0 to 1.0 |
| entities | array | Named entities (staff names, programs) |
| key_phrases | array | Short phrases capturing main points |
| urgency_level | string | low/medium/high/critical |
| actionable | boolean | Whether review suggests improvement |
| suggested_action | string | Brief action item if actionable |

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| S3ReviewSource | `src/ingestion/s3_source.py` | S3 bucket operations, file listing |
| IngestionPipeline | `src/ingestion/pipeline.py` | Orchestrates S3-based ingestion |
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
# Run full pipeline (local files)
python src/run_ingestion.py

# Start API server
python src/api/app.py

# Environment variables for S3 ingestion
export REVIEWS_S3_BUCKET=review-intelligence-data
export REVIEWS_S3_PREFIX=reviews/
export AWS_REGION=us-east-1
```

### S3 Ingestion via API

```bash
# List pending files
curl http://localhost:8000/api/ingestion/pending

# Process specific files
curl -X POST http://localhost:8000/api/ingestion/process \
  -H "Content-Type: application/json" \
  -d '{"s3_keys": ["reviews/LAX_google_10_01_2026.json"], "enrich": true}'

# Check ingestion history
curl http://localhost:8000/api/ingestion/history
```
