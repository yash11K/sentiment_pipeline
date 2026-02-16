# Review Intelligence Product

Dual-mode system for analyzing car rental reviews with AI-powered insights.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials (for Bedrock):
```bash
aws configure
```

3. Run ingestion:
```bash
python src/run_ingestion.py
```

4. Start the API:
```bash
python src/api/app.py
```

5. Open browser: `http://localhost:8000`

## Modes

**Monitor Mode**: Pre-computed insights, top topics, key drivers
**Explore Mode**: Chat interface + filters for ad-hoc analysis

## Project Structure

- `src/ingestion/` - Data parsing and enrichment
- `src/storage/` - Database and vector store
- `src/monitor/` - Insights generation
- `src/explore/` - Chat and filters
- `src/api/` - FastAPI backend
