from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from collections import Counter
from datetime import datetime
import sys
import json
import uuid
import threading
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.db import Database
from monitor.insights import InsightGenerator
from explore.chat import ChatEngine
from explore.filters import FilterEngine
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Review Intelligence API", description="Dashboard APIs for review sentiment analysis")

# Enable CORS for external frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


db = Database()
insights_gen = InsightGenerator(db)
chat_engine = ChatEngine(db)
filter_engine = FilterEngine(db)

# In-memory job tracking for background ingestion tasks
ingestion_jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()

logger.api("Review Intelligence API initialized")

class ChatRequest(BaseModel):
    query: str
    location_id: Optional[str] = None
    use_semantic: bool = True

@app.get("/")
async def home():
    return {"status": "ok", "service": "Review Intelligence API"}

@app.get("/api/locations")
async def get_locations():
    """Get all locations with coordinates for map display"""
    locations = db.get_locations_with_coords()
    brands_map = db.get_brands_by_location()
    for loc in locations:
        loc['brands'] = brands_map.get(loc['location_id'], [])
    return {"locations": locations}


class LocationUpdateRequest(BaseModel):
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None


@app.patch("/api/locations/{location_id}")
async def update_location(location_id: str, request: LocationUpdateRequest):
    """Update location details (name, address, coordinates)"""
    updated = db.update_location(
        location_id=location_id,
        name=request.name,
        latitude=request.latitude,
        longitude=request.longitude,
        address=request.address
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Location {location_id} not found")
    return {"location": updated}

@app.get("/api/insights/{location_id}")
async def get_insights(location_id: str, regenerate: bool = False):
    if regenerate:
        insights = insights_gen.generate_insights(location_id)
    else:
        insights = insights_gen.get_cached_insights(location_id)
        if not insights:
            insights = insights_gen.generate_insights(location_id)
    return insights

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.chat(f"Chat query: {request.query[:50]}...")
        response = chat_engine.chat(request.query, request.location_id, request.use_semantic)
        logger.success("Chat response generated")
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews")
async def get_reviews(
    location_id: Optional[str] = None,
    min_rating: Optional[float] = Query(None, ge=1, le=5),
    max_rating: Optional[float] = Query(None, ge=1, le=5),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topics: Optional[str] = None,
    sentiment: Optional[str] = None,
    brand: Optional[str] = None,
    is_competitor: Optional[bool] = None,
    limit: int = Query(100, le=1000)
):
    topic_list = topics.split(',') if topics else None
    reviews = filter_engine.apply_filters(
        location_id=location_id, min_rating=min_rating, max_rating=max_rating,
        start_date=start_date, end_date=end_date, topics=topic_list,
        sentiment=sentiment, brand=brand, is_competitor=is_competitor, limit=limit
    )
    return {"reviews": reviews, "count": len(reviews)}

@app.get("/api/stats/{location_id}")
async def get_stats(location_id: str):
    reviews = db.get_reviews(location_id=location_id, limit=10000)
    if not reviews:
        return {"error": "No reviews found"}
    ratings = [r['rating'] for r in reviews if r.get('rating')]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    return {
        "total_reviews": len(reviews),
        "average_rating": round(avg_rating, 2),
        "rating_distribution": {
            "1": len([r for r in ratings if r == 1]),
            "2": len([r for r in ratings if r == 2]),
            "3": len([r for r in ratings if r == 3]),
            "4": len([r for r in ratings if r == 4]),
            "5": len([r for r in ratings if r == 5])
        }
    }


# ============ DASHBOARD APIs ============

@app.get("/api/dashboard/summary")
async def get_dashboard_summary(location_id: Optional[str] = None):
    """Get complete dashboard summary with all key metrics"""
    from sqlalchemy import text
    
    with db.get_session() as session:
        # Base query filter
        location_filter = "WHERE r.location_id = :loc" if location_id else ""
        params = {"loc": location_id} if location_id else {}
        
        # Total reviews and average rating
        result = session.execute(text(f"""
            SELECT COUNT(*) as total, AVG(rating) as avg_rating 
            FROM reviews r {location_filter}
        """), params)
        stats = result.fetchone()
        
        # Sentiment breakdown
        result = session.execute(text(f"""
            SELECT e.sentiment, COUNT(*) as count
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            {location_filter}
            GROUP BY e.sentiment
        """), params)
        sentiment_data = {row.sentiment: row.count for row in result.fetchall()}
        
        # Rating distribution
        result = session.execute(text(f"""
            SELECT rating, COUNT(*) as count
            FROM reviews r {location_filter}
            GROUP BY rating ORDER BY rating
        """), params)
        rating_dist = {int(row.rating): row.count for row in result.fetchall()}
        
        # Top topics
        result = session.execute(text(f"""
            SELECT e.topics FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            {location_filter}
        """), params)
        topic_counter = Counter()
        for row in result.fetchall():
            if row.topics:
                topics = json.loads(row.topics)
                topic_counter.update(topics)
    
    return {
        "total_reviews": stats.total,
        "average_rating": round(stats.avg_rating, 2) if stats.avg_rating else 0,
        "sentiment_breakdown": sentiment_data,
        "rating_distribution": rating_dist,
        "top_topics": [{"topic": t, "count": c} for t, c in topic_counter.most_common(10)],
        "generated_at": datetime.now().isoformat()
    }


@app.get("/api/dashboard/trends")
async def get_trends(location_id: Optional[str] = None, period: str = "month"):
    """Get rating and sentiment trends over time"""
    from sqlalchemy import text
    
    # Date grouping based on period (PostgreSQL syntax)
    if period == "day":
        date_format = "DATE(r.review_date)"
    elif period == "week":
        date_format = "TO_CHAR(r.review_date::date, 'IYYY-IW')"
    else:  # month
        date_format = "TO_CHAR(r.review_date::date, 'YYYY-MM')"
    
    location_filter = "WHERE r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}
    
    with db.get_session() as session:
        # Rating trends
        result = session.execute(text(f"""
            SELECT {date_format} as period, 
                   AVG(r.rating) as avg_rating,
                   COUNT(*) as review_count
            FROM reviews r {location_filter}
            GROUP BY {date_format}
            ORDER BY period
        """), params)
        
        rating_trends = [
            {"period": row.period, "avg_rating": round(float(row.avg_rating), 2), "count": row.review_count}
            for row in result.fetchall() if row.period
        ]
        
        # Sentiment trends
        result = session.execute(text(f"""
            SELECT {date_format} as period,
                   e.sentiment,
                   COUNT(*) as count
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            {location_filter}
            GROUP BY {date_format}, e.sentiment
            ORDER BY period
        """), params)
        
        sentiment_by_period = {}
        for row in result.fetchall():
            if row.period:
                if row.period not in sentiment_by_period:
                    sentiment_by_period[row.period] = {}
                sentiment_by_period[row.period][row.sentiment] = row.count
    
    return {
        "rating_trends": rating_trends,
        "sentiment_trends": [{"period": p, **s} for p, s in sentiment_by_period.items()]
    }


@app.get("/api/dashboard/topics")
async def get_topic_analysis(location_id: Optional[str] = None):
    """Get detailed topic analysis with sentiment correlation"""
    from sqlalchemy import text
    
    location_filter = "WHERE r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}
    
    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT e.topics, e.sentiment, e.sentiment_score, r.rating
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            {location_filter}
        """), params)
        
        topic_stats = {}
        for row in result.fetchall():
            if row.topics:
                topics = json.loads(row.topics)
                for topic in topics:
                    if topic not in topic_stats:
                        topic_stats[topic] = {
                            "count": 0, "ratings": [], "sentiments": [],
                            "positive": 0, "negative": 0, "neutral": 0
                        }
                    topic_stats[topic]["count"] += 1
                    topic_stats[topic]["ratings"].append(row.rating)
                    topic_stats[topic][row.sentiment] += 1
    
    # Calculate averages and format response
    result = []
    for topic, stats in topic_stats.items():
        avg_rating = sum(stats["ratings"]) / len(stats["ratings"]) if stats["ratings"] else 0
        result.append({
            "topic": topic,
            "count": stats["count"],
            "avg_rating": round(avg_rating, 2),
            "sentiment_split": {
                "positive": stats["positive"],
                "negative": stats["negative"],
                "neutral": stats["neutral"]
            }
        })
    
    # Sort by count descending
    result.sort(key=lambda x: x["count"], reverse=True)
    return {"topics": result}


@app.get("/api/dashboard/reviews-by-topic/{topic}")
async def get_reviews_by_topic(topic: str, location_id: Optional[str] = None, limit: int = 20):
    """Get reviews filtered by a specific topic"""
    from sqlalchemy import text
    
    location_filter = "AND r.location_id = :loc" if location_id else ""
    params = {"topic_pattern": f'%"{topic}"%', "limit": limit}
    if location_id:
        params["loc"] = location_id
    
    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT r.id, r.location_id, r.source, r.review_id, r.rating, r.review_text,
                   r.reviewer_name, r.reviewer_type, r.review_date, r.relative_date,
                   e.sentiment, e.sentiment_score, e.topics, e.entities
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            WHERE e.topics LIKE :topic_pattern {location_filter}
            ORDER BY r.review_date DESC
            LIMIT :limit
        """), params)
        
        reviews = []
        for row in result.fetchall():
            review = {
                "id": row.id, "location_id": row.location_id, "source": row.source,
                "review_id": row.review_id, "rating": row.rating, "review_text": row.review_text,
                "reviewer_name": row.reviewer_name, "reviewer_type": row.reviewer_type,
                "review_date": row.review_date, "relative_date": row.relative_date,
                "sentiment": row.sentiment, "sentiment_score": row.sentiment_score,
                "topics": json.loads(row.topics) if row.topics else [],
                "entities": json.loads(row.entities) if row.entities else []
            }
            reviews.append(review)
    
    return {"topic": topic, "reviews": reviews, "count": len(reviews)}


@app.get("/api/dashboard/sentiment-details")
async def get_sentiment_details(location_id: Optional[str] = None):
    """Get detailed sentiment analysis with score distribution"""
    from sqlalchemy import text
    
    location_filter = "WHERE r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}
    
    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT e.sentiment, e.sentiment_score, r.rating, r.review_text, r.review_id
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            {location_filter}
        """), params)
        
        sentiment_groups = {"positive": [], "negative": [], "neutral": []}
        score_distribution = []
        
        for row in result.fetchall():
            sentiment = row.sentiment
            score_distribution.append(row.sentiment_score)
            if sentiment in sentiment_groups:
                sentiment_groups[sentiment].append({
                    "review_id": row.review_id,
                    "score": row.sentiment_score,
                    "rating": row.rating,
                    "text": row.review_text
                })
    
    # Get most extreme examples
    for sentiment in sentiment_groups:
        sentiment_groups[sentiment].sort(key=lambda x: abs(x['score']), reverse=True)
        sentiment_groups[sentiment] = sentiment_groups[sentiment][:5]  # Top 5 examples
    
    return {
        "summary": {
            "positive_count": len([s for s in score_distribution if s > 0.3]),
            "negative_count": len([s for s in score_distribution if s < -0.3]),
            "neutral_count": len([s for s in score_distribution if -0.3 <= s <= 0.3]),
            "avg_score": round(sum(score_distribution) / len(score_distribution), 2) if score_distribution else 0
        },
        "examples": sentiment_groups
    }


@app.get("/api/dashboard/recent-reviews")
async def get_recent_reviews(location_id: Optional[str] = None, limit: int = 10):
    """Get most recent reviews with enrichments"""
    from sqlalchemy import text
    
    location_filter = "WHERE r.location_id = :loc" if location_id else ""
    params = {"limit": limit}
    if location_id:
        params["loc"] = location_id
    
    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT r.id, r.location_id, r.source, r.review_id, r.rating, r.review_text,
                   r.reviewer_name, r.reviewer_type, r.review_date, r.relative_date,
                   e.sentiment, e.sentiment_score, e.topics, e.entities
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            {location_filter}
            ORDER BY r.review_date DESC
            LIMIT :limit
        """), params)
        
        reviews = []
        for row in result.fetchall():
            review = {
                "id": row.id, "location_id": row.location_id, "source": row.source,
                "review_id": row.review_id, "rating": row.rating, "review_text": row.review_text,
                "reviewer_name": row.reviewer_name, "reviewer_type": row.reviewer_type,
                "review_date": row.review_date, "relative_date": row.relative_date,
                "sentiment": row.sentiment, "sentiment_score": row.sentiment_score,
                "topics": json.loads(row.topics) if row.topics else [],
                "entities": json.loads(row.entities) if row.entities else []
            }
            reviews.append(review)
    
    return {"reviews": reviews}


# Predefined queries for highlight detection
HIGHLIGHT_QUERIES = [
    {"query": "long wait times delays queue complaints", "topic": "wait_times", "label": "Wait Times"},
    {"query": "rude staff unhelpful bad customer service", "topic": "staff_behavior", "label": "Staff Behavior"},
    {"query": "dirty car vehicle condition mechanical issues", "topic": "vehicle_condition", "label": "Vehicle Condition"},
    {"query": "hidden fees overcharges unexpected billing costs", "topic": "pricing_fees", "label": "Pricing & Fees"},
    {"query": "reservation not honored booking cancelled", "topic": "reservation_issues", "label": "Reservation Issues"},
    {"query": "cleanliness dirty smell unclean", "topic": "cleanliness", "label": "Cleanliness"},
]


@app.get("/api/dashboard/highlight")
async def get_highlight(location_id: Optional[str] = None):
    """Get the most critical complaint highlight for the dashboard alert"""
    from utils.bedrock import BedrockClient
    from sqlalchemy import text
    
    bedrock = BedrockClient()
    
    # Gather complaint stats for each topic
    location_filter = "AND r.location_id = :loc" if location_id else ""
    base_params = {"loc": location_id} if location_id else {}
    
    topic_complaints = []
    
    with db.get_session() as session:
        for hq in HIGHLIGHT_QUERIES:
            topic = hq["topic"]
            params = {"topic_pattern": f'%"{topic}"%', **base_params}
            
            # Count negative reviews for this topic
            result = session.execute(text(f"""
                SELECT COUNT(*) as count, AVG(e.sentiment_score) as avg_score
                FROM reviews r
                JOIN enrichments e ON r.review_id = e.review_id
                WHERE e.sentiment = 'negative'
                AND e.topics LIKE :topic_pattern
                {location_filter}
            """), params)
            
            row = result.fetchone()
            if row and row.count > 0:
                topic_complaints.append({
                    "topic": topic,
                    "label": hq["label"],
                    "query": hq["query"],
                    "count": row.count,
                    "avg_score": row.avg_score or 0
                })
        
        # No complaints found
        if not topic_complaints:
            return {"highlight": None, "generated_at": datetime.now().isoformat()}
        
        # Sort by count * severity (more negative score = higher priority)
        topic_complaints.sort(key=lambda x: x['count'] * abs(x['avg_score']), reverse=True)
        top_complaint = topic_complaints[0]
        
        # Get a sample review for this topic
        params = {"topic_pattern": f'%"{top_complaint["topic"]}"%', **base_params}
        result = session.execute(text(f"""
            SELECT r.review_text, r.rating, e.sentiment_score
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            WHERE e.sentiment = 'negative'
            AND e.topics LIKE :topic_pattern
            {location_filter}
            ORDER BY e.sentiment_score ASC
            LIMIT 1
        """), params)
        
        sample_row = result.fetchone()
    
    sample_quote = ""
    sample_rating = None
    if sample_row:
        sample_quote = sample_row.review_text
        sample_rating = sample_row.rating
    
    # Determine severity
    count = top_complaint['count']
    avg_score = top_complaint['avg_score']
    if count >= 10 or avg_score < -0.7:
        severity = "high"
    elif count >= 5 or avg_score < -0.5:
        severity = "medium"
    else:
        severity = "low"
    
    # Generate dynamic headline using LLM
    headline_prompt = f"""Generate a short, urgent alert headline (max 10 words) for a car rental dashboard.
Topic: {top_complaint['label']}
Complaint count: {count}
Sample complaint: {sample_quote}

Return ONLY the headline text, no quotes or explanation. Make it actionable and specific."""

    headline = bedrock.invoke(headline_prompt, max_tokens=50, temperature=0.7).strip().strip('"')
    
    # Fallback headline if LLM fails
    if not headline:
        headline = f"{top_complaint['label']} issues need attention"
    
    location_context = f" at {location_id}" if location_id else ""
    analysis_query = f"Show me all complaints about {top_complaint['label'].lower()}{location_context}"
    
    return {
        "highlight": {
            "headline": headline,
            "description": f"{count} complaints identified",
            "severity": severity,
            "topic": top_complaint['topic'],
            "topic_label": top_complaint['label'],
            "complaint_count": count,
            "sample_quote": sample_quote,
            "sample_rating": sample_rating,
            "analysis_query": analysis_query
        },
        "generated_at": datetime.now().isoformat()
    }


# ============ COMPETITIVE ANALYSIS APIs ============

from ingestion.s3_source import OUR_BRANDS


@app.get("/api/competitive/summary")
async def get_competitive_summary(location_id: Optional[str] = None):
    """
    Market overview: our brands vs competitors at a location.
    Returns avg rating, review count, and sentiment breakdown per brand.
    """
    from sqlalchemy import text

    location_filter = "AND r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}

    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT
                r.brand,
                r.is_competitor,
                COUNT(*) as review_count,
                AVG(r.rating) as avg_rating,
                SUM(CASE WHEN e.sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN e.sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN e.sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM reviews r
            LEFT JOIN enrichments e ON r.review_id = e.review_id
            WHERE r.brand IS NOT NULL {location_filter}
            GROUP BY r.brand, r.is_competitor
            ORDER BY avg_rating DESC
        """), params)

        brands = []
        for row in result.fetchall():
            total = row.positive + row.negative + row.neutral
            brands.append({
                "brand": row.brand,
                "is_competitor": row.is_competitor,
                "review_count": row.review_count,
                "avg_rating": round(float(row.avg_rating), 2) if row.avg_rating else None,
                "sentiment": {
                    "positive": row.positive,
                    "negative": row.negative,
                    "neutral": row.neutral,
                    "positive_pct": round(row.positive / total * 100, 1) if total else 0,
                },
            })

    our = [b for b in brands if not b["is_competitor"]]
    competitors = [b for b in brands if b["is_competitor"]]

    return {
        "location_id": location_id,
        "our_brands": our,
        "competitors": competitors,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/api/competitive/topics")
async def get_competitive_topics(location_id: Optional[str] = None, brand: Optional[str] = None):
    """
    Compare topic distribution across brands.
    Shows what customers talk about for each brand — great for gap analysis.
    """
    from sqlalchemy import text

    filters = ["r.brand IS NOT NULL"]
    params = {}
    if location_id:
        filters.append("r.location_id = :loc")
        params["loc"] = location_id
    if brand:
        filters.append("r.brand = :brand")
        params["brand"] = brand.lower()

    where = " AND ".join(filters)

    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT r.brand, r.is_competitor, e.topics, e.sentiment
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            WHERE {where}
        """), params)

        from collections import Counter
        brand_topics: Dict[str, Counter] = {}
        brand_sentiment_topics: Dict[str, Dict[str, Counter]] = {}

        for row in result.fetchall():
            b = row.brand
            if b not in brand_topics:
                brand_topics[b] = Counter()
                brand_sentiment_topics[b] = {"positive": Counter(), "negative": Counter()}
            topics = json.loads(row.topics) if row.topics else []
            brand_topics[b].update(topics)
            if row.sentiment in ("positive", "negative"):
                brand_sentiment_topics[b][row.sentiment].update(topics)

        comparison = []
        for b, counter in brand_topics.items():
            comparison.append({
                "brand": b,
                "is_competitor": b.lower() not in OUR_BRANDS,
                "top_topics": [{"topic": t, "count": c} for t, c in counter.most_common(10)],
                "top_complaints": [{"topic": t, "count": c} for t, c in brand_sentiment_topics[b]["negative"].most_common(5)],
                "top_praise": [{"topic": t, "count": c} for t, c in brand_sentiment_topics[b]["positive"].most_common(5)],
            })

    return {"location_id": location_id, "brands": comparison}


@app.get("/api/competitive/trends")
async def get_competitive_trends(location_id: Optional[str] = None, period: str = "month"):
    """
    Rating trends over time per brand — spot who's improving or declining.
    """
    from sqlalchemy import text

    if period == "month":
        date_trunc = "SUBSTRING(r.review_date, 1, 7)"  # YYYY-MM
    else:
        date_trunc = "SUBSTRING(r.review_date, 1, 10)"  # YYYY-MM-DD

    location_filter = "AND r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}

    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT
                r.brand,
                r.is_competitor,
                {date_trunc} as period,
                COUNT(*) as review_count,
                AVG(r.rating) as avg_rating
            FROM reviews r
            WHERE r.brand IS NOT NULL AND r.review_date IS NOT NULL {location_filter}
            GROUP BY r.brand, r.is_competitor, {date_trunc}
            ORDER BY period, r.brand
        """), params)

        trends = {}
        for row in result.fetchall():
            b = row.brand
            if b not in trends:
                trends[b] = {"brand": b, "is_competitor": row.is_competitor, "periods": []}
            trends[b]["periods"].append({
                "period": row.period,
                "review_count": row.review_count,
                "avg_rating": round(float(row.avg_rating), 2) if row.avg_rating else None,
            })

    return {"location_id": location_id, "brands": list(trends.values())}


@app.get("/api/competitive/gap-analysis")
async def get_gap_analysis(location_id: Optional[str] = None):
    """
    Identifies topics where competitors outperform us and vice versa.
    Returns actionable gaps: where we're weaker and where we're stronger.
    """
    from sqlalchemy import text

    location_filter = "AND r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}

    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT r.is_competitor, e.topics, e.sentiment, e.sentiment_score
            FROM reviews r
            JOIN enrichments e ON r.review_id = e.review_id
            WHERE r.brand IS NOT NULL {location_filter}
        """), params)

        from collections import defaultdict
        # topic -> { ours: {pos, neg, scores}, competitor: {pos, neg, scores} }
        topic_data = defaultdict(lambda: {
            "ours": {"positive": 0, "negative": 0, "scores": []},
            "competitor": {"positive": 0, "negative": 0, "scores": []},
        })

        for row in result.fetchall():
            bucket = "competitor" if row.is_competitor else "ours"
            topics = json.loads(row.topics) if row.topics else []
            for topic in topics:
                if row.sentiment == "positive":
                    topic_data[topic][bucket]["positive"] += 1
                elif row.sentiment == "negative":
                    topic_data[topic][bucket]["negative"] += 1
                if row.sentiment_score is not None:
                    topic_data[topic][bucket]["scores"].append(row.sentiment_score)

    gaps = []
    for topic, data in topic_data.items():
        ours_avg = (sum(data["ours"]["scores"]) / len(data["ours"]["scores"])) if data["ours"]["scores"] else 0
        comp_avg = (sum(data["competitor"]["scores"]) / len(data["competitor"]["scores"])) if data["competitor"]["scores"] else 0
        ours_total = data["ours"]["positive"] + data["ours"]["negative"]
        comp_total = data["competitor"]["positive"] + data["competitor"]["negative"]

        if ours_total < 2 and comp_total < 2:
            continue

        gap_score = comp_avg - ours_avg  # positive = competitor is better
        gaps.append({
            "topic": topic,
            "our_avg_sentiment": round(ours_avg, 3),
            "competitor_avg_sentiment": round(comp_avg, 3),
            "gap_score": round(gap_score, 3),
            "our_mentions": ours_total,
            "competitor_mentions": comp_total,
            "our_positive_pct": round(data["ours"]["positive"] / ours_total * 100, 1) if ours_total else 0,
            "competitor_positive_pct": round(data["competitor"]["positive"] / comp_total * 100, 1) if comp_total else 0,
        })

    # Sort: biggest gaps where competitors beat us first
    gaps.sort(key=lambda x: x["gap_score"], reverse=True)

    weaknesses = [g for g in gaps if g["gap_score"] > 0]
    strengths = [g for g in gaps if g["gap_score"] < 0]
    strengths.sort(key=lambda x: x["gap_score"])

    return {
        "location_id": location_id,
        "weaknesses": weaknesses[:10],
        "strengths": strengths[:10],
        "all_gaps": gaps,
    }


@app.get("/api/competitive/market-position")
async def get_market_position(location_id: Optional[str] = None):
    """
    Market share and positioning: review volume and rating rank per brand at a location.
    """
    from sqlalchemy import text

    location_filter = "AND r.location_id = :loc" if location_id else ""
    params = {"loc": location_id} if location_id else {}

    with db.get_session() as session:
        result = session.execute(text(f"""
            SELECT
                r.brand,
                r.is_competitor,
                COUNT(*) as review_count,
                AVG(r.rating) as avg_rating,
                MIN(r.review_date) as earliest_review,
                MAX(r.review_date) as latest_review
            FROM reviews r
            WHERE r.brand IS NOT NULL {location_filter}
            GROUP BY r.brand, r.is_competitor
            ORDER BY review_count DESC
        """), params)

        brands = []
        total_reviews = 0
        for row in result.fetchall():
            total_reviews += row.review_count
            brands.append({
                "brand": row.brand,
                "is_competitor": row.is_competitor,
                "review_count": row.review_count,
                "avg_rating": round(float(row.avg_rating), 2) if row.avg_rating else None,
                "earliest_review": row.earliest_review,
                "latest_review": row.latest_review,
            })

        # Add market share and rank
        for i, b in enumerate(sorted(brands, key=lambda x: x["avg_rating"] or 0, reverse=True)):
            b["rating_rank"] = i + 1
        for b in brands:
            b["review_share_pct"] = round(b["review_count"] / total_reviews * 100, 1) if total_reviews else 0

    return {
        "location_id": location_id,
        "total_reviews": total_reviews,
        "brands": brands,
    }


# ============ INGESTION APIs ============

# Configuration - can be overridden via environment variables
import os
S3_BUCKET = os.getenv("REVIEWS_S3_BUCKET", "review-intelligence-data")
S3_PREFIX = os.getenv("REVIEWS_S3_PREFIX", "reviews/")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")


class ProcessFilesRequest(BaseModel):
    s3_keys: List[str]
    enrich: bool = True


def run_ingestion_job(job_id: str, s3_keys: List[str], enrich: bool):
    """Background task to run ingestion pipeline"""
    from ingestion.pipeline import IngestionPipeline
    
    with jobs_lock:
        ingestion_jobs[job_id]["status"] = "running"
        ingestion_jobs[job_id]["started_at"] = datetime.now().isoformat()
    
    logger.start(f"[Job {job_id}] Starting background ingestion for {len(s3_keys)} files...")
    
    try:
        pipeline = IngestionPipeline(S3_BUCKET, S3_PREFIX, S3_REGION)
        results = pipeline.process_files(s3_keys, enrich=enrich)
        
        successful = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] == 'failed']
        
        with jobs_lock:
            ingestion_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "results": results,
                "summary": {
                    "total": len(results),
                    "successful": len(successful),
                    "failed": len(failed),
                    "total_reviews": sum(r['reviews_count'] for r in successful),
                    "total_enriched": sum(r['enriched_count'] for r in successful)
                }
            })
        
        logger.complete(f"[Job {job_id}] Completed: {len(successful)} successful, {len(failed)} failed")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}")
        with jobs_lock:
            ingestion_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e)
            })


@app.get("/api/ingestion/pending")
async def get_pending_files():
    """
    List S3 files that haven't been processed yet.
    
    Compares files in the configured S3 bucket against the ingestion tracking table.
    Files must follow naming convention: {LOCODE}_{SOURCE}_{DD}_{MM}_{YYYY}.json
    Example: LAX_google_10_01_2026.json, JFK_tripadvisor_15_02_2026.json
    """
    try:
        from ingestion.s3_source import S3ReviewSource
        logger.s3(f"Listing pending files from s3://{S3_BUCKET}/{S3_PREFIX}")
        s3_source = S3ReviewSource(S3_BUCKET, S3_PREFIX, S3_REGION)
        pending = s3_source.get_pending_files()
        logger.success(f"Found {len(pending)} pending files")
        
        return {
            "pending_files": pending,
            "count": len(pending),
            "bucket": S3_BUCKET,
            "prefix": S3_PREFIX
        }
    except Exception as e:
        logger.error(f"Error listing pending files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing pending files: {str(e)}")


@app.post("/api/ingestion/process")
async def process_files(request: ProcessFilesRequest, background_tasks: BackgroundTasks):
    """
    Process selected S3 files through the ingestion pipeline (runs in background).
    
    Returns immediately with a job_id. Use /api/ingestion/jobs/{job_id} to check status.
    
    Steps (run in background):
    1. Download JSON file from S3
    2. Validate file structure
    3. Parse and insert reviews into database
    4. Enrich reviews with LLM (sentiment, topics, entities, etc.) if enrich=true
    
    Request body:
    - s3_keys: Array of S3 keys to process
    - enrich: Whether to run LLM enrichment (default: true)
    """
    if not request.s3_keys:
        raise HTTPException(status_code=400, detail="No files specified for processing")
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    
    with jobs_lock:
        ingestion_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "s3_keys": request.s3_keys,
            "enrich": request.enrich,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "results": None,
            "summary": None,
            "error": None
        }
    
    # Run in background thread (not blocking the event loop)
    background_tasks.add_task(run_ingestion_job, job_id, request.s3_keys, request.enrich)
    
    logger.info(f"📋 Created ingestion job {job_id} for {len(request.s3_keys)} files")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Ingestion job created for {len(request.s3_keys)} files. Use /api/ingestion/jobs/{job_id} to check status.",
        "files_count": len(request.s3_keys)
    }


@app.get("/api/ingestion/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an ingestion job"""
    with jobs_lock:
        job = ingestion_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job


@app.get("/api/ingestion/jobs")
async def list_jobs():
    """List all ingestion jobs"""
    with jobs_lock:
        jobs = list(ingestion_jobs.values())
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return {
        "jobs": jobs,
        "count": len(jobs)
    }


@app.get("/api/ingestion/history")
async def get_ingestion_history(limit: int = Query(50, le=200)):
    """Get recent ingestion history showing processed files and their status"""
    history = db.get_ingestion_history(limit)
    return {"history": history, "count": len(history)}


@app.get("/api/ingestion/status/{s3_key:path}")
async def get_ingestion_status(s3_key: str):
    """Get the ingestion status of a specific S3 file"""
    record = db.get_ingestion_file(s3_key)
    if not record:
        return {"status": "not_found", "s3_key": s3_key}
    return record


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
