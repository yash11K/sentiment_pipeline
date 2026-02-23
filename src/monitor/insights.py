import json
from collections import Counter
from datetime import datetime
from storage.db import Database
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)


class InsightGenerator:
    def __init__(self, db: Database = None):
        self.db = db or Database()
        logger.insight("Insight generator initialized")
    
    def generate_insights(self, location_id: str, time_window: str = "all") -> Dict:
        logger.insight(f"Generating insights for {location_id} (window={time_window})")
        reviews = self.db.get_reviews(location_id=location_id, limit=10000)
        if not reviews:
            logger.warning(f"No reviews found for {location_id}")
            return {}
        
        insights = {
            'location_id': location_id,
            'time_window': time_window,
            'total_reviews': len(reviews),
            'rating_distribution': self._rating_distribution(reviews),
            'top_topics': self._top_topics(reviews),
            'key_drivers': self._key_drivers(reviews),
            'sentiment_breakdown': self._sentiment_breakdown(reviews),
            'representative_quotes': self._representative_quotes(reviews),
            'generated_at': datetime.now().isoformat()
        }
        self._cache_insights(insights)
        logger.success(f"Generated insights for {location_id}: {len(reviews)} reviews analyzed")
        return insights
    
    def _rating_distribution(self, reviews: List[Dict]) -> Dict:
        ratings = [r['rating'] for r in reviews if r.get('rating')]
        counter = Counter(ratings)
        return {'distribution': dict(counter), 'average': sum(ratings) / len(ratings) if ratings else 0, 'total': len(ratings)}
    
    def _top_topics(self, reviews: List[Dict]) -> List[Dict]:
        topic_counts = Counter()
        for review in reviews:
            enrichment = self.db.get_review_with_enrichment(review['review_id'])
            if enrichment and enrichment.get('topics'):
                topics = json.loads(enrichment['topics']) if isinstance(enrichment['topics'], str) else enrichment['topics']
                topic_counts.update(topics)
        return [{'topic': topic, 'count': count} for topic, count in topic_counts.most_common(10)]
    
    def _key_drivers(self, reviews: List[Dict]) -> List[Dict]:
        negative_topics = Counter()
        positive_topics = Counter()
        for review in reviews:
            enrichment = self.db.get_review_with_enrichment(review['review_id'])
            if enrichment:
                topics = json.loads(enrichment['topics']) if isinstance(enrichment.get('topics'), str) else enrichment.get('topics', [])
                sentiment = enrichment.get('sentiment', 'neutral')
                if sentiment == 'negative':
                    negative_topics.update(topics)
                elif sentiment == 'positive':
                    positive_topics.update(topics)
        return {
            'complaints': [{'topic': t, 'count': c} for t, c in negative_topics.most_common(5)],
            'praise': [{'topic': t, 'count': c} for t, c in positive_topics.most_common(5)]
        }
    
    def _sentiment_breakdown(self, reviews: List[Dict]) -> Dict:
        sentiments = []
        for review in reviews:
            enrichment = self.db.get_review_with_enrichment(review['review_id'])
            if enrichment and enrichment.get('sentiment'):
                sentiments.append(enrichment['sentiment'])
        return dict(Counter(sentiments))
    
    def _representative_quotes(self, reviews: List[Dict]) -> List[Dict]:
        quotes = []
        seen_topics = set()
        for review in reviews:
            enrichment = self.db.get_review_with_enrichment(review['review_id'])
            if enrichment:
                topics = json.loads(enrichment['topics']) if isinstance(enrichment.get('topics'), str) else enrichment.get('topics', [])
                for topic in topics:
                    if topic not in seen_topics and len(quotes) < 5:
                        quotes.append({
                            'topic': topic,
                            'quote': review['review_text'],
                            'rating': review['rating'],
                            'sentiment': enrichment.get('sentiment')
                        })
                        seen_topics.add(topic)
        return quotes
    
    def _cache_insights(self, insights: Dict):
        self.db.save_insights(
            location_id=insights['location_id'],
            time_window=insights['time_window'],
            insights={
                'top_topics': insights.get('top_topics', []),
                'key_drivers': insights.get('key_drivers', []),
                'representative_quotes': insights.get('representative_quotes', []),
                'generated_summary': json.dumps(insights),
            }
        )
    
    def get_cached_insights(self, location_id: str) -> Dict:
        result = self.db.get_cached_insights(location_id)
        if not result:
            return {}
        if result.get('generated_summary'):
            try:
                return json.loads(result['generated_summary'])
            except (json.JSONDecodeError, TypeError):
                pass
        return result
