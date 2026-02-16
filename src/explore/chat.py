from utils.bedrock import BedrockClient
from storage.db import Database
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)


class ChatEngine:
    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.bedrock = BedrockClient()
        logger.chat("Chat engine initialized")
    
    def chat(self, query: str, location_id: str = None, use_semantic: bool = True) -> Dict:
        logger.chat(f"Processing query: {query[:50]}...")
        if use_semantic:
            # Use Bedrock Knowledge Base for semantic search
            logger.llm("Using semantic search via Knowledge Base")
            kb_results = self.bedrock.retrieve(query, limit=12, location_id=location_id)
            context = self._build_context_from_kb(kb_results)
            citations = [{'text': r['text'], 'score': r['score'], 'location': r['location'], 'metadata': r['metadata']} for r in kb_results]
        else:
            logger.database("Using database search")
            reviews = self.db.get_reviews(location_id=location_id, limit=10)
            context = self._build_context_from_db(reviews)
            citations = [{'review_id': r['review_id'], 'text': r['review_text']} for r in reviews[:3]]
        
        prompt = f"""You are analyzing car rental reviews. Answer based on the provided reviews.

Reviews:
{context}

User Question: {query}

Provide a concise answer with specific examples from the reviews."""
        
        logger.llm("Generating response...")
        response = self.bedrock.invoke(prompt, max_tokens=500)
        logger.success("Response generated")
        return {
            'answer': response,
            'citations': citations,
            'source': 'knowledge_base' if use_semantic else 'database'
        }
    
    def _build_context_from_kb(self, results: List[Dict]) -> str:
        context_parts = []
        for i, result in enumerate(results[:12], 1):
            context_parts.append(f"Review {i}:\n{result['text']}\n")
        return "\n".join(context_parts)
    
    def _build_context_from_db(self, reviews: List[Dict]) -> str:
        context_parts = []
        for i, review in enumerate(reviews[:5], 1):
            context_parts.append(f"Review {i} (ID: {review['review_id']}, Rating: {review['rating']}):\n{review['review_text']}\n")
        return "\n".join(context_parts)
