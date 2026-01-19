from utils.bedrock import BedrockClient
from storage.db import Database
from typing import Dict, List

class ChatEngine:
    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.bedrock = BedrockClient()
    
    def chat(self, query: str, location_id: str = None, use_semantic: bool = True) -> Dict:
        if use_semantic:
            # Use Bedrock Knowledge Base for semantic search
            kb_results = self.bedrock.retrieve(query, limit=5)
            context = self._build_context_from_kb(kb_results)
            citations = [{'text': r['text'][:150], 'score': r['score']} for r in kb_results[:3]]
        else:
            reviews = self.db.get_reviews(location_id=location_id, limit=10)
            context = self._build_context_from_db(reviews)
            citations = [{'review_id': r['review_id'], 'text': r['review_text'][:150]} for r in reviews[:3]]
        
        prompt = f"""You are analyzing car rental reviews. Answer based on the provided reviews.

Reviews:
{context}

User Question: {query}

Provide a concise answer with specific examples from the reviews."""
        
        response = self.bedrock.invoke(prompt, max_tokens=500)
        return {
            'answer': response,
            'citations': citations,
            'source': 'knowledge_base' if use_semantic else 'database'
        }
    
    def _build_context_from_kb(self, results: List[Dict]) -> str:
        context_parts = []
        for i, result in enumerate(results[:5], 1):
            context_parts.append(f"Review {i}:\n{result['text'][:500]}\n")
        return "\n".join(context_parts)
    
    def _build_context_from_db(self, reviews: List[Dict]) -> str:
        context_parts = []
        for i, review in enumerate(reviews[:5], 1):
            context_parts.append(f"Review {i} (ID: {review['review_id']}, Rating: {review['rating']}):\n{review['review_text'][:300]}\n")
        return "\n".join(context_parts)
