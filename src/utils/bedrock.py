import boto3
import json
import os
from typing import Dict, List, Optional

class BedrockClient:
    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.region = region
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.agent_client = boto3.client('bedrock-agent-runtime', region_name=region)
        self.kb_id = os.getenv('BEDROCK_KB_ID')
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant chunks from Bedrock Knowledge Base"""
        if not self.kb_id:
            raise ValueError("BEDROCK_KB_ID environment variable not set")
        
        try:
            response = self.agent_client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': limit
                    }
                }
            )
            
            results = []
            for item in response.get('retrievalResults', []):
                results.append({
                    'text': item.get('content', {}).get('text', ''),
                    'score': item.get('score', 0.0),
                    'metadata': item.get('metadata', {}),
                    'location': item.get('location', {})
                })
            return results
        
        except Exception as e:
            print(f"Error retrieving from Knowledge Base: {e}")
            return []
    
    def invoke(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Invoke Bedrock model with a prompt"""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        
        except Exception as e:
            print(f"Error invoking Bedrock: {e}")
            return ""
    
    def extract_topics(self, review_text: str) -> List[str]:
        """Extract topics from review text"""
        prompt = f"""Analyze this car rental review and extract the main topics/themes.
Choose from: wait_times, staff_behavior, vehicle_condition, pricing_fees, reservation_issues, 
customer_service, cleanliness, preferred_program, tolls_epass, insurance, location_access.

Review: {review_text}

Return ONLY a JSON array of topics, e.g., ["wait_times", "staff_behavior"]"""
        
        response = self.invoke(prompt, max_tokens=100, temperature=0.3)
        try:
            return json.loads(response)
        except:
            return []
    
    def analyze_sentiment(self, review_text: str) -> Dict:
        """Analyze sentiment of review"""
        prompt = f"""Analyze the sentiment of this review.
Return ONLY a JSON object with "sentiment" (positive/negative/neutral) and "score" (-1 to 1).

Review: {review_text}

Format: {{"sentiment": "negative", "score": -0.8}}"""
        
        response = self.invoke(prompt, max_tokens=50, temperature=0.1)
        try:
            return json.loads(response)
        except:
            return {"sentiment": "neutral", "score": 0.0}
    
    def extract_entities(self, review_text: str) -> List[str]:
        """Extract named entities from review"""
        prompt = f"""Extract key entities from this review (employee names, programs like "Preferred", 
specific issues like "EZ Pass", locations, etc.).

Review: {review_text}

Return ONLY a JSON array of entities, e.g., ["Preferred", "EZ Pass"]"""
        
        response = self.invoke(prompt, max_tokens=100, temperature=0.3)
        try:
            return json.loads(response)
        except:
            return []
    
    def enrich_reviews_batch(self, reviews: List[Dict]) -> List[Dict]:
        """Enrich multiple reviews in a single LLM call"""
        reviews_text = ""
        for idx, review in enumerate(reviews, 1):
            reviews_text += f"""Review {idx} (ID: {review['review_id']}, Rating: {review['rating']}):
{review['review_text'][:500]}

"""
        
        prompt = f"""Analyze these car rental reviews and extract topics, sentiment, and entities for each.

Topics: wait_times, staff_behavior, vehicle_condition, pricing_fees, reservation_issues, customer_service, cleanliness, preferred_program, tolls_epass, insurance, location_access

{reviews_text}

Return ONLY a JSON array with this exact structure:
[
  {{
    "review_id": "<id>",
    "topics": ["topic1", "topic2"],
    "sentiment": "positive|negative|neutral",
    "sentiment_score": 0.5,
    "entities": ["entity1"]
  }}
]
"""
        
        print(f"  -> Calling AWS Bedrock (Claude)...")
        response = self.invoke(prompt, max_tokens=3000, temperature=0.3)
        print(f"  -> Response received, parsing JSON...")
        
        try:
            return json.loads(response)
        except Exception as e:
            print(f"  -> ERROR parsing response: {e}")
            print(f"  -> Raw response: {response[:200]}...")
            return []
