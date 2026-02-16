import boto3
import json
import os
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class BedrockClient:
    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.region = region
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.agent_client = boto3.client('bedrock-agent-runtime', region_name=region)
        self.kb_id = '4EJ0BSHUTO'
        logger.start(f"Bedrock client initialized (region={region}, model={model_id})")

    def retrieve(self, query: str, limit: int = 5, location_id: str = None) -> List[Dict]:
        """Retrieve relevant chunks from Bedrock Knowledge Base"""
        if not self.kb_id:
            raise ValueError("BEDROCK_KB_ID environment variable not set")

        try:
            # Fetch more results if filtering, to ensure we get enough after filter
            fetch_limit = limit * 3 if location_id else limit

            response = self.agent_client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': fetch_limit
                    }
                }
            )

            results = []
            for item in response.get('retrievalResults', []):
                text = item.get('content', {}).get('text', '')
                location = item.get('location', {})

                # Filter by location_id if provided (check S3 URI for location)
                if location_id:
                    s3_uri = location.get('s3Location', {}).get('uri', '')
                    if location_id.upper() not in s3_uri.upper():
                        continue

                # Fix UTF-16 encoding issue - remove null bytes
                if '\x00' in text:
                    text = text.replace('\x00', '')

                results.append({
                    'text': text,
                    'score': item.get('score', 0.0),
                    'metadata': item.get('metadata', {}),
                    'location': location
                })

                # Stop once we have enough filtered results
                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(f"Error retrieving from Knowledge Base: {e}")
            return []

    def invoke(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7,
               return_raw: bool = False):
        """Invoke Bedrock model with a prompt

        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            return_raw: If True, returns full API response dict instead of just text

        Returns:
            str: Generated text (default)
            dict: Full response with raw_response, metadata, content_type (if return_raw=True)
        """
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

            if return_raw:
                return {
                    "raw_response": response_body,
                    "metadata": response.get('ResponseMetadata', {}),
                    "content_type": response.get('contentType', '')
                }

            return response_body['content'][0]['text']

        except Exception as e:
            logger.error(f"Error invoking Bedrock: {e}")
            return {} if return_raw else ""

    def extract_topics(self, review_text: str) -> List[str]:
        """Extract topics from review text"""
        prompt = f"""Analyze this car rental review and extract the main topics/themes.
            Choose from: wait_times, staff_behavior, vehicle_condition, pricing_fees, reservation_issues,
            customer_service, cleanliness, preferred_program, tolls_epass, insurance, location_access, shuttle_service, upgrade_downgrade, damage_claims.

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
        """Enrich multiple reviews in a single LLM call with enhanced metadata.
        Uses raw_json if available, falls back to review_text otherwise."""
        reviews_text = ""
        for idx, review in enumerate(reviews, 1):
            # Use raw_json if available, otherwise fall back to review_text
            raw_json = review.get('raw_json')
            if raw_json:
                # Pass the raw JSON as-is for LLM to read naturally
                reviews_text += f"""Review {idx} (ID: {review['review_id']}):
{raw_json}

"""
            else:
                # Fallback to parsed fields
                reviews_text += f"""Review {idx} (ID: {review['review_id']}, Rating: {review.get('rating')}):
{review.get('review_text') or ''}

"""

        prompt = f"""Analyze these car rental reviews and extract comprehensive insights for each.

Topics (choose applicable): wait_times, staff_behavior, vehicle_condition, pricing_fees, reservation_issues, customer_service, cleanliness, preferred_program, tolls_epass, insurance, location_access, shuttle_service, upgrade_downgrade, damage_claims

{reviews_text}

Return ONLY a JSON array with this exact structure:
[
  {{
    "review_id": "<id>",
    "topics": ["topic1", "topic2"],
    "sentiment": "positive|negative|neutral",
    "sentiment_score": 0.5,
    "entities": ["entity1"],
    "key_phrases": ["short phrase capturing main complaint/praise"],
    "urgency_level": "low|medium|high|critical",
    "actionable": true|false,
    "suggested_action": "brief action item if actionable, null otherwise"
  }}
]

Guidelines:
- urgency_level: critical=safety/legal issues, high=service failures affecting many, medium=individual complaints, low=minor feedback
- actionable: true if the review suggests a specific improvement opportunity
- key_phrases: 1-3 short phrases (max 8 words each) capturing the essence
"""

        logger.llm("Calling AWS Bedrock (Claude)...")
        response = self.invoke(prompt, max_tokens=4000, temperature=0.3)
        logger.llm("Response received, parsing JSON...")

        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.debug(f"Raw response: {response[:200]}...")
            return []

    def get_location_info(self, location_code: str) -> Dict:
        """Get location name and address from a location code (e.g., airport IATA code).
        
        Args:
            location_code: Location identifier (e.g., 'JFK', 'LAX', 'MCO')
            
        Returns:
            Dict with 'name' and 'address' keys
        """
        prompt = f"""Given the location code "{location_code}", identify what location this refers to (likely an airport IATA code or city code).

Return ONLY a JSON object with:
- "name": The full official name of the location (e.g., "John F. Kennedy International Airport")
- "address": The full address including city, state/region, and country (e.g., "Queens, NY 11430, USA")

If you cannot identify the location, return:
{{"name": "{location_code}", "address": "{location_code}"}}

Return ONLY the JSON object, no explanation."""

        response = self.invoke(prompt, max_tokens=200, temperature=0.1)
        try:
            result = json.loads(response)
            return {
                "name": result.get("name", location_code),
                "address": result.get("address", location_code)
            }
        except Exception as e:
            logger.error(f"Error parsing location info: {e}")
            return {"name": location_code, "address": location_code}
