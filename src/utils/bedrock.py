import boto3
import json
import os
from typing import Dict, List, Optional
from config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class BedrockClient:
    def __init__(self, region: str = None, model_id: str = None):
        self.region = region or config.AWS_REGION
        self.model_id = model_id or config.BEDROCK_MODEL_ID
        self.client = boto3.client('bedrock-runtime', region_name=self.region)
        self.agent_client = boto3.client('bedrock-agent-runtime', region_name=self.region)
        self.kb_id = config.BEDROCK_KB_ID
        logger.start(f"Bedrock client initialized (region={self.region}, model={self.model_id}, kb={self.kb_id})")
    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences and handle truncated JSON arrays.

        Returns a string that is valid JSON — either the original content
        or a reconstructed array of the complete objects found before truncation.
        """
        import re

        # 1. Try standard fence extraction
        match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. Strip opening fence if present (truncated = no closing fence)
        open_match = re.search(r'```(?:json)?\s*\n?', text)
        content = text[open_match.end():] if open_match else text.strip()

        # 3. Try direct parse
        try:
            json.loads(content)
            return content
        except (json.JSONDecodeError, ValueError):
            pass

        # 4. Salvage complete objects from truncated array
        return BedrockClient._salvage_complete_objects(content)


    @staticmethod
    def _salvage_complete_objects(text: str) -> str:
        """Extract complete JSON objects from a truncated JSON array.

        Scans character-by-character tracking brace depth. Each time depth
        returns to 0 after a '{', that object is complete. Returns a valid
        JSON array string containing only the complete objects.
        """
        # Find the opening bracket
        start = text.find('[')
        if start == -1:
            return '[]'

        objects = []
        i = start + 1
        while i < len(text):
            # Skip whitespace and commas
            while i < len(text) and text[i] in ' \t\n\r,':
                i += 1
            if i >= len(text) or text[i] == ']':
                break
            if text[i] != '{':
                i += 1
                continue

            # Track brace depth to find complete object
            obj_start = i
            depth = 0
            in_string = False
            escape_next = False

            while i < len(text):
                ch = text[i]
                if escape_next:
                    escape_next = False
                elif ch == '\\' and in_string:
                    escape_next = True
                elif ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            objects.append(text[obj_start:i+1])
                            i += 1
                            break
                i += 1
            else:
                # Reached end of text without closing brace — incomplete object, discard
                break

        if not objects:
            return '[]'
        return '[' + ','.join(objects) + ']'


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


    def retrieve_and_generate(self, query: str) -> Dict:
        """Use Bedrock KB RetrieveAndGenerate API — retrieval + response in one call"""
        if not self.kb_id:
            raise ValueError("BEDROCK_KB_ID environment variable not set")

        # Cross-region inference profiles (us.*, eu.*, global.*) need account ID in ARN
        if self.model_id.split('.')[0] in ('us', 'eu', 'ap', 'global'):
            import boto3
            account_id = boto3.client('sts', region_name=self.region).get_caller_identity()['Account']
            model_arn = f"arn:aws:bedrock:{self.region}:{account_id}:inference-profile/{self.model_id}"
        else:
            model_arn = f"arn:aws:bedrock:{self.region}::foundation-model/{self.model_id}"

        try:
            response = self.agent_client.retrieve_and_generate(
                input={'text': query},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.kb_id,
                        'modelArn': model_arn,
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': 100
                            }
                        }
                    }
                }
            )

            answer = response.get('output', {}).get('text', '')
            session_id = response.get('sessionId', '')

            citations = []
            for citation in response.get('citations', []):
                for ref in citation.get('retrievedReferences', []):
                    citations.append({
                        'text': ref.get('content', {}).get('text', ''),
                        'location': ref.get('location', {}),
                        'metadata': ref.get('metadata', {})
                    })

            return {
                'answer': answer,
                'citations': citations,
                'session_id': session_id
            }

        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {e}")
            raise

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
            return json.loads(self._extract_json(response))
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
            return json.loads(self._extract_json(response))
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
            return json.loads(self._extract_json(response))
        except:
            return []

    def enrich_reviews_batch(self, reviews: List[Dict]) -> List[Dict]:
        """Enrich multiple reviews in a single LLM call with enhanced metadata.

        Accepts pre-trimmed review dicts containing only:
        review_id, rating, text, reviewer, relative_date.
        The caller (enricher) is responsible for trimming before calling this method.
        """
        reviews_text = ""
        for idx, review in enumerate(reviews, 1):
            reviews_text += f"""Review {idx} (ID: {review['review_id']}, Rating: {review.get('rating')}):
    {review.get('text') or ''}

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

    CRITICAL SENTIMENT RULES (follow strictly):
    - sentiment MUST be exactly one of: "positive", "negative", or "neutral" (NO other values)
    - DEFAULT TO NEGATIVE when in doubt - car rental reviews are typically complaints
    - Star rating is the PRIMARY signal:
      * 1-2 stars = ALWAYS "negative" (no exceptions)
      * 3 stars = "negative" unless content is clearly praising the service
      * 4 stars = "positive" unless there are specific complaints mentioned
      * 5 stars = "positive"
    - Content analysis (secondary):
      * ANY complaint, issue, problem, or frustration = lean toward "negative"
      * Words like "but", "however", "although" followed by issues = "negative"
      * If review mentions ANY negative experience, even with positives, classify as "negative"
      * Only classify as "positive" if the review is genuinely praising without complaints
    - "neutral" should be RARE:
      * Only use for purely factual statements with no opinion
      * Never use "neutral" for 1-3 star reviews
      * Never use "neutral" if ANY complaint is mentioned
    - sentiment_score: -1.0 to 1.0
      * 1-2 stars: -0.5 to -1.0
      * 3 stars: -0.3 to -0.6 (lean negative)
      * 4 stars: 0.3 to 0.6
      * 5 stars: 0.6 to 1.0
    - urgency_level: critical=safety/legal issues, high=service failures affecting many, medium=individual complaints, low=minor feedback
    - actionable: true if the review suggests a specific improvement opportunity
    - key_phrases: 1-3 short phrases (max 8 words each) capturing the essence
    """

        logger.llm("Calling AWS Bedrock (Claude)...")
        response = self.invoke(prompt, max_tokens=16000, temperature=0.3)
        logger.llm("Response received, parsing JSON...")

        try:
            return json.loads(self._extract_json(response))
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.error(f"Raw response: {response[:500]}")
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
            result = json.loads(self._extract_json(response))
            return {
                "name": result.get("name", location_code),
                "address": result.get("address", location_code)
            }
        except Exception as e:
            logger.error(f"Error parsing location info: {e}")
            return {"name": location_code, "address": location_code}
