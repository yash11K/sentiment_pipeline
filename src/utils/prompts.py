"""Centralised LLM prompt templates.

Every prompt sent to Bedrock lives here so they can be reviewed,
versioned, and tweaked in one place.
"""

# ---------------------------------------------------------------------------
# Enrichment  (used by BedrockClient.enrich_reviews_batch)
# ---------------------------------------------------------------------------

ENRICH_BATCH_PROMPT = """\
Analyze these car rental reviews and extract comprehensive insights for each.

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

# ---------------------------------------------------------------------------
# Location info  (used by BedrockClient.get_location_info)
# ---------------------------------------------------------------------------

LOCATION_INFO_PROMPT = """\
Given the location code "{location_code}", identify what location this refers to (likely an airport IATA code or city code).

Return ONLY a JSON object with:
- "name": The full official name of the location (e.g., "John F. Kennedy International Airport")
- "address": The full address including city, state/region, and country (e.g., "Queens, NY 11430, USA")

If you cannot identify the location, return:
{{"name": "{location_code}", "address": "{location_code}"}}

Return ONLY the JSON object, no explanation."""

# ---------------------------------------------------------------------------
# Highlight briefing  (used by app.py highlight endpoints)
# ---------------------------------------------------------------------------

HIGHLIGHT_PROMPT_TEMPLATE = """\
Analyze customer reviews for the {brand_context} {location_context} car rental location.

Respond in this exact format:

OVERVIEW:
A 2-3 sentence summary of the location's overall customer experience. Mention the dominant themes (e.g. wait times, vehicle condition, staff behavior) without referencing individual reviews or specific cases.

PROBLEMS:
1. <emoji> <short heading>
<2-3 sentences describing the pattern. No individual complaints, dates, or names.>

2. <emoji> <short heading>
<2-3 sentences describing the pattern. No individual complaints, dates, or names.>

3. <emoji> <short heading>
<2-3 sentences describing the pattern. No individual complaints, dates, or names.>

FOLLOWUPS:
1. <question>
2. <question>
3. <question>

RULES:
- The OVERVIEW must not reference specific customers, dates, or individual cases. It should mention key topic areas only.
- Report a maximum of 3 problems.
- Prioritize critical issues (safety, legal, systemic failures). If fewer than 3 critical problems exist, include high-severity issues. If no critical or high issues exist, include medium-severity issues and begin the PROBLEMS section with: "No critical or high-severity problems identified."
- Each problem: an emoji prefix, a short heading, then 2-3 sentences summarizing the pattern. Do not reference individual complaints, specific dates, or customer names.
- Emoji prefix per severity: ⚠️ for critical, 🔶 for high/warning, ℹ️ for medium/info.
- Followup questions must address broad operational or systemic themes (e.g. staffing processes, training gaps, fleet maintenance policies). Do not reference specific complaints, time periods, or individual cases."""
