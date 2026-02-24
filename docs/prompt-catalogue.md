# Prompt Catalogue

All LLM prompts live in `src/utils/prompts.py`. This document describes each template, its purpose, where it's called from, the placeholders it expects, and the response format the model should return.

## Quick Reference

| Constant | Called by | Placeholders | Response format |
|---|---|---|---|
| `ENRICH_BATCH_PROMPT` | `BedrockClient.enrich_reviews_batch` | `reviews_text` | JSON array |
| `LOCATION_INFO_PROMPT` | `BedrockClient.get_location_info` | `location_code` | JSON object |
| `HIGHLIGHT_PROMPT_TEMPLATE` | `app.py` highlight endpoints | `brand_context`, `location_context` | Structured text (OVERVIEW + PROBLEMS + FOLLOWUPS) |

---

## ENRICH_BATCH_PROMPT

Enriches a batch of reviews in a single LLM call. This is the primary enrichment prompt used during ingestion.

- **File:** `src/utils/bedrock.py` → `BedrockClient.enrich_reviews_batch()`
- **Placeholder:** `{reviews_text}` — pre-formatted block of reviews, each with ID, rating, and text
- **Model params:** `max_tokens=16000`, `temperature=0.3`
- **Expected response:** JSON array where each element contains:

```json
{
  "review_id": "<id>",
  "topics": ["topic1", "topic2"],
  "sentiment": "positive|negative|neutral",
  "sentiment_score": 0.5,
  "entities": ["entity1"],
  "key_phrases": ["short phrase"],
  "urgency_level": "low|medium|high|critical",
  "actionable": true,
  "suggested_action": "brief action item or null"
}
```

### Allowed topics

`wait_times`, `staff_behavior`, `vehicle_condition`, `pricing_fees`, `reservation_issues`, `customer_service`, `cleanliness`, `preferred_program`, `tolls_epass`, `insurance`, `location_access`, `shuttle_service`, `upgrade_downgrade`, `damage_claims`

### Sentiment rules embedded in the prompt

- Star rating is the primary signal (1-2 → negative, 5 → positive, 3 → lean negative, 4 → lean positive).
- Defaults to negative when ambiguous — car rental reviews skew toward complaints.
- `neutral` is reserved for purely factual statements with no opinion.
- `sentiment_score` ranges from -1.0 to 1.0, calibrated to the star rating.
- `urgency_level`: critical = safety/legal, high = widespread failures, medium = individual complaints, low = minor feedback.

### Exact prompt

```text
Analyze these car rental reviews and extract comprehensive insights for each.

    Topics (choose applicable): wait_times, staff_behavior, vehicle_condition, pricing_fees, reservation_issues, customer_service, cleanliness, preferred_program, tolls_epass, insurance, location_access, shuttle_service, upgrade_downgrade, damage_claims

    {reviews_text}

    Return ONLY a JSON array with this exact structure:
    [
      {
    "review_id": "<id>",
    "topics": ["topic1", "topic2"],
    "sentiment": "positive|negative|neutral",
    "sentiment_score": 0.5,
    "entities": ["entity1"],
    "key_phrases": ["short phrase capturing main complaint/praise"],
    "urgency_level": "low|medium|high|critical",
    "actionable": true|false,
    "suggested_action": "brief action item if actionable, null otherwise"
      }
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
```

---

## LOCATION_INFO_PROMPT

Resolves an IATA airport code (or city code) to a human-readable location name and address.

- **File:** `src/utils/bedrock.py` → `BedrockClient.get_location_info()`
- **Placeholder:** `{location_code}` — e.g. `JFK`, `LAX`, `MCO`
- **Model params:** `max_tokens=200`, `temperature=0.1`
- **Expected response:** JSON object:

```json
{"name": "John F. Kennedy International Airport", "address": "Queens, NY 11430, USA"}
```

Falls back to `{"name": "<code>", "address": "<code>"}` if the model can't identify the location.

### Exact prompt

```text
Given the location code "{location_code}", identify what location this refers to (likely an airport IATA code or city code).

Return ONLY a JSON object with:
- "name": The full official name of the location (e.g., "John F. Kennedy International Airport")
- "address": The full address including city, state/region, and country (e.g., "Queens, NY 11430, USA")

If you cannot identify the location, return:
{"name": "{location_code}", "address": "{location_code}"}

Return ONLY the JSON object, no explanation.
```

---

## HIGHLIGHT_PROMPT_TEMPLATE

Generates a compact problem briefing for a car rental location. Sent as the query to the Bedrock Knowledge Base via `retrieve_and_generate` (not `invoke`), so the KB retrieves relevant review chunks and the model synthesises them. Overview, problems, severity, and followup questions are all returned in a single structured response — no second LLM call needed.

- **File:** `src/api/app.py` → `get_highlight()` and `stream_highlight()`
- **Placeholders:**
  - `{brand_context}` — brand name or `"all brands at"`
  - `{location_context}` — e.g. `"JFK airport"`
- **Expected response:** Structured text with sections: OVERVIEW, PROBLEMS (max 3, emoji-prefixed), FOLLOWUPS

### Response structure

```
OVERVIEW:
2-3 sentence location summary mentioning dominant themes.

PROBLEMS:
1. ⚠️ Short Heading
2-3 sentences describing the pattern.

2. ⚠️ Short Heading
2-3 sentences describing the pattern.

3. 🔶 Short Heading
2-3 sentences describing the pattern.

FOLLOWUPS:
1. Broad operational question
2. Broad operational question
3. Broad operational question
```

### Severity via emoji

| Emoji | Severity | Meaning |
|---|---|---|
| ⚠️ | critical | Widespread systemic failures affecting most customers |
| 🔶 | warning | Recurring issues impacting a significant portion but not universal |
| ℹ️ | info | Isolated or minor complaints without a clear pattern |

The parser (`_parse_highlight_response`) derives the overall severity from the first emoji found in the response. Followup questions are extracted from the FOLLOWUPS section. Fallback defaults apply if parsing fails.

### Exact prompt

```text
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
- Followup questions must address broad operational or systemic themes (e.g. staffing processes, training gaps, fleet maintenance policies). Do not reference specific complaints, time periods, or individual cases.
```

---

## Adding a New Prompt

1. Define the template as a module-level constant in `src/utils/prompts.py`.
2. Use `{placeholder}` syntax for dynamic values — the caller formats with `.format()`.
3. Use `{{` and `}}` to escape literal braces in JSON examples within the template.
4. Import the constant in the calling module.
5. Add an entry to this document with the exact prompt text in a fenced code block.
