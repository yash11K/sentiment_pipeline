# Frontend Integration: Dashboard Highlight Component

## Context

The backend serves a `GET /api/dashboard/highlight` endpoint that returns an AI-generated problem briefing for a car rental location. This replaces the old stat-card highlight with a rich, KB-powered analysis. The frontend needs a component to display this on the Monitor Mode dashboard.

## API Contract

### Request
```
GET /api/dashboard/highlight?location_id={IATA_CODE}&brand={brand_name}&refresh={true|false}
```

| Param | Type | Required | Description |
|---|---|---|---|
| `location_id` | string | Yes | IATA airport code (e.g. `JFK`, `LAX`) |
| `brand` | string | No | Filter to a specific brand (e.g. `avis`, `hertz`) |
| `refresh` | boolean | No | `true` bypasses cache and regenerates from KB. Default `false` |

### Response (highlight exists)
```json
{
  "highlight": {
    "location_id": "JFK",
    "brand": "avis",
    "analysis": "The JFK airport Avis location has several critical problems requiring immediate attention:\n\n**Excessive Wait Times**: Customers consistently report waiting 1-3 hours just to pick up pre-reserved vehicles...\n\n**Severe Understaffing**: Multiple reviews indicate only one or two staff members working at counters during peak times...",
    "severity": "critical",
    "followup_questions": [
      "What specific time periods see the worst wait times at JFK?",
      "How does JFK Avis staffing compare to competitor locations?",
      "What reservation system improvements have been attempted?"
    ],
    "citations": [
      {
        "text": "Waited over 2 hours for a car I reserved weeks ago...",
        "location": {},
        "metadata": {}
      }
    ]
  },
  "cached": false,
  "generated_at": "2026-02-23T14:30:00.000000"
}
```

### Response (no location provided)
```json
{
  "highlight": null,
  "generated_at": "2026-02-23T14:30:00.000000"
}
```

### Error (KB failure)
```
HTTP 502 — { "detail": "Failed to generate highlight from Knowledge Base" }
```

## Component Requirements

### 1. Highlight Card

This is the primary component — a prominent card at the top of the Monitor dashboard for the selected location.

**Layout:**
- Header row: location code badge (e.g. "JFK") + brand pill (if filtered) + severity indicator
- Body: the `analysis` field rendered as markdown (it contains `**bold**` formatting for section headers)
- Footer: "Generated {relative_time}" + refresh button + cached indicator

**Severity styling:**
| Severity | Color | Icon suggestion |
|---|---|---|
| `critical` | Red/destructive | Alert triangle or fire |
| `warning` | Amber/yellow | Warning circle |
| `info` | Blue/neutral | Info circle |

The severity badge should be visually prominent — this is the first thing a manager sees.

**Refresh behavior:**
- A refresh/reload icon button in the card header or footer
- On click: re-call the endpoint with `refresh=true`
- Show a loading/spinner state while the KB generates (this can take 5-15 seconds)
- After refresh, update the card and show `cached: false` indicator briefly

**Cached indicator:**
- When `cached: true`, show a subtle "Cached" label or icon near the timestamp
- When `cached: false` (fresh generation), optionally show "Just generated" briefly

### 2. Followup Questions

Display the 3 `followup_questions` as clickable chips/buttons below the analysis.

**Behavior on click:**
- Route the question to the Explore Mode chat interface (the existing `/api/explore/chat` endpoint)
- If the app supports it, switch the user to Explore tab with the question pre-filled
- Alternatively, open a small inline chat panel or modal that sends the question and shows the response

This is the key UX bridge between Monitor and Explore modes — the highlight surfaces the problem, the followup questions let the manager dig deeper.

### 3. Citations (optional/collapsible)

The `citations` array contains source review snippets that the KB used to generate the analysis.

- Show as a collapsible "Sources" section at the bottom of the card
- Each citation shows the `text` field (truncated to ~100 chars with expand)
- This builds trust in the AI-generated analysis by showing the underlying evidence

### 4. Empty/Loading States

- **No location selected**: Don't render the highlight card, or show a placeholder "Select a location to see the highlight briefing"
- **Loading (first load or refresh)**: Skeleton loader or spinner with "Analyzing reviews..." text
- **Error (502)**: Show an error state with retry button: "Couldn't generate highlight. The Knowledge Base may be unavailable."
- **Null highlight**: If `highlight` is null, show "No highlight available for this location"

### 5. Integration Points

- The highlight should reload when the user changes the selected `location_id` or `brand` filter in the dashboard
- On initial dashboard load for a location, call with `refresh=false` (serve from cache)
- The refresh button is the only way to trigger `refresh=true` — don't auto-refresh on every page load

## Data Flow

```
Dashboard loads → user selects location
  → GET /api/dashboard/highlight?location_id=JFK (cached)
  → Render highlight card with analysis + severity + followups

User clicks refresh button
  → GET /api/dashboard/highlight?location_id=JFK&refresh=true
  → Show loading state (5-15s)
  → Re-render with fresh data

User clicks followup question
  → Navigate to Explore mode OR open inline chat
  → POST /api/explore/chat { "query": "<followup question>" }
  → Show chat response
```

## Design Notes

- The `analysis` field is markdown — use a markdown renderer (e.g. `react-markdown`, `marked`, or equivalent for your framework)
- The analysis can be 200-500 words — make sure the card handles long content gracefully (scrollable or expandable)
- Severity should drive the card's border/accent color, not just an icon
- The followup questions are the most important interactive element — make them visually inviting to click
- Consider adding a subtle animation when severity is `critical` (e.g. a pulsing border) to draw attention
