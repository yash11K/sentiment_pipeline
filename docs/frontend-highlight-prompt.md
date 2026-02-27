# Frontend Integration: Streaming Highlight Component

## Context

The backend serves a `GET /api/dashboard/highlight/stream` SSE endpoint that streams an AI-generated problem briefing for a car rental location. There is also a `GET /api/dashboard/highlight` endpoint that returns cached results. The frontend should use the cached endpoint for initial load and the streaming endpoint for refresh — but the rendering component is the same, just progressively filled during streaming.

## API Contracts

### 1. Cached Highlight (initial load)

```
GET /api/dashboard/highlight?location_id={IATA_CODE}&brand={brand_name}
```

| Param | Type | Required | Description |
|---|---|---|---|
| `location_id` | string | Yes | IATA airport code (e.g. `JFK`, `LAX`) |
| `brand` | string | No | Filter to a specific brand (e.g. `avis`, `hertz`) |

Response when cached data exists:
```json
{
  "highlight": {
    "location_id": "JFK",
    "brand": "avis",
    "analysis": "The JFK airport Avis location has several critical problems...",
    "severity": "critical",
    "followup_questions": [
      "What specific time periods see the worst wait times at JFK?",
      "How does JFK Avis staffing compare to competitor locations?",
      "What reservation system improvements have been attempted?"
    ],
    "citations": [
      { "text": "Waited over 2 hours for a car I reserved weeks ago...", "location": {}, "metadata": {} }
    ]
  },
  "cached": true,
  "generated_at": "2026-02-23T14:30:00.000000"
}
```

Response when no data:
```json
{ "highlight": null, "generated_at": "2026-02-23T14:30:00.000000" }
```

### 2. Streaming Highlight (refresh / first-time generation)

```
GET /api/dashboard/highlight/stream?location_id={IATA_CODE}&brand={brand_name}
```

Same query params as above. Returns `text/event-stream` (Server-Sent Events).

#### SSE Event Sequence

Events arrive in this order:

```
data: {"type": "chunk", "text": "The JFK airport"}
data: {"type": "chunk", "text": " Avis location has several"}
data: {"type": "chunk", "text": " critical problems requiring immediate attention:\n\n"}
data: {"type": "chunk", "text": "**Excessive Wait Times**:"}
...more chunks...
data: {"type": "citation", "citations": [{"text": "Waited 2 hours...", "location": {}, "metadata": {}}]}
data: {"type": "citation", "citations": [{"text": "Only one staff member...", "location": {}, "metadata": {}}]}
data: {"type": "metadata", "severity": "critical", "followup_questions": ["...", "...", "..."]}
data: {"type": "done"}
```

| Event type | Payload | When it arrives |
|---|---|---|
| `chunk` | `{ "type": "chunk", "text": "..." }` | Repeatedly, as the LLM generates text |
| `citation` | `{ "type": "citation", "citations": [...] }` | One or more times during/after generation |
| `error` | `{ "type": "error", "message": "..." }` | If the KB call fails |
| `metadata` | `{ "type": "metadata", "severity": "critical\|warning\|info", "followup_questions": [...] }` | Once, after all chunks — contains the parsed structured data |
| `done` | `{ "type": "done" }` | Final event — stream is complete |

## Component Architecture

### State Model

```typescript
interface HighlightState {
  status: 'idle' | 'loading' | 'streaming' | 'complete' | 'error';
  analysis: string;           // accumulated text from chunks (or full text from cache)
  severity: 'critical' | 'warning' | 'info' | null;
  followupQuestions: string[];
  citations: Citation[];
  cached: boolean | null;     // true if the response was served from cache
  generatedAt: string | null; // ISO timestamp of when the analysis was generated
  errorMessage: string | null;
}
```

### Data Flow

```
Dashboard loads → user selects location
  → GET /api/dashboard/highlight?location_id=JFK (cached endpoint)
  → If cached: render immediately with full analysis, severity, followups
  → If null/error: show empty state or trigger streaming automatically

User clicks refresh button (or first-time with no cache)
  → Set status = 'streaming', clear analysis text
  → Connect to GET /api/dashboard/highlight/stream?location_id=JFK
  → On each 'chunk' event: append text to analysis, re-render (markdown)
  → On 'citation' events: accumulate into citations array
  → On 'metadata' event: set severity + followup questions, show severity badge
  → On 'done' event: set status = 'complete'
  → On 'error' event: set status = 'error', show error message

User changes location or brand filter
  → Reset state to idle
  → Abort any active stream (close EventSource)
  → Start fresh with cached endpoint for new location
```

### SSE Connection (JavaScript reference)

```javascript
function streamHighlight(locationId, brand, callbacks) {
  const params = new URLSearchParams({ location_id: locationId });
  if (brand) params.set('brand', brand);

  const url = `/api/dashboard/highlight/stream?${params}`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch (data.type) {
      case 'chunk':
        callbacks.onChunk(data.text);
        break;
      case 'citation':
        callbacks.onCitations(data.citations);
        break;
      case 'metadata':
        callbacks.onMetadata(data.severity, data.followup_questions);
        break;
      case 'done':
        callbacks.onComplete();
        eventSource.close();
        break;
      case 'error':
        callbacks.onError(data.message);
        eventSource.close();
        break;
    }
  };

  eventSource.onerror = () => {
    callbacks.onError('Connection lost');
    eventSource.close();
  };

  // Return close function so caller can abort on unmount or location change
  return () => eventSource.close();
}
```

## Component Requirements

### 1. Highlight Card

The primary component — a prominent card at the top of the Monitor dashboard.

**Layout:**
- Header row: location code badge (e.g. "JFK") + brand pill (if filtered) + severity indicator + refresh button
- Body: the `analysis` field rendered as markdown (contains `**bold**` section headers)
- Footer: followup question chips + "Generated {relative_time}" + cached indicator

**During streaming (`status === 'streaming'`):**
- The analysis text grows in real-time as chunks arrive — render markdown progressively
- Show a subtle typing/streaming indicator (blinking cursor or pulsing dot at the end of text)
- Severity badge appears blank or as a skeleton until the `metadata` event arrives
- Followup questions area shows skeleton loaders until `metadata` event
- Refresh button is disabled with a spinner

**After complete (`status === 'complete'`):**
- Full analysis rendered as markdown
- Severity badge fully styled
- Followup questions rendered as clickable chips
- Refresh button re-enabled
- Show "Just generated" indicator briefly, then fade to timestamp

### 2. Severity Styling

| Severity | Card accent | Badge color | Icon |
|---|---|---|---|
| `critical` | Red left border or red tint | Red background | Alert triangle / fire |
| `warning` | Amber left border | Amber/yellow background | Warning circle |
| `info` | Blue left border | Blue background | Info circle |

- During streaming, before `metadata` arrives, use a neutral/gray style
- When `metadata` arrives with severity, animate the transition to the correct color
- For `critical`, consider a subtle pulsing border animation to draw attention

### 3. Followup Questions

Display the 3 `followup_questions` as clickable chips/buttons below the analysis.

**Behavior on click:**
- Route the question to the Explore Mode chat interface
- Either switch the user to the Explore tab with the question pre-filled, or open an inline chat panel
- This bridges Monitor → Explore mode

**During streaming:** show 3 skeleton pill loaders in the followup area. Replace with actual questions when `metadata` arrives.

### 4. Citations (collapsible)

- Show as a collapsible "Sources ({count})" section at the bottom of the card
- Each citation shows the `text` field truncated to ~100 chars with expand on click
- Citations accumulate during streaming — update the count as they arrive
- Collapsed by default

### 5. States

| State | What to show |
|---|---|
| `idle` (no location) | Placeholder: "Select a location to see the highlight briefing" |
| `loading` (fetching cache) | Skeleton card |
| `streaming` | Progressive text + typing indicator + skeleton severity/followups |
| `complete` | Full card with all data |
| `error` | Error message with retry button: "Couldn't generate highlight. Try again." |

### 6. Abort Handling

- When the user changes location/brand while a stream is active, close the EventSource immediately
- When the component unmounts, close any active EventSource
- This prevents stale data from a previous location appearing in the card

## Markdown Rendering Notes

- The `analysis` field contains markdown with `**bold**` headers, `\n\n` paragraph breaks, and occasionally bullet points
- Use a markdown renderer (react-markdown, marked, etc.)
- During streaming, the markdown is partial — the renderer should handle incomplete markdown gracefully (e.g. an unclosed `**` mid-stream)
- Consider rendering on a slight debounce (every 50-100ms) rather than on every single chunk to avoid excessive re-renders

## Responsive Behavior

- On desktop: the highlight card spans the full width of the dashboard content area
- On mobile: stack the header elements vertically (location badge above severity badge)
- The analysis text should be scrollable if it exceeds a max-height (~400px), with a "Show more" expand option
- Followup question chips should wrap on narrow screens
