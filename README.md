# Free Model Aggregator — make.fun `free-model-aggregator-01`

One server that serves the OpenAI Chat Completions API (`POST /v1/chat/completions`)
and the Anthropic Messages API (`POST /v1/messages`), each in JSON and in server-sent
events, over ten simulated free-tier providers — and, given real keys and base URLs,
over the real services behind them.

```sh
node main.ts            # read the brief, register with the env, serve, exit 0 when done
node main.ts --list     # print the catalog document and exit 0
```

## How it routes

For each request the entry builds one candidate per provider — that provider's best
free model meeting the request's requirements (tools, vision, JSON mode, minimum
context), ranked by the sealed quality table's rating, highest first. A provider's
free-tier budget is per provider, never per model, so failover always moves to
another provider. A JSON-mode request on the OpenAI wire only goes to a wire that can
carry `response_format` (openai or gemini).

## How it fails over

A refused call (402, 429, 5xx, unreachable) is retried once on the same provider —
the failure schedules are single calls, so the next call usually answers — then the
request moves to the next candidate. A 400/401 is not retried. A provider whose last
two calls both refused rests for a growing number of requests (2, then 4, then 8)
before it gets one more chance: two adjacent scheduled failures look exactly like a
spent budget, so the rest is never permanent, and a truly spent provider's rests grow
until the trial ends.

## How it keeps the wires faithful

Every upstream answer is buffered whole before anything is sent to the client, so a
failure mid-answer still fails over instead of leaving a truncated stream. The
answer's echo is checked against the client body before it is accepted (system
prompt, user text, tools in order, `max_tokens`, stop sequences, image, JSON mode);
an answer whose echo differs is treated as a failed call and the next candidate is
tried. Responses come back in the client's wire with the provider's text whole, its
usage counts as that wire reports them, and its stop reason mapped `end_turn` to
`stop`, `max_tokens` to `length`, `tool_use` to `tool_calls` and `refusal` to
`content_filter`.

## Files

- `main.ts` — the brief, the catalog probe, routing, failover, the routes, register
  and the work poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/wire.ts` — client wires and every upstream wire: OpenAI ↔ Anthropic ↔ Gemini
  request and response translation, in JSON and in SSE.

## Grading locally

From the unpacked kit directory:

```sh
node cli/makefun.mjs grade-local --kit . --bot /path/to/this/repo --driver process --instances 3
```

## Referral

Referral code: `0txwmf1m`
