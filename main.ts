// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the brief (provider directory, sealed quality table, workload manifest),
// probes each provider's catalog route, registers one catalog with the env, then serves the
// env client's 20 requests, one at a time, each with the best free model that meets the
// request's requirements, failing over when a provider refuses, and exits 0 when the env
// says the workload is done.
//
// ROUTING (quality is 60% of the score): for each request the candidates are one per
// provider — that provider's best free model meeting the requirements — ordered by the
// quality table's rating, highest first. A provider's budget is per provider, never per
// model, so a second model on a refused provider cannot rescue a request; the next
// candidate is always another provider. A JSON-mode request on the OpenAI wire is only
// routed to a wire that can carry response_format (openai or gemini): an anthropic-wire
// provider echoes json_mode absent and would fail the echo check.
//
// FAILOVER: a refused call (402, 429, 5xx, unreachable) is retried once on the same
// provider — the failure schedules are single calls, so the next call usually answers —
// and then the request moves to the next candidate. A 400/401 is not retried: the same
// call would answer the same way. A provider whose last two calls both refused rests
// for a growing number of requests (2, then 4, then 8) before it gets one more chance:
// two adjacent scheduled failures look exactly like a spent budget, so the rest is
// never permanent, and a truly spent provider's rests grow until the trial ends.
//
// WIRE FIDELITY: every upstream answer is buffered whole before anything is sent to the
// client, so a failure mid-answer still fails over instead of leaving a truncated stream.
// The answer's echo is checked against the client body before it is accepted (system
// prompt, user text, tools in order, max_tokens, stop sequences, image, JSON mode): an
// answer whose echo differs is treated as a failed call and the next candidate is tried.
// Responses are translated back into the client's wire with the provider's text whole,
// its usage counts as that wire reports them, and its stop reason mapped end_turn to
// stop, max_tokens to length, tool_use to tool_calls and refusal to content_filter.
//
// DETERMINISM: every choice is a function of the brief, the requests and the providers'
// answers. No wall clock and no unseeded randomness anywhere in a route: the poll timer
// and the register retry only decide when the process exits, never what it answers.

import { createHash } from 'node:crypto';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type ProbedModel,
  type Probe,
  type ProviderDirectoryEntry,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  finishReasonOf,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseSseFrames,
  splitText,
  sseEvent,
  stopReasonOf,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How many times the register POST is retried before the trial is abandoned. */
const REGISTER_ATTEMPTS = 5;
/** Pause between register attempts, in milliseconds. */
const REGISTER_RETRY_MS = 500;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far, by provider id, for the log only. */
const callsMade = new Map<string, number>();
/**
 * Providers resting for this trial: a provider whose last two calls both refused
 * rests for a growing number of requests (2, then 4, then 8) before it gets one more
 * chance. A single refusal never rests a provider: the failure schedules are single
 * calls, so the next call usually answers. Resting, never a permanent mark, because
 * two adjacent scheduled failures look exactly like a spent budget — and a provider
 * wrongly rested for good would take its best model away from every later request,
 * while a rested one only costs the requests it sits out, which its neighbours serve
 * at the same rating or one step down. A truly spent provider's rests grow until the
 * trial ends, so the wasted probes stay few.
 */
const restUntil = new Map<string, number>();
/** Rest level per provider: the next rest lasts 2^level requests. */
const restLevel = new Map<string, number>();
/** Consecutive refused calls per provider, reset by any answer. */
const consecutiveFailures = new Map<string, number>();
/** Requests served so far, counted at the start of each serve. */
let requestsSeen = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** A catalog built from the brief alone, when no env answers the probes (the `--list` fallback). */
function catalogFromBrief(): CatalogReport {
  const providers: CatalogReport['providers'] = [];
  if (brief.ratings === null) return { providers };
  for (const provider of brief.directory.providers) {
    const rows = brief.ratings.ratings.filter((r) => r.provider === provider.id && r.free);
    const seen = new Set<string>();
    const models = rows
      .filter((r) => !seen.has(r.model) && (seen.add(r.model), true))
      .sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))
      .map((r) => ({ id: r.model, free: true, context_length: r.context_length }));
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (ENV_URL !== '') {
    probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
    for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  if (catalog.providers.length === 0 && brief.ratings !== null) catalog = catalogFromBrief();
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
}

/**
 * One candidate per provider: that provider's best free model meeting the requirements.
 * Ordered by rating, highest first, ties by provider id then model id. A JSON-mode request
 * on the OpenAI wire only considers wires that can carry response_format.
 */
function candidatesFor(requires: Requirement, jsonEcho: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if ((restUntil.get(probe.provider.id) ?? -1) > requestsSeen) continue;
    if (jsonEcho && probe.provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) {
        best = { probe, provider: probe.provider, model, rating };
      }
    }
    if (best !== null) out.push(best);
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

// ---------------------------------------------------------------------------
// The echo check: the provider echoes what it parsed, and the client compares
// that echo with what it sent. Mirror the env's parse here, so an answer whose
// echo differs is caught before it is served and the next candidate is tried.
// ---------------------------------------------------------------------------

function sha8Text(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}

function sha8Bytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex').slice(0, 8);
}

function imageShaOfUrl(url: string): string | null {
  if (url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return sha8Bytes(Buffer.from(data[1], 'base64'));
  return sha8Text(url);
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

interface Echo {
  system_present: boolean;
  system_sha8: string | null;
  user_sha8: string;
  tools: string[];
  max_tokens: number | null;
  stops: string[];
  image_sha8: string | null;
  json_mode: boolean;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** What the client sent, as the provider's echo will report it. */
function expectedEcho(api: 'openai' | 'anthropic', body: Json): Echo {
  if (api === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    let userText = '';
    let imageSha8: string | null = null;
    for (const message of messages) {
      if (message.role === 'system') {
        const text = blocksText(message.content);
        if (text !== '') systemParts.push(text);
      } else if (message.role === 'user') {
        userText = blocksText(message.content);
        if (Array.isArray(message.content)) {
          for (const block of message.content) {
            if (!isRecord(block)) continue;
            if (block.type === 'image_url' && isRecord(block.image_url)) imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
          }
        }
      }
    }
    const tools: string[] = [];
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    const systemText = systemParts.join('\n');
    return {
      system_present: systemText !== '',
      system_sha8: systemText === '' ? null : sha8Text(systemText),
      user_sha8: sha8Text(userText),
      tools,
      max_tokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      stops: stringList(body.stop),
      image_sha8: imageSha8,
      json_mode: format === 'json_object' || format === 'json_schema',
    };
  }
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') imageSha8 = sha8Bytes(Buffer.from(block.source.data, 'base64'));
        else if (block.source.type === 'url') imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
  }
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8Text(systemText),
    user_sha8: sha8Text(userText),
    tools,
    max_tokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    image_sha8: imageSha8,
    json_mode: false,
  };
}

/** The echo inside a provider answer's text, or null when the text carries none. */
function echoFromText(text: string): Echo | null {
  const prefix = 'ANSWER ECHO';
  if (!text.startsWith(prefix)) return null;
  let value: unknown;
  try {
    value = JSON.parse(text.slice(prefix.length));
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const keys = Object.keys(value).sort();
  const want = ['image_sha8', 'json_mode', 'max_tokens', 'stops', 'system_present', 'system_sha8', 'tools', 'user_sha8'];
  if (keys.length !== want.length || keys.some((key, i) => key !== want[i])) return null;
  if (typeof value.system_present !== 'boolean' || typeof value.json_mode !== 'boolean') return null;
  if (value.system_sha8 !== null && typeof value.system_sha8 !== 'string') return null;
  if (typeof value.user_sha8 !== 'string') return null;
  if (value.image_sha8 !== null && typeof value.image_sha8 !== 'string') return null;
  if (value.max_tokens !== null && numberOrNull(value.max_tokens) === null) return null;
  if (!Array.isArray(value.tools) || !value.tools.every((x) => typeof x === 'string')) return null;
  if (!Array.isArray(value.stops) || !value.stops.every((x) => typeof x === 'string')) return null;
  return {
    system_present: value.system_present,
    system_sha8: value.system_sha8,
    user_sha8: value.user_sha8,
    tools: value.tools as string[],
    max_tokens: value.max_tokens as number | null,
    stops: value.stops as string[],
    image_sha8: value.image_sha8 as string | null,
    json_mode: value.json_mode,
  };
}

/** The fields the env's client checks, compared exactly as it compares them. */
function echoMatches(expected: Echo, echo: Echo | null): string | null {
  if (echo === null) return 'the response did not carry the whole ANSWER text';
  if (expected.system_present && !(echo.system_present && echo.system_sha8 === expected.system_sha8)) {
    return `the provider echoed system ${echo.system_sha8 ?? 'none'}, the client sent ${expected.system_sha8 ?? 'none'}`;
  }
  if (echo.user_sha8 !== expected.user_sha8) return `the provider echoed user_sha8 ${echo.user_sha8}, the client sent ${expected.user_sha8}`;
  if (expected.tools.length > 0 && (echo.tools.length !== expected.tools.length || echo.tools.some((t, i) => t !== expected.tools[i]))) {
    return `the provider echoed tools [${echo.tools.join(', ')}], the client sent [${expected.tools.join(', ')}]`;
  }
  if (echo.max_tokens !== expected.max_tokens) return `the provider echoed max_tokens ${String(echo.max_tokens)}, the client sent ${String(expected.max_tokens)}`;
  if (expected.stops.length > 0 && (echo.stops.length !== expected.stops.length || echo.stops.some((s, i) => s !== expected.stops[i]))) {
    return `the provider echoed stops ${JSON.stringify(echo.stops)}, the client sent ${JSON.stringify(expected.stops)}`;
  }
  if (expected.image_sha8 !== null && echo.image_sha8 !== expected.image_sha8) {
    return `the provider echoed image ${echo.image_sha8 ?? 'none'}, the client sent ${expected.image_sha8}`;
  }
  if (expected.json_mode && echo.json_mode !== true) return `the provider echoed json_mode ${String(echo.json_mode)}, the client asked for JSON mode`;
  return null;
}

// ---------------------------------------------------------------------------
// Upstream calls: one provider, one wire, buffered whole before it is served.
// ---------------------------------------------------------------------------

interface UpstreamAnswer {
  ok: true;
  candidate: Candidate;
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
  rawJson: string | null;
  rawSse: string | null;
  frames: { event: string | null; data: string }[];
}

interface UpstreamFailure {
  ok: false;
  status: number;
  detail: string;
}

type UpstreamResult = UpstreamAnswer | UpstreamFailure;

function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

function openAITextOf(body: Json): { text: string; input: number; output: number; stop: string } | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    input: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
    output: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

function anthropicTextOf(body: Json): { text: string; input: number; output: number; stop: string } | null {
  if (!Array.isArray(body.content)) return null;
  const text = body.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    input: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
    output: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** Parse one upstream answer (JSON or SSE) into text, usage and stop reason, per wire. */
function parseUpstream(wire: Wire, stream: boolean, raw: string): { text: string; input: number; output: number; stop: string; frames: { event: string | null; data: string }[] } | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(raw);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    if (wire === 'openai') {
      const parsed = openAITextOf(value);
      return parsed === null ? null : { ...parsed, frames: [] };
    }
    if (wire === 'anthropic') {
      const parsed = anthropicTextOf(value);
      return parsed === null ? null : { ...parsed, frames: [] };
    }
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text: geminiTextOf(value),
      input: typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0,
      output: typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0,
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
      frames: [],
    };
  }
  const frames = parseSseFrames(raw);
  if (wire === 'openai') {
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'stop';
    let sawDone = false;
    for (const frame of frames) {
      const payload = frame.data.trim();
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        return null;
      }
      if (!isRecord(chunk)) return null;
      if (isRecord(chunk.usage)) {
        input = typeof chunk.usage.prompt_tokens === 'number' ? chunk.usage.prompt_tokens : input;
        output = typeof chunk.usage.completion_tokens === 'number' ? chunk.usage.completion_tokens : output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
    if (!sawDone) return null;
    return { text, input, output, stop, frames };
  }
  if (wire === 'anthropic') {
    let text = '';
    let input = 0;
    let output = 0;
    let stop = 'end_turn';
    let sawStop = false;
    for (const frame of frames) {
      let value: unknown;
      try {
        value = JSON.parse(frame.data);
      } catch {
        return null;
      }
      if (!isRecord(value)) return null;
      if (frame.event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) {
        input = typeof value.message.usage.input_tokens === 'number' ? value.message.usage.input_tokens : input;
      }
      if (frame.event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') text += value.delta.text;
      if (frame.event === 'message_delta') {
        if (isRecord(value.delta) && typeof value.delta.stop_reason === 'string') stop = value.delta.stop_reason;
        if (isRecord(value.usage)) output = typeof value.usage.output_tokens === 'number' ? value.usage.output_tokens : output;
      }
      if (frame.event === 'message_stop') sawStop = true;
    }
    if (!sawStop) return null;
    return { text, input, output, stop, frames };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const frame of frames) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    text += geminiTextOf(value);
    if (isRecord(value.usageMetadata)) {
      input = typeof value.usageMetadata.promptTokenCount === 'number' ? value.usageMetadata.promptTokenCount : input;
      output = typeof value.usageMetadata.candidatesTokenCount === 'number' ? value.usageMetadata.candidatesTokenCount : output;
    }
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
  }
  return { text, input, output, stop, frames };
}

/** One call to one provider on its own wire, buffered whole and echo-checked. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, expected: Echo): Promise<UpstreamResult> {
  const { provider, model } = candidate;
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  let upstreamBody: Json;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: model.id } : anthropicToOpenAI(body, model.id);
    if (stream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: model.id } : openAIToAnthropicRequest(body, model.id);
    if (stream) {
      upstreamBody.stream = true;
      headers.accept = 'text/event-stream';
    }
  } else {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, detail: `the call to ${provider.id} failed: ${(err as Error).message}` };
  }
  const raw = await res.text();
  if (!res.ok) {
    return { ok: false, status: res.status, detail: `${provider.id} answered HTTP ${res.status}: ${raw.slice(0, 200)}` };
  }
  const parsed = parseUpstream(provider.wire, stream, raw);
  if (parsed === null) {
    return { ok: false, status: 200, detail: `${provider.id} answered an unreadable ${stream ? 'stream' : 'body'}` };
  }
  const mismatch = echoMatches(expected, echoFromText(parsed.text));
  if (mismatch !== null) {
    return { ok: false, status: 200, detail: `${provider.id} echoed wrong: ${mismatch}` };
  }
  return {
    ok: true,
    candidate,
    text: parsed.text,
    inputTokens: parsed.input,
    outputTokens: parsed.output,
    stop: parsed.stop,
    rawJson: stream ? null : raw,
    rawSse: stream ? raw : null,
    frames: parsed.frames,
  };
}

/** Serve one client request with the best free model that answers, failing over as needed. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamAnswer | null> {
  const requires = requirementsFor(body);
  const jsonEcho = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const expected = expectedEcho(clientApi, body);
  const candidates = candidatesFor(requires, jsonEcho);
  requestsSeen += 1;
  for (const candidate of candidates) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const result = await callProvider(candidate, clientApi, body, stream, expected);
      if (result.ok) {
        consecutiveFailures.set(candidate.provider.id, 0);
        restLevel.set(candidate.provider.id, 0);
        restUntil.delete(candidate.provider.id);
        log(`served by ${candidate.provider.id}/${candidate.model.id} (rating ${candidate.rating})`);
        return result;
      }
      log(`refused by ${candidate.provider.id}/${candidate.model.id}: ${result.detail}`);
      // A 400 or 401 would answer the same way again: move to the next provider at once.
      if (result.status === 400 || result.status === 401) break;
      const streak = (consecutiveFailures.get(candidate.provider.id) ?? 0) + 1;
      consecutiveFailures.set(candidate.provider.id, streak);
      if (streak >= 2) {
        const level = (restLevel.get(candidate.provider.id) ?? 0) + 1;
        restLevel.set(candidate.provider.id, level);
        const rest = Math.min(1 << level, 8);
        restUntil.set(candidate.provider.id, requestsSeen + rest);
        log(`${candidate.provider.id} refused twice running: resting ${rest} requests`);
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Send one served answer back in the client's wire, streamed or not as it asked. */
function answer(res: ServerResponse, clientApi: 'openai' | 'anthropic', stream: boolean, served: UpstreamAnswer): void {
  const model = served.candidate.model.id;
  const wire = served.candidate.provider.wire;
  if (!stream) {
    if (clientApi === 'openai') {
      if (wire === 'openai' && served.rawJson !== null) return sendJson(res, 200, served.rawJson);
      if (wire === 'anthropic') {
        const parsed: unknown = served.rawJson !== null ? JSON.parse(served.rawJson) : null;
        return sendJson(res, 200, anthropicToOpenAIResponse(isRecord(parsed) ? parsed : {}, model));
      }
      const parsed: unknown = served.rawJson !== null ? JSON.parse(served.rawJson) : null;
      return sendJson(res, 200, geminiToOpenAI(isRecord(parsed) ? parsed : {}, model));
    }
    if (wire === 'anthropic' && served.rawJson !== null) return sendJson(res, 200, served.rawJson);
    if (wire === 'openai') {
      const parsed: unknown = served.rawJson !== null ? JSON.parse(served.rawJson) : null;
      return sendJson(res, 200, openAIToAnthropic(isRecord(parsed) ? parsed : {}, model));
    }
    const parsed: unknown = served.rawJson !== null ? JSON.parse(served.rawJson) : null;
    return sendJson(res, 200, geminiToAnthropic(isRecord(parsed) ? parsed : {}, model));
  }
  if (clientApi === 'openai') {
    if (wire === 'openai' && served.rawSse !== null) return sendSse(res, served.rawSse);
    if (wire === 'anthropic') return sendSse(res, openAIEventsFromAnthropic(served.frames, model).join(''));
    return sendSse(res, openAIEventsFromGemini(served.frames, model).join(''));
  }
  if (wire === 'anthropic' && served.rawSse !== null) return sendSse(res, served.rawSse);
  if (wire === 'openai' && served.rawSse !== null) {
    const streamBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(served.rawSse));
        controller.close();
      },
    });
    void (async () => {
      const events: string[] = [];
      for await (const event of anthropicEventsFromOpenAI(streamBody, model)) events.push(event);
      sendSse(res, events.join(''));
    })();
    return;
  }
  return sendSse(res, anthropicEventsFromGemini(served.frames, model).join(''));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, body.stream === true);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  answer(res, 'openai', body.stream === true, served);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, body.stream === true);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  answer(res, 'anthropic', body.stream === true, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      log(`register attempt ${attempt} failed: ${lastError.message}`);
      if (attempt < REGISTER_ATTEMPTS) await sleep(REGISTER_RETRY_MS);
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
