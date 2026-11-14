// The Bayesian failure tracker.
//
// The trial seed cannot tell the entry the failure schedule: MAKEFUN_TRIAL_SEED
// is a decimal u64 derived from the instance's sha256 (verified: it is the first
// 8 bytes of sha256("agent-fun/trial-seed/v1\n<instance_sha256>\n<index>")), not
// the 64-hex instance seed the kit's generator draws from, and the instance seed
// reaches the entry nowhere -- not in the env, not in GET /instance, not in the
// brief. So no fixed plan can dodge the scheduled failures; the entry has to
// learn them as the trial runs, one request at a time, and spend its calls where
// they are likeliest to succeed.
//
// What is known before the first request, from the kit's public generator
// (program/generator/draw.mjs, the same sealed table the env and the scorer use):
//   - each provider's budget is uniform in [calls_min, calls_max];
//   - its fail count is 1 or 2, its fail calls distinct uniform draws from
//     2..budget-1 (call 1 NEVER fails: the draws start at 2, an outage starts
//     at 2 at the earliest, and every budget is at least 4);
//   - one provider, picked uniformly, is the second-call provider: its fail
//     set is forced to contain call 2;
//   - one provider, picked uniformly, goes out: its outage_from is uniform in
//     2..budget-1, and every call from it on refuses.
// The tracker holds, per provider, the exact posterior over those hypotheses
// (budget, fail set, outage, designated), starting from the prior above and
// keeping only the hypotheses consistent with every call outcome observed so
// far. pFail() is the posterior probability that the provider's NEXT call
// refuses. A request is served by the candidate maximising
// rating_milli - 250 * pFail, so a provider whose next call is likely to fail
// loses to a slightly worse model that will answer, and a provider never
// called yet (next call 1, pFail 0) is always a safe harbour. After a refusal
// the same provider is retried once when its next call still looks likely to
// succeed (pFail < 0.5): a scheduled fail is an isolated ordinal, so the next
// one usually answers and keeps the best rating; an outage makes pFail jump
// towards 1 and the provider is left alone.
//
// Everything is a function of the brief and the providers' answers: no wall
// clock and no unseeded randomness, so the live pass and its replay compute
// the same choices and leave the same state export.

import {
  meetsRequirements,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './providers.ts';

/**
 * A provider's best free model meeting the requirements, or null. The rating
 * row is the brief's sealed quality table; a JSON-mode request on the OpenAI
 * wire only considers wires that can carry response_format (openai, gemini).
 */
export function bestModelFor(
  ratings: readonly RatingRow[],
  provider: ProviderDirectoryEntry,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
): { model: string; rating: number } | null {
  if (clientApi === 'openai' && requires.json === true && provider.wire === 'anthropic') return null;
  let best: { model: string; rating: number } | null = null;
  for (const row of ratings) {
    if (row.provider !== provider.id || !row.free) continue;
    if (!meetsRequirements(requires, { context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json })) continue;
    if (best === null || row.rating_milli > best.rating || (row.rating_milli === best.rating && row.model < best.model)) {
      best = { model: row.model, rating: row.rating_milli };
    }
  }
  return best;
}

/** The generator's per-provider call ranges (calls_min, calls_max). */
const CALL_RANGES: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/** Milli-points of rating one unit of failure probability is worth. */
export const FAIL_PENALTY_MILLI = 250;
/** A refused provider is retried once only while its next call looks this likely to succeed. */
export const RETRY_P_FAIL_MAX = 0.5;

interface Hypothesis {
  budget: number;
  /** Bit n set when call n is on the provider's fail schedule (n >= 2). */
  failMask: number;
  /** The first out ordinal, or 0 when this provider never goes out. */
  outage: number;
  weight: number;
}

/** All k-subsets of `xs`, in lexicographic order. */
function combinations(xs: number[], k: number): number[][] {
  if (k === 0) return [[]];
  if (xs.length < k) return [];
  const [first, ...rest] = xs as [number, ...number[]];
  return [...combinations(rest, k - 1).map((c) => [first, ...c]), ...combinations(rest, k)];
}

function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let out = 1;
  for (let i = 0; i < k; i++) out = (out * (n - i)) / (i + 1);
  return out;
}

/** The prior hypotheses for one provider, exactly as the generator draws them. */
function buildHypotheses(providerId: string): Hypothesis[] {
  const range = CALL_RANGES[providerId];
  if (range === undefined) return [];
  const [lo, hi] = range;
  const out: Hypothesis[] = [];
  for (let budget = lo; budget <= hi; budget++) {
    const pBudget = 1 / (hi - lo + 1);
    const vals: number[] = [];
    for (let v = 2; v <= budget - 1; v++) vals.push(v);
    for (const count of [1, 2]) {
      const subsets = combinations(vals, count);
      const pSubset = 0.5 / binom(vals.length, count);
      for (const subset of subsets) {
        const sorted = [...subset].sort((a, b) => a - b);
        for (const designated of [0, 1]) {
          const pDesignated = designated === 1 ? 0.1 : 0.9;
          const fails = designated === 1 && !subset.includes(2) ? [2, ...sorted.slice(1)] : sorted;
          const failMask = fails.reduce((m, n) => m | (1 << n), 0);
          for (const outage of [0, ...vals]) {
            const pOutage = outage === 0 ? 0.9 : 0.1 / vals.length;
            out.push({ budget, failMask, outage, weight: pBudget * pSubset * pDesignated * pOutage });
          }
        }
      }
    }
  }
  return out;
}

/** Whether call `n` succeeds under one hypothesis. */
function succeeds(h: Hypothesis, n: number): boolean {
  return n <= h.budget && (h.failMask & (1 << n)) === 0 && (h.outage === 0 || n < h.outage);
}

export class BayesTracker {
  private hypotheses = new Map<string, Hypothesis[]>();
  private used = new Map<string, number>();

  constructor(providerIds: readonly string[]) {
    for (const id of providerIds) {
      this.hypotheses.set(id, buildHypotheses(id));
      this.used.set(id, 0);
    }
  }

  /** Calls made to this provider so far. */
  usedCalls(providerId: string): number {
    return this.used.get(providerId) ?? 0;
  }

  /** The posterior probability that this provider's NEXT call refuses. */
  pFail(providerId: string): number {
    const hyps = this.hypotheses.get(providerId);
    if (hyps === undefined || hyps.length === 0) return 1;
    const n = (this.used.get(providerId) ?? 0) + 1;
    let total = 0;
    let bad = 0;
    for (const h of hyps) {
      total += h.weight;
      if (!succeeds(h, n)) bad += h.weight;
    }
    return total > 0 ? bad / total : 1;
  }

  /** Record one call just made (its ordinal is usedCalls() after counting it). */
  observe(providerId: string, ok: boolean): void {
    const n = (this.used.get(providerId) ?? 0) + 1;
    this.used.set(providerId, n);
    let hyps = this.hypotheses.get(providerId);
    if (hyps === undefined) return;
    const kept = hyps.filter((h) => succeeds(h, n) === ok);
    let total = 0;
    for (const h of kept) total += h.weight;
    if (total <= 0) {
      // No hypothesis explains the outcome: restart from the prior (never happens
      // on a trial drawn by the kit's generator; it only guards the tracker).
      hyps = buildHypotheses(providerId);
      total = 0;
      for (const h of hyps) total += h.weight;
      this.hypotheses.set(providerId, hyps);
    } else {
      this.hypotheses.set(providerId, kept);
    }
    if (total > 0) for (const h of this.hypotheses.get(providerId) as Hypothesis[]) h.weight /= total;
  }

  /** A candidate's value: its rating less the failure penalty for its next call. */
  utility(providerId: string, ratingMilli: number): number {
    return ratingMilli - FAIL_PENALTY_MILLI * this.pFail(providerId);
  }
}
