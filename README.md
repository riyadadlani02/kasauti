# Does agentic capability survive quantization?

Router instability as a predictor of capability collapse in quantized MoE models.

Working name: `kasauti`, the touchstone used to test the purity of gold.

---

## Problem

**Quantization is universal in deployment, and the evidence that it is safe is measured with the wrong instrument.**

Every serious LLM deployment quantizes. FP8, INT8, W4A16 are standard in production serving stacks. The justification is almost always an aggregate benchmark delta: MMLU drops half a point at 4-bit, therefore the model is fine.

That instrument was built for a different workload. Aggregate benchmarks are single-turn, short-output, knowledge-recall tasks. Production deployments are increasingly agentic: multi-turn trajectories, twenty or more tool calls, structured output that has to parse, constraints stated once at the start and honoured thirty steps later. In that regime errors compound rather than average out. A benchmark that samples independent short questions is structurally incapable of seeing compounding failure.

So the load-bearing question is not how much accuracy is lost. It is **which capabilities break first, and whether the ones that break first are exactly the ones nobody is measuring.**

**Mixture-of-Experts makes this sharper.**

Most quantization analysis assumes degradation is continuous: weights get noisier, outputs get slightly worse, error metrics track it. MoE breaks that assumption, because routing is a discrete decision. The gate produces logits, top-k selection collapses them to a hard choice. When quantization perturbs router logits by a small amount, tokens sitting near a decision boundary flip to a different expert set.

Once routing changes, the model is not a slightly degraded version of the original. It is executing a different computational path. Per-layer reconstruction error cannot detect this, because it measures continuous error and the damage here is a discrete flip. Perplexity largely cannot either, since an alternative expert often produces a plausible next token while degrading behaviour that only shows up over a long horizon.

**The gap.** Nobody has measured capability-specific degradation and router instability together, on an open MoE model that is publicly positioned for agentic use.

**Why Sarvam 30B / 105B is the right subject.** Apache 2.0, weights on Hugging Face and AI Kosh, MoE architecture, and Sarvam published agentic benchmark results positioning the models for autonomous agent workloads. There is a public claim to test against, which is worth more than testing a model nobody has made claims about.

---

## Hypotheses

**H1 — Degradation is capability-specific, not uniform.**
Long-horizon instruction adherence, format stability, calibration, and error recovery degrade at higher bit-widths than knowledge recall does.

**H2 — There is a crossing point.**
A bit-width exists at which aggregate benchmark scores remain visually flat while agentic task success has already fallen materially. This is the chart the paper is built around.

**H3 — Router divergence predicts agentic degradation better than perplexity or reconstruction error.**

H3 is the differentiated contribution. H1 and H2 are careful measurement. H3 is a mechanism, and if it holds it yields something practically useful: a cheap diagnostic that predicts whether a quantization config will break agentic behaviour, without paying for a full agentic evaluation.

---

## Method

### Configurations

Model: **Sarvam 30B**, served under vLLM. Sarvam 105B as a scaling check only if the 30B result holds.

Quantization sweep:

| Config | Purpose |
|---|---|
| BF16 | baseline |
| FP8 | current production default |
| INT8 W8A8 | conservative |
| INT4 W4A16, GPTQ | aggressive |
| INT4 W4A16, AWQ | separates method from bit-width |

Running two INT4 methods matters. Without it, any 4-bit finding is confounded with the specific quantization algorithm.

**Component ablation:** quantize experts only, attention only, and everything. This localises where the damage originates and is what turns a measurement paper into a mechanism paper.

### Layer 1 — the instrument being critiqued

Standard aggregate benchmarks. The point is to establish the "looks fine" reading that practitioners currently rely on. This is the control, not the result.

### Layer 2 — capability probes

Four probes, each isolating one capability, each with **ground truth by construction** so scoring is mechanical.

**Long-horizon instruction adherence.** N constraints stated at step 0, verified at step N. Sweep N from 5 to 50. Constraints chosen to be programmatically checkable, for example never use a specific token, always append a given field, keep a running counter accurate.

**Format stability under pressure.** Structured output requested while the input contains nested delimiters, escaped quotes, and injected braces. Metric is parse rate, not similarity.

**Calibration and refusal.** Unanswerable and underspecified questions mixed with answerable ones. Measures whether abstention behaviour survives, which is the failure mode most likely to be silently lost and most consequential in production.

**Error recovery.** Inject a malformed or contradictory tool result mid-trajectory. Measure whether the agent recovers, retries sensibly, or spirals. Recovery is a distinct capability from getting it right first time, and it is the one that determines whether a real deployment degrades gracefully.

No LLM-as-judge anywhere in this layer. A quantized judge is circular, and an external API judge adds variance you cannot bound. Everything here is scored by construction.

### Layer 3 — end-to-end agentic

A τ-bench style tool-use benchmark, chosen to overlap with what Sarvam has published on, so the result is directly comparable to their claim.

### Router instrumentation

This is the part that has to be done carefully.

Hook the MoE gating layers. For identical inputs, log per token per layer:

- selected top-k expert IDs
- router logits before top-k
- resulting expert load distribution

**Run teacher-forced.** If both the BF16 and quantized models generate freely, their token sequences diverge within a few steps and any routing comparison after that point is meaningless, because you are comparing routing on different inputs. Force both models through the same token sequence so routing is the only thing varying. Getting this wrong invalidates the entire H3 result and it is easy to get wrong.

Metrics:

- **Top-1 flip rate.** Fraction of tokens whose highest-scoring expert changed.
- **Top-k Jaccard distance.** Captures partial reshuffling that flip rate misses.
- **Router logit margin distribution.** Gap between the k-th and (k+1)-th expert under BF16. Narrow margins mark tokens sitting near a decision boundary, and the prediction is that these are where flips concentrate. If margin distribution predicts flip location, the mechanism story is established rather than asserted.
- **Expert load shift.** KL divergence between BF16 and quantized load distributions. Tests whether quantization collapses traffic onto fewer experts.

Then correlate each divergence metric against per-probe degradation, with perplexity and per-layer reconstruction error as competing predictors. H3 holds only if router divergence beats both.

---

## Deliverables

1. **The crossing-point chart.** Bit-width on x, normalised score on y. Aggregate benchmark flat, agentic curves falling. One figure carrying the argument.
2. **Router divergence versus capability degradation**, with perplexity and reconstruction error plotted as baseline predictors.
3. **A cheap diagnostic.** Router divergence measured on a small calibration set, used to predict agentic safety of a quantization config without running the full agentic evaluation. This is the part a serving team would actually adopt.
4. Probe suite and instrumentation released as a repo, runnable against any open MoE.

---

## What this is not

Not a claim that quantization is harmful, and not an argument against deploying quantized models. The claim is narrower and more defensible: **the current evidence standard for quantization safety does not measure the capabilities that agentic deployments depend on, and for MoE models there is a specific mechanism that explains why.**

---

## Risks

**Null on H2.** Benchmarks may track agentic degradation adequately. Publishable as a negative result, but weaker. Pre-register hypotheses and thresholds before running so the null does not look like a failed fishing expedition.

**H3 collapses into perplexity.** If router divergence correlates with perplexity and adds no independent predictive power, the mechanism contribution evaporates and you are left with a measurement paper. Test this early, on one probe, before building the full suite.

**Compute.** Agentic evaluation is expensive per configuration. Five quant configs times three ablations times full agentic eval will not fit a student budget. Run the full grid on the cheap probes, and the agentic benchmark only on configs that bracket the crossing point.

**Scope creep.** One model. Five configs. Four probes. One agentic benchmark. No additional languages, no additional model families, no additional quantization methods. Every one of those is a tempting extension that turns a finishable project into an unfinished one.

---

## Sequence

1. Reproduce the BF16 baseline and confirm the published agentic number is reachable. If it is not, stop and find out why before proceeding.
2. Build router instrumentation and verify teacher-forcing on a toy input.
3. Early test of H3 on a single probe at INT4. This is the go/no-go gate.
4. Build the full probe suite.
5. Full sweep, ablations, agentic evaluation.
6. Write up.

Step 3 is the gate. If router divergence has no predictive power there, revert to the capability-degradation paper alone, which still stands on its own, and do not spend the remaining months on the mechanism.
---

## The tool

The study comes first, because the diagnostic is only trustworthy once the correlation it rests on has been demonstrated. `kasauti check` is that diagnostic, built now so the instrumentation is shared between the study and the shipped tool — but its verdict thresholds are placeholders until step 3 of the sequence above fixes them.

```bash
pip install -e .
kasauti check ./sarvam-30b-w4a16 --base ./sarvam-30b-bf16
```

```
router flip rate:              11.3%
top-k jaccard distance:        0.184
expert load KL:                0.061
predicted agentic degradation: HIGH
worst layers:                  18-26
flips by base margin quartile (narrow to wide): 34.1%  12.8%  4.9%  1.6%
suggested: hold the gate and attention at higher precision and re-check
```

Both checkpoints see the identical token sequence in a single forward pass, so routing is the only thing varying. `--generate N` extends each prompt with N tokens generated by the base model and forces that same sequence through the quantized one, which is the teacher-forced comparison the method requires — never let the two models generate independently.

| Flag | |
|---|---|
| `--calibration FILE` | prompts separated by lines of `---`; defaults to a built-in agentic-shaped set |
| `--generate N` | teacher-force N base-generated tokens |
| `--gate-pattern RE` | override gate module matching for an unseen architecture |
| `--low / --high` | flip-rate cut points for the LOW/MEDIUM/HIGH verdict |
| `--json` | full per-layer report |

Gates that return selected expert ids rather than logits are handled, minus the margin analysis. `python test_kasauti.py` runs the self-check.

---

## Running the study

```bash
pip install -e .
python sweep.py granite-1b-a400m --device mps --out results.jsonl   # the grid
python analyze.py results.jsonl                                     # figures + RESULTS.md
python compat.py                                                    # architecture check, no weights
```

`sweep.py` runs nine configs: a bit-width sweep (BF16, W8, W6, W4, W3) applied to experts and attention, a second W4 method to separate method from bit-width, and three component ablations (experts only, attention only, everything including the router). Each config is measured five ways — the probe suite, perplexity, decoder-layer reconstruction error, router divergence, and quantized-weight error — so H3's competing predictors are all scored against the same target on the same configs.

Quantization is round-to-nearest over groups of the input dimension, applied to parameters rather than modules, because MoE experts are usually one 3D weight bank rather than a Linear per expert. It runs on any device, which is what makes the component ablation affordable. Vendor-quantized checkpoints (FP8, NVFP4, MXFP4) are swept by loading them instead, with `--vendor`.

`compat.py` builds a model on the meta device from its config alone, so gate discovery and component classification can be verified for a 105B model without downloading a byte of weights.

## Architecture compatibility

Verified by constructing each model on the meta device (`python compat.py`):

| model | architecture | experts | top-k | gates found | status |
|---|---|---|---|---|---|
| `sarvamai/sarvam-30b` | SarvamMoE | 128 | 6 | 18/18 | instrumentation attaches |
| `openai/gpt-oss-20b` | GptOss | 32 | 4 | 24/24 | instrumentation attaches, ships MXFP4 |
| `allenai/OLMoE-1B-7B-0125-Instruct` | Olmoe | 64 | 8 | 16/16 | instrumentation attaches |
| `ibm-granite/granite-3.0-1b-a400m-instruct` | GraniteMoe | 32 | 8 | 24/24 | instrumentation attaches |
| `sarvamai/sarvam-105b` | SarvamMoE | — | — | — | its bundled modeling code needs transformers < 5 |
| `nvidia/Nemotron-3-Nano-30B-A3B` | NemotronH | 128 | 6 | — | hybrid Mamba, needs `mamba-ssm` and CUDA |

Most MoE routers return the selection rather than the logits behind it, so the logits are recomputed from the router weight against the hidden state it saw. That keeps the margin analysis available on any architecture, while the model's own reported selection is still used where it gives one, so group-limited routing is not misread as plain top-k.

Worth noting what OpenAI already does in `gpt-oss-20b`: its `modules_to_not_convert` exempts `mlp.router` and `self_attn` from MXFP4. The router is kept at full precision by the one vendor shipping a natively quantized agentic MoE — which is the practice this study is trying to give evidence for or against.

---

## Results

[RESULTS.md](RESULTS.md) has the full write-up; [RESULTS_1b.md](RESULTS_1b.md) is the generated table dump.

On `granite-3.0-1b-a400m-instruct`, at 4-bit: **perplexity improved 4.6% and knowledge recall moved 4%, while format stability fell 24%, long-horizon instruction adherence fell 19%, and half of all tokens were routed to a different expert set.** The crossing point is real and it sits at the most widely deployed bit-width.

Router divergence predicts agentic degradation (r = −0.82) where perplexity does not (−0.26); it does not clearly beat per-layer reconstruction error (−0.79), so H3 half-holds. Flips concentrate at narrow router margins — 10.0% of narrow-margin tokens against 0.0% of wide-margin tokens at 8-bit, monotone in every config, replicated on a 3.3B model.

Quantizing attention alone, touching no expert or router weight, re-routed 43% of tokens and cost more long-horizon adherence than any other config — while *improving* perplexity.

## The search agent

The study says which signal to trust; `search.py` uses it to search.

```bash
python search.py granite-1b-a400m --budget 1.0 --validate-every 2 --min-agentic 0.95
```

It lowers one component's bit-width at a time, scoring each candidate by model size bought per unit of damage, and every few accepted steps it stops trusting the cheap signal and runs the real probe suite. If the audit fails, the step is reverted and the budget tightened below the cost of whatever just failed.

On the 1.3B subject it converged in 24 evaluations to `gate 5 / attention 8 / expert 5` — 68% smaller than BF16, keeping 95% of agentic capability where uniform 4-bit keeps 78%. Full trace in [RESULTS.md](RESULTS.md).

## Roadmap

Ship the study, then ship the CLI that operationalises it. A diagnostic with no validation behind it is a number generator, and that is the first thing a reviewer would spot.

1. Study: the crossing-point chart and the H3 correlation.
2. Tool: thresholds calibrated against that correlation, so `kasauti check` predicts agentic safety in minutes instead of the hours a full agentic eval costs.

The user is anyone quantizing an MoE for serving, who right now has no way to answer this question except by running the expensive eval.
