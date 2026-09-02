# Results

What ran, where, and what it does and does not support.

**Subject:** `ibm-granite/granite-3.0-1b-a400m-instruct` — a real trained MoE, 24 layers, 32 experts, top-8, 1.3B total parameters. **Hardware:** one 16GB M4 laptop, MPS. Nine configs, ~70 minutes.

The 30B-class subjects the method specifies — Sarvam 30B, Nemotron 3 Nano 30B, gpt-oss-20b — do not fit in 16GB. The instrumentation is verified against their real architectures (see the compatibility table in the README), and `sweep.py` runs them unchanged on a GPU box. Everything below is measured on the 1.3B subject and should be read as a pipeline result at small scale, not as the headline claim.

**Measured:** 100 probe items per config, 794 calibration tokens through 24 gates (≈19k routing decisions per config), perplexity, decoder-layer reconstruction error, and quantized-weight error.

---

## The crossing point exists, and perplexity crosses it backwards

![crossing point](crossing_point.png)

| config | perplexity (794 tok) | recall (control) | long-horizon | format | tokens re-routed |
|---|---|---|---|---|---|
| BF16 | 27.65 | 1.000 | 1.000 | 1.000 | — |
| W8 | 27.72 | 1.000 | 1.000 | 1.000 | 8.7% |
| W6 | 27.21 | 1.154 | 0.927 | 1.000 | 19.0% |
| **W4** | **26.39** | **0.962** | **0.805** | **0.762** | **49.3%** |
| W3 | 53.33 | 0.385 | 0.951 | 0.667 | 76.9% |

At 4-bit, **perplexity improved by 4.6% while format stability fell 24% and long-horizon instruction adherence fell 19%**, and half of all tokens were routed to a different expert set. The control probe moved 4%. Both instruments a practitioner would actually consult said the model was fine; two of the four agentic probes said it was not.

The sweep measures perplexity on only 794 calibration tokens, so that number was re-measured independently on 12,663 tokens of agentic-shaped text. It replicates: **ratio 0.954 against the sweep's 0.955** at 4-bit, and 1.000 at 8-bit. The direction is not a sampling artifact.

Only at 3-bit do the aggregate measures finally react — and there they collapse hardest of all (recall −62%, perplexity 1.9×), which is the point: they are sensitive in the region where nobody deploys and insensitive in the region where everybody does.

**H1 holds** on this subject: degradation is capability-specific, not uniform. **H2 holds**: there is a bit-width at which the aggregate reading is flat and agentic capability has already fallen materially. Here that width is 4, the most widely deployed one.

## H3 — fails. Router divergence beats perplexity; it does not beat reconstruction error

![predictors](predictors.png)

The first pass ran 8 configs and put router divergence on top (r = −0.82 against perplexity's −0.26). That result did not survive. Nineteen more configs were added — gate-only quantization at four bit-widths, three group sizes, per-channel scaling, and a second method — chosen specifically to make the competing predictors disagree. On 27 configs:

| predictor | r (core target) | rho |
|---|---|---|
| layer reconstruction error | −0.577 | −0.542 |
| routing change × compute damage | −0.584 | −0.560 |
| router expert-set change rate | −0.477 | −0.403 |
| perplexity ratio | **−0.664** | **−0.217** |

Perplexity has the highest Pearson and nearly the lowest Spearman, which is the signature of one outlier carrying a correlation: `x3-ea-awq` blew perplexity out to 4.5× and dragged the line with it. Nobody deploys that config.

**So ask the decision-relevant question instead.** Restricted to the 21 configs whose perplexity stayed within 25% of baseline — the ones that *look fine*, which is what this whole study is about:

| predictor | r (core target) | rho |
|---|---|---|
| **layer reconstruction error** | **−0.590** | **−0.561** |
| router expert-set change rate | −0.509 | −0.455 |
| router top-k jaccard | −0.487 | −0.452 |
| routing change × compute damage | −0.290 | −0.409 |
| perplexity ratio | −0.118 | −0.012 |

And as a decision rule rather than a correlation — split the 25 configs into healthy (agentic score ≥ 0.95) and damaged (< 0.90), then find each predictor's best threshold:

| rule | accuracy |
|---|---|
| reconstruction error < 0.100 | **84%** |
| router set-change rate < 0.385 | **84%** |
| perplexity ratio < 0.850 | 60% |

Fifteen of 25 configs were damaged, so 60% is the base rate. **As a decision rule, perplexity is exactly as good as guessing.** That is the study's central claim, and it is now measured on 27 configs rather than argued.

But router divergence and reconstruction error tie, at 84% each, and reconstruction error wins every correlation in the deployable regime. **H3 does not hold.** Router divergence is a good predictor and a decisively better one than perplexity; it is not the best available cheap predictor, and the method's own contingency plan applies — the capability-degradation result stands on its own, and the mechanism claim becomes a secondary contribution rather than the headline.

### Why routing alone is not enough

The gate-only configs are what killed it, and they are interesting in their own right:

| config | tokens re-routed | agentic score | perplexity |
|---|---|---|---|
| 4-bit router only | 15.0% | 1.012 | 0.971 |
| 3-bit router only | 26.2% | 0.899 | 1.006 |
| 2-bit router only | 48.1% | 1.009 | 1.714 |

**Quantizing only the router to 2 bits re-routed nearly half of all tokens and cost nothing.** Sending a token to a different expert is harmless when every expert it might land on is intact — the alternative expert is a legitimately trained function, not a broken one. Routing divergence over-predicts damage exactly when the destinations are undamaged.

That suggests the damage is the *product* of routing change and destination damage, so I tested `routing change × weight error of the computing components`. It is the best predictor over the full range (−0.584) and one of the worst inside the deployable regime (−0.290). The interaction idea is not supported by this data.

## Mechanism — the flips are where the router was undecided

The prediction was that quantization flips exactly the tokens sitting near a routing decision boundary. Pairing each margin with the event it governs — the top-1 margin with a change of argmax, the k-th/(k+1)-th margin with a change of the selected set:

| config | top-1 flips, narrow → wide quartile | set changes, narrow → wide |
|---|---|---|
| W8 | **0.100** 0.002 0.000 **0.000** | **0.289** 0.051 0.008 **0.000** |
| W6 | 0.211 0.019 0.002 0.000 | 0.481 0.213 0.059 0.006 |
| W4 | 0.443 0.205 0.066 0.011 | 0.717 0.614 0.462 0.181 |
| W3 | 0.580 0.437 0.295 0.106 | 0.885 0.850 0.778 0.565 |

Monotone in every config, at every bit-width. At 8-bit, a narrow-margin token is *infinitely* more likely to flip than a wide-margin one — 10.0% against 0.0%. This is the mechanism story demonstrated rather than asserted, and it is what licenses the cheap diagnostic: the margin distribution is computable from the base model alone, before any quantized checkpoint exists.

Paired the other way round — the k-boundary margin against top-1 flips — the same data shows a flat profile and no mechanism at all. The pairing is the whole result.

## Component ablation — attention damage is routing damage

| what was quantized to 4-bit | perplexity | long-horizon | format | tokens re-routed | worst layers |
|---|---|---|---|---|---|
| experts only | 30.77 | 0.854 | 1.048 | 34.8% | 12–22 |
| attention only | **0.969×** | **0.585** | 0.952 | 42.8% | 4–16 |
| experts + attention | 26.39 | 0.805 | 0.762 | 49.3% | 11–22 |
| all three, router included | 26.72 | 0.683 | 0.857 | 55.0% | 11–19 |

Quantizing **attention alone improved perplexity and produced the worst long-horizon adherence in the study, 41% below baseline.** On the 794-token calibration set that config looked 15% better than BF16; re-measured on 12,663 tokens the improvement is a more modest 3.1%, which is the figure in the table — the small-sample number was inflated, the sign was not. For a practitioner reading perplexity it is still an attractive-looking config, and it is the worst one on the board for keeping instructions across a long trajectory.

It also re-routed 42.8% of tokens without touching a single expert or router weight. Routing damage is not only expert damage: the gate reads what attention produced, so degrading attention moves the router's input and changes the computational path. Damage also localises differently — deep layers for expert quantization, early-to-middle layers for attention.

Adding the router itself to the quantization set (`w4-all`) cost a further 5.7 points of re-routing over experts+attention. That is the empirical case for the exemption OpenAI already ships in `gpt-oss-20b`, whose `modules_to_not_convert` keeps `mlp.router` and `self_attn` out of MXFP4.

## Scale check — the routing result replicates on a 2.5× model

A second subject, `ibm-granite/granite-3.1-3b-a800m-instruct` — a different model generation, 32 layers instead of 24, 40 experts instead of 32, 3.3B parameters. The full probe sweep did not fit in 16GB (batch-16 generation over 2k-token contexts drove the machine into 25GB of swap), so this is routing only: eight configs, no probes.

| config | top-1 flips, 1.3B → 3.3B | tokens re-routed, 1.3B → 3.3B |
|---|---|---|
| W8 | 2.5% → 3.3% | 8.7% → 11.4% |
| W6 | 5.8% → 6.4% | 19.0% → 22.2% |
| W4 | 18.1% → 18.2% | 49.3% → 53.8% |
| W3 | 35.5% → 36.6% | 76.9% → 81.3% |
| W4, experts only | 12.3% → 10.4% | 34.8% → 34.2% |
| W4, attention only | 15.1% → 15.8% | 42.8% → 48.4% |

The margin concentration replicates too, and slightly more sharply. At 8-bit the 3.3B model flips 12.8% of its narrowest-margin tokens and 0.0% of its widest, against 10.0% and 0.0% on the 1.3B. Every config on both models is monotone across all four quartiles.

Attention-only quantization again causes more routing damage than expert-only quantization (15.8% against 10.4% at 3.3B), without touching a single expert or router weight. That is the one structural claim in this study that now rests on two models rather than one.

What does **not** replicate here is the capability half — no probes were run at 3.3B, so H1 and H2 remain single-subject results.

---

## The search agent

`search.py` closes the loop. It walks each component's bit-width down one rung at a time, scores each candidate by how much model size it buys per unit of damage, and keeps the best step — where damage is the worse of the two signals above, each measured against its own cut point. Every few accepted steps it stops trusting that cheap signal and pays for a real probe run.

Run on the 1.3B subject with a floor of 0.95 relative agentic score, it converged in 24 evaluations and 30 minutes:

```
accept expert -> 8    gate 16, attn 16, expert  8   mean 8.48 bits   cost 0.19
accept attention -> 8 gate 16, attn  8, expert  8   mean 8.00 bits   cost 0.23
  VALIDATE: agentic 0.985
accept expert -> 6    gate 16, attn  8, expert  6   mean 6.12 bits   cost 0.47
accept expert -> 5    gate 16, attn  8, expert  5   mean 5.18 bits   cost 0.91
  VALIDATE: agentic 0.983
accept gate -> 6      gate  6, attn  8, expert  5   mean 5.18 bits   cost 0.88
  VALIDATE: agentic 1.059
accept gate -> 5      gate  5, attn  8, expert  5   mean 5.18 bits   cost 0.96
no step left within budget
```

Its first decision is the interesting one. Lowering the router to 8-bit and lowering the experts to 8-bit cost the same damage (0.19), but the router is 0.1% of the parameters, so it saves 0.00 bits against the experts' 7.52. The agent went for the experts and left the router alone until nothing else was affordable — arriving at the exemption `gpt-oss-20b` ships, from the cheap signal alone.

It also stopped attention at 8-bit while pushing experts to 5-bit, which is the study's attention finding rediscovered rather than told.

Against the config a practitioner would actually pick, verified afterwards with the full probe suite:

| config | mean bit-width | agentic (long-horizon + format) | recall |
|---|---|---|---|
| BF16 | 16.00 | 1.000 | 0.650 |
| uniform 4-bit, experts + attention | 4.01 | 0.783 | 0.625 |
| **what the agent found** | **5.18** | **0.952** | 0.675 |

For 1.2 more bits than uniform 4-bit — still 68% smaller than BF16 — it keeps 95% of the agentic capability instead of 78%.

**What the loop cannot do.** It is greedy: an early choice is only undone when an audit fails. Its cut points come from one 1.3B model. Its auditor uses the same probe suite the study used, so it cannot catch a failure mode those probes do not cover — a self-improving loop is bounded by the quality of its own judge, and that is the honest limit of this design, not a detail. And each evaluation is one forward pass per calibration text plus a model load, so on a 30B model the wall-clock cost per step is far higher than the 40 seconds it takes here.

---

## Threats to validity

Stated in full, because the finding is only worth what survives them.

1. **Scale.** 1.3B parameters for the capability results, not 30B. The routing results replicate at 3.3B; the probe results do not have a second subject. Small models have less redundancy, so they should degrade *earlier*; whether the crossing point sits at the same bit-width on a 30B MoE is exactly what this study cannot say.
2. **Statistical power.** 100 probe items, 12–40 per probe. A single item is 2.5–8 percentage points. Single-config deltas are noisy; the monotone trend across the bit sweep is the signal, not any one cell.
3. **Perplexity sample.** The sweep uses 794 calibration tokens, which is too few to trust a 5% move. The two claims that depend on the direction of that move were re-measured on 12,663 tokens: the 4-bit improvement replicated (0.954 vs 0.955), the attention-only improvement shrank from 15% to 3.1%. Every other perplexity figure in the tables is still the 794-token measure and should be read as indicative only.
4. **The calibration probe is confounded.** It scores abstention on unanswerable questions, and a damaged model hedges more. At 2-bit router quantization the model's fluency visibly degraded while its calibration score rose from 0.50 to 0.80 — it was abstaining more, not calibrating better. Every correlation is therefore reported twice: over all probes with headroom, and over a `core` target of long-horizon adherence and format stability only. That split was chosen after seeing the confound, not before.

5. **Two probes were at their floor.** `recovery` scored 0.10 at BF16 and `calibration` 0.50, which is exactly what always-answering scores. A 1.3B model cannot do those tasks, so they measure nothing about degradation and are excluded from the agentic mean. H1/H2 here rest on long-horizon adherence and format stability.
6. **Fake quantization.** Round-to-nearest through a quantized grid, not GPTQ or AWQ kernels. It isolates the arithmetic from any one implementation, and it is not what a serving stack runs.
7. **The `awq` config is not AWQ.** It is activation-aware scaling with a fixed alpha and no search. It produced *higher* weight error than plain RTN (0.165 against 0.110 on attention) and worse perplexity. Read it as evidence that method matters at fixed bit-width, not as a result about AWQ.
8. **The control is a proxy.** 40 open-ended recall items, not MMLU. On a GPU box, swap in lm-eval-harness.
9. **Batch composition is not neutral.** Probes are generated in left-padded batches, and changing the batch composition moves a score by about one item — the BF16 format score was 0.875 in the 100-item run and 0.917 in the 60-item run, same model, same items, greedy decoding. Each run is therefore normalised against its own baseline, and roughly one item of noise is unavoidable.

10. **n=27, still small.** Enough to separate perplexity from the other two predictors decisively; not enough to separate router divergence from reconstruction error, which tie.

11. **Two post-hoc analysis choices.** The `core` target and the deployable-regime restriction were both made after seeing the data, for reasons stated above. Both are reported alongside the unrestricted numbers rather than replacing them.

## What needs a GPU box

The sweep runs unchanged on the real subjects; only the hardware is missing.

```bash
python sweep.py sarvam-30b --vendor --device auto      # + sarvamai/sarvam-30b-fp8
python sweep.py gpt-oss-20b --device auto
python sweep.py nemotron-3-nano-30b --vendor           # needs mamba-ssm, CUDA only
```

`--vendor` sweeps the vendor's own quantized checkpoints — Sarvam's FP8, NVIDIA's FP8 and NVFP4 — rather than faking the arithmetic. Those are the configs serving teams actually deploy, and comparing them against their own BF16 release is the strongest available version of this experiment. Two blockers found while checking: `sarvam-105b`'s bundled modeling code needs transformers < 5, and Nemotron 3 Nano is a hybrid Mamba MoE requiring `mamba-ssm`, which needs CUDA to build.
