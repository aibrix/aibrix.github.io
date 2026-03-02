# LLM Batch Cost Calculator

A self-hosted vs cloud API cost comparison tool for LLM batch inference. Helps you answer: **"Is it cheaper to run this workload on my own GPUs or pay a cloud API?"**

---

## How to Use

The dashboard has two sides: **Infrastructure** (left panel, configure once) and **Job** (right panel, change per workload).

### Left Panel — Infrastructure Setup

Work through the steps top to bottom, once per cluster configuration.

**① GPU Setup**

Select your GPU type from the grid. This sets the default market rate (Lambda/RunPod/CoreWeave on-demand, Feb–Mar 2026). Fields to override:

- **GPU Count** — how many GPUs in your cluster
- **$/GPU/hr override** — enter your actual contract rate if different from market defaults

The total rental cost (with overhead) is shown below the fields.

Use the sliders to account for cost overhead:
- **Startup overhead** — percentage of GPU time spent loading the model. Set to 0% if you're running a persistent serving process (always-on). Set to 5–10% for batch jobs that cold-start each time.
- **Interruption rate** — percentage of GPU-hours lost to spot instance preemptions or hardware failures. Near 0% on stable on-prem, 5–15% on spot cloud instances.

**② Benchmark Results**

This is the most important step for accuracy. The default throughput numbers are rough estimates for a 70B-class model — replace them with your actual measurements.

Run these two commands on your cluster using [vLLM's benchmark tool](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py):

```bash
# Input tokens/s — measures pure prefill (compute-bound)
python benchmark_throughput.py \
  --model <your-model> \
  --input-len 2048 \
  --output-len 1

# Output tokens/s — measures pure decode (memory-bandwidth-bound)
python benchmark_throughput.py \
  --model <your-model> \
  --input-len 1 \
  --output-len 256
```

Enter the reported `throughput (tokens/s)` values into the dashboard. The two benchmarks represent extremes — real workloads fall between them.

> **Why two separate numbers?** Prefill (processing your input) is compute-bound and runs ~8–15× faster than decode (generating output), which is memory-bandwidth-bound. Cloud providers price these differently (e.g. input tokens are 3–5× cheaper than output tokens). Separating them gives you an accurate apples-to-apples comparison.

**③ Effective Price / M Tokens**

This is the output of your infrastructure setup. Once you have your benchmark numbers, these two figures — **Input $/M** and **Output $/M** — are your self-hosted "rack rates". They tell you exactly what your GPU cluster charges per million tokens processed, equivalent to how cloud APIs publish their pricing.

The math:
```
cost per second = (numGPUs × $/hr × overheadMult) / 3600
Input  $/M = (cost per second / inputTokSec)  × 1,000,000
Output $/M = (cost per second / outputTokSec) × 1,000,000
```

**④ API Cache Hit Rate**

For cloud APIs that support prompt caching, adjust the slider to match your expected cache hit rate. This only affects API cost calculations — self-hosted cost is unaffected.

Prompt caching applies to input tokens only. Effective savings:
- Anthropic: 90% off input tokens on cache hit
- Doubao: ~60% off
- Qwen: ~60% off
- Fireworks: 50% off
- OpenAI batch API: no prompt caching

---

### Right Panel — Job Definition & Comparison

**⑤ Define Your Job**

Three inputs define any job:

| Field | What to enter |
|---|---|
| Input tokens / request | Token count of your prompt (system prompt + user message). Use a tokenizer or estimate: ~1 token per 0.75 words. |
| Output tokens / request | Token count of the model's response. Check your actual logs or run a sample. |
| Number of requests | 1 for a quick cost check on a single call. Your daily/weekly/monthly volume for scale estimates. |

The summary bar below shows per-request GPU time, total token volume, and self-hosted cost per request at a glance.

**⑥ APIs to Compare**

Toggle which cloud APIs appear in the comparison. Active APIs are highlighted. The ⬡ symbol marks open-source models — the same weights you could run yourself, making them the cleanest cost comparison (no model quality variable).

**⑦ Cost Comparison**

The main output. Self-hosted is shown prominently at the top, with a breakdown of input cost, output cost, and GPU rental. Each API is shown below with:
- Total job cost
- Cost relative to self-hosted (green = API is cheaper, red = self-host wins)
- Per-request cost and $/M rates
- A relative cost bar for visual comparison

**⑧ Cost at Scale**

Set **Batches / day** to see monthly projections. The table shows per-request, per-job, and per-month costs across all providers. Use this to understand when the crossover point is between self-hosting (fixed GPU cost) and APIs (pure variable cost).

---

## Maintainer Guide — Updating Prices

All prices and defaults live in the first ~50 lines of `llm-batch-cost-calculator.jsx`. No other part of the file needs to change for price updates.

### GPU Prices (`GPUS` array, lines 5–18)

```js
const GPUS = [
  { id:"h100-sxm", label:"H100 SXM 80GB", memGB:80,
    pricePerHr:3.44,               // ← update this
    defaultInputTokSec:6000,       // ← rough estimate only, users override
    defaultOutputTokSec:1200 },    // ← rough estimate only, users override
  ...
];
```

**Fields:**
- `pricePerHr` — on-demand market rate in USD. This is a starting point; users override with their contract rate.
- `defaultInputTokSec` / `defaultOutputTokSec` — rough estimates for a 70B-class model on this GPU. These are replaced by the user's vLLM benchmark. Update if the estimates are egregiously wrong for a new GPU generation.

**Where to find current GPU prices:**

| Provider | URL | Notes |
|---|---|---|
| Lambda Labs | https://lambdalabs.com/service/gpu-cloud | Most reliable reference for H100/A100/L40S |
| RunPod | https://www.runpod.io/gpu-instance/pricing | "Secure Cloud" tab for stable pricing |
| CoreWeave | https://www.coreweave.com/pricing | Contact for contract rates; list prices available |
| Vast.ai | https://vast.ai/pricing | Spot market — lowest but variable |
| TensorDock | https://tensordock.com/pricing | Good for RTX consumer cards |

Prices fluctuate monthly. Check 2–3 providers and use a representative average. Note the date in a comment when updating.

---

### Cloud API Prices (`APIS` array, lines 26–47)

```js
const APIS = [
  { id:"gpt4o",
    name:"GPT-4o",
    vendor:"OpenAI",
    inCPM:1.25,      // ← input $/M tokens (batch API rate)
    outCPM:5.00,     // ← output $/M tokens (batch API rate)
    cacheCPM:null,   // ← cached input $/M tokens (null if not supported)
    hasCache:false   // ← true if prompt caching is available
  },
  ...
];
```

**Fields:**
- `inCPM` — input price in **$/M tokens**. Use the batch API rate where available (typically 50% off standard).
- `outCPM` — output price in **$/M tokens**.
- `cacheCPM` — cached input price. Set to `null` if caching is not supported.
- `hasCache` — set to `true` if the provider supports prompt caching.
- `oss:true` — open-source weights flag (no functional effect, shown as ⬡ in UI).
- `flat:true` — flat rate flag (input and output same price, shown as "flat" badge).

**Where to find current API prices:**

| Provider | Pricing URL | Notes |
|---|---|---|
| OpenAI | https://openai.com/api/pricing | Use "Batch" column (50% off). No prompt caching in batch mode. |
| Anthropic | https://www.anthropic.com/pricing | Use "Batch" column. Prompt caching rate is `cacheCPM`. |
| Doubao (ByteDance) | https://www.volcengine.com/product/doubao | Prices in CNY — convert using `CNY` constant at top of file |
| Qwen (Alibaba CN) | https://bailian.console.aliyun.com/ → 模型广场 → 计费说明 | Prices in CNY. Batch prices are ~50% off standard. |
| Qwen (International) | https://www.alibabacloud.com/en/product/modelstudio/pricing | USD prices, Singapore endpoint |
| Fireworks AI | https://fireworks.ai/pricing | Flat $/M rate (input = output). Check "Open Source Models" section. |

**Qwen CNY conversion:**

Qwen-CN prices are stored in CNY and converted at runtime using the `CNY` constant at line 3:

```js
const CNY = 7.28;  // ← update this if CNY/USD rate shifts significantly
```

Qwen-CN entries use inline conversion:
```js
{ id:"qwen3max", inCPM:+(1.25/CNY).toFixed(4), outCPM:+(5.00/CNY).toFixed(4), ... }
//                        ↑ ¥1.25 per M input        ↑ ¥5.00 per M output
```

To update: change the CNY numerator (the yuan price), not the formula.

---

### Adding a New API Provider

Copy an existing entry and fill in the fields:

```js
{ id:"my-provider",           // unique string, no spaces
  name:"My Provider Model",   // display name in UI
  vendor:"MyVendor",          // must match a key in VENDOR_COLOR
  inCPM:0.50,                 // $/M input tokens
  outCPM:2.00,                // $/M output tokens
  cacheCPM:0.05,              // $/M cached input, or null
  hasCache:true,              // true if caching available
  note:"optional note",       // shown in comparison table
  oss:true,                   // optional: open-source weights
  flat:true,                  // optional: flat rate (in=out price)
},
```

Then add the vendor to `VENDOR_COLOR` (if new) and `VENDORS` array:

```js
const VENDOR_COLOR = {
  ...
  "MyVendor": "#AABBCC",   // hex color for this vendor's UI elements
};

const VENDORS = [..., "MyVendor"];   // controls display order
```

---

### Adding a New GPU

```js
{ id:"h200",                    // unique string
  label:"H200 SXM 141GB",      // display name
  memGB:141,                    // VRAM in GB (shown in GPU picker)
  pricePerHr:4.50,              // on-demand market rate
  defaultInputTokSec:8000,      // rough prefill estimate for 70B model
  defaultOutputTokSec:1600,     // rough decode estimate for 70B model
},
```

Throughput defaults are order-of-magnitude estimates — they're clearly marked "estimated" in the UI and are always meant to be overridden by real benchmarks. A reasonable starting point: scale linearly from a known GPU's numbers based on memory bandwidth ratio.

---

## Price Verification Checklist

When doing a price review, check these sources in order:

- [ ] GPU rates: Lambda Labs + RunPod (update `GPUS[*].pricePerHr`)
- [ ] OpenAI batch pricing: https://openai.com/api/pricing
- [ ] Anthropic batch + cache pricing: https://www.anthropic.com/pricing
- [ ] Doubao pricing: Volcengine console (convert CNY)
- [ ] Qwen-CN batch pricing: Alibaba Cloud Bailian console (convert CNY)
- [ ] Qwen-Intl pricing: Alibaba Cloud international site
- [ ] Fireworks flat rates: https://fireworks.ai/pricing
- [ ] CNY/USD rate: update `const CNY` if moved more than ~3%
- [ ] Update the date comment in the footer (bottom of file, search "Feb–Mar 2026")

Prices in this space change frequently — monthly reviews are recommended.
