import { useState, useMemo } from "react";

const CNY = 7.28;

const GPUS = [
  { id:"b200",      label:"B200 SXM 180GB",     memGB:180, pricePerHr:5.74, defaultInputTokSec:10000, defaultOutputTokSec:2000 },
  { id:"h100-sxm",  label:"H100 SXM 80GB",      memGB:80,  pricePerHr:3.44, defaultInputTokSec:6000,  defaultOutputTokSec:1200 },
  { id:"h100-pcie", label:"H100 PCIe 80GB",      memGB:80,  pricePerHr:2.86, defaultInputTokSec:4500,  defaultOutputTokSec:900  },
  { id:"gh200",     label:"GH200 96GB",           memGB:96,  pricePerHr:1.99, defaultInputTokSec:6500,  defaultOutputTokSec:1300 },
  { id:"a100-80",   label:"A100 80GB",            memGB:80,  pricePerHr:2.06, defaultInputTokSec:4000,  defaultOutputTokSec:800  },
  { id:"a100-40",   label:"A100 40GB",            memGB:40,  pricePerHr:1.48, defaultInputTokSec:2200,  defaultOutputTokSec:450  },
  { id:"l40s",      label:"L40S / 6000 Ada 48GB", memGB:48,  pricePerHr:1.35, defaultInputTokSec:3500,  defaultOutputTokSec:430  },
  { id:"a6000",     label:"A6000 / A40 48GB",     memGB:48,  pricePerHr:1.10, defaultInputTokSec:2000,  defaultOutputTokSec:280  },
  { id:"rtx5090",   label:"RTX 5090 32GB",        memGB:32,  pricePerHr:1.10, defaultInputTokSec:1500,  defaultOutputTokSec:200  },
  { id:"rtx4090",   label:"RTX 4090 24GB",        memGB:24,  pricePerHr:0.69, defaultInputTokSec:1200,  defaultOutputTokSec:160  },
  { id:"a10",       label:"A10 / A10G 24GB",      memGB:24,  pricePerHr:0.86, defaultInputTokSec:1000,  defaultOutputTokSec:120  },
  { id:"l4-3090",   label:"L4 / RTX 3090 24GB",   memGB:24,  pricePerHr:0.58, defaultInputTokSec:700,   defaultOutputTokSec:80   },
];
const GPU_BY_ID = Object.fromEntries(GPUS.map(g => [g.id, g]));

const VENDOR_COLOR = {
  OpenAI:"#10A37F", Anthropic:"#CC785C", Doubao:"#4F8EF7",
  "Qwen-CN":"#E8943A", "Qwen-Intl":"#F0C060", Fireworks:"#B06EFF",
};

const APIS = [
  { id:"gpt4o",       name:"GPT-4o",               vendor:"OpenAI",    inCPM:1.25,  outCPM:5.00,  cacheCPM:null,  hasCache:false },
  { id:"gpt4o-mini",  name:"GPT-4o-mini",          vendor:"OpenAI",    inCPM:0.075, outCPM:0.30,  cacheCPM:null,  hasCache:false },
  { id:"gpt41",       name:"GPT-4.1",               vendor:"OpenAI",    inCPM:1.00,  outCPM:4.00,  cacheCPM:null,  hasCache:false },
  { id:"gpt41-mini",  name:"GPT-4.1-mini",         vendor:"OpenAI",    inCPM:0.20,  outCPM:0.80,  cacheCPM:null,  hasCache:false },
  { id:"sonnet45",    name:"Claude Sonnet 4.5",    vendor:"Anthropic", inCPM:1.50,  outCPM:7.50,  cacheCPM:0.15,  hasCache:true  },
  { id:"haiku45",     name:"Claude Haiku 4.5",     vendor:"Anthropic", inCPM:0.50,  outCPM:2.50,  cacheCPM:0.05,  hasCache:true  },
  { id:"haiku35",     name:"Claude Haiku 3.5",     vendor:"Anthropic", inCPM:0.40,  outCPM:2.00,  cacheCPM:0.04,  hasCache:true  },
  { id:"doubao-pro",  name:"Doubao 1.5-pro",       vendor:"Doubao",    inCPM:0.056, outCPM:0.141, cacheCPM:0.023, hasCache:true  },
  { id:"doubao-lite", name:"Doubao 1.5-lite",      vendor:"Doubao",    inCPM:0.021, outCPM:0.042, cacheCPM:0.008, hasCache:true  },
  { id:"qwen3max",    name:"Qwen3-Max",             vendor:"Qwen-CN",   inCPM:+(1.25/CNY).toFixed(4),   outCPM:+(5.00/CNY).toFixed(4),   cacheCPM:+(0.50/CNY).toFixed(4),   hasCache:true,  note:"¥1.25/¥5 per M" },
  { id:"qwen-plus",   name:"Qwen-Plus",             vendor:"Qwen-CN",   inCPM:+(0.40/CNY).toFixed(4),   outCPM:+(1.00/CNY).toFixed(4),   cacheCPM:null,                     hasCache:false, note:"¥0.4/¥1 per M" },
  { id:"qwen-flash",  name:"Qwen-Flash",            vendor:"Qwen-CN",   inCPM:+(0.075/CNY).toFixed(5),  outCPM:+(0.75/CNY).toFixed(4),   cacheCPM:+(0.030/CNY).toFixed(5),  hasCache:true,  note:"¥0.075/¥0.75 per M" },
  { id:"qwen3-32b",   name:"Qwen3-32B (OSS)",      vendor:"Qwen-CN",   inCPM:+(2.00/CNY).toFixed(4),   outCPM:+(8.00/CNY).toFixed(4),   cacheCPM:null,                     hasCache:false, note:"same weights, no batch discount", oss:true },
  { id:"qwen3-30b",   name:"Qwen3-30B-A3B (OSS)",  vendor:"Qwen-CN",   inCPM:+(0.75/CNY).toFixed(4),   outCPM:+(3.00/CNY).toFixed(4),   cacheCPM:null,                     hasCache:false, note:"MoE, same weights", oss:true },
  { id:"qwen-flash-i",name:"Qwen-Flash (Intl)",    vendor:"Qwen-Intl", inCPM:+(0.367/CNY/2).toFixed(5), outCPM:+(2.936/CNY/2).toFixed(4),cacheCPM:+(0.147/CNY/2).toFixed(5),hasCache:true,  note:"SG endpoint" },
  { id:"qwen-plus-i", name:"Qwen-Plus (Intl)",     vendor:"Qwen-Intl", inCPM:+(1.468/CNY/2).toFixed(4), outCPM:+(4.404/CNY/2).toFixed(4),cacheCPM:null,                    hasCache:false, note:"SG endpoint" },
  { id:"fw-70b",      name:"Fireworks >16B",        vendor:"Fireworks", inCPM:0.45, outCPM:0.45, cacheCPM:0.225, hasCache:true, note:"Llama-3.1-70B, Qwen2.5-72B", oss:true, flat:true },
  { id:"fw-moe-s",    name:"Fireworks MoE 0–56B",  vendor:"Fireworks", inCPM:0.25, outCPM:0.25, cacheCPM:0.125, hasCache:true, note:"Qwen3-30B-A3B", oss:true, flat:true },
  { id:"fw-moe-l",    name:"Fireworks MoE 56–176B",vendor:"Fireworks", inCPM:0.60, outCPM:0.60, cacheCPM:0.30,  hasCache:true, note:"Qwen3-235B-A22B", oss:true, flat:true },
  { id:"fw-small",    name:"Fireworks 4B–16B",     vendor:"Fireworks", inCPM:0.10, outCPM:0.10, cacheCPM:0.05,  hasCache:true, note:"Qwen3-14B", oss:true, flat:true },
];

const VENDORS = ["OpenAI","Anthropic","Doubao","Qwen-CN","Qwen-Intl","Fireworks"];

// ─── Compute ──────────────────────────────────────────────────────────────────
// cost per token = (GPU $/s) / throughput
// inputCPM  = (gpuCostPerHr / 3600 / inputTokSec)  * 1e6
// outputCPM = (gpuCostPerHr / 3600 / outputTokSec) * 1e6
function deriveRates({ pricePerHr, numGpus, inputTokSec, outputTokSec, overheadMult }) {
  const gpuCostPerHr = numGpus * pricePerHr * overheadMult;
  const costPerSec   = gpuCostPerHr / 3600;
  return {
    inputCPM:    inputTokSec  > 0 ? (costPerSec / inputTokSec)  * 1e6 : 0,
    outputCPM:   outputTokSec > 0 ? (costPerSec / outputTokSec) * 1e6 : 0,
    gpuCostPerHr,
  };
}

function reqCostAt(inTok, outTok, inCPM, outCPM) {
  return (inTok * inCPM + outTok * outCPM) / 1e6;
}

function effInCPM(api, cacheHitPct) {
  const h = cacheHitPct / 100;
  return (api.hasCache && api.cacheCPM)
    ? (1 - h) * api.inCPM + h * api.cacheCPM
    : api.inCPM;
}

// ─── Format helpers ───────────────────────────────────────────────────────────
const f$ = n => {
  if (n === 0) return "$0.00";
  if (n < 0.0001) return "$" + n.toFixed(6);
  if (n < 0.01)   return "$" + n.toFixed(4);
  if (n < 1)      return "$" + n.toFixed(4);
  if (n < 1000)   return "$" + n.toFixed(2);
  if (n < 1e6)    return "$" + (n/1000).toFixed(1) + "K";
  return "$" + (n/1e6).toFixed(2) + "M";
};
const fM  = n => n >= 1e6 ? "$" + (n/1e6).toFixed(2) + "M" : n >= 1000 ? "$" + (n/1000).toFixed(1) + "K" : "$" + n.toFixed(2);
const fCPM = n => n < 0.001 ? "$" + n.toFixed(5) : n < 0.01 ? "$" + n.toFixed(4) : "$" + n.toFixed(3);

// ─── Styles ───────────────────────────────────────────────────────────────────
const panel = { background:"#0C1520", border:"1px solid #1C2B3A", borderRadius:12, padding:16, marginBottom:12 };
const lbl   = { fontSize:10, fontWeight:700, letterSpacing:"0.08em", color:"#4A6A80", textTransform:"uppercase", display:"block", marginBottom:5 };
const inp   = { background:"#070D14", border:"1px solid #1C2B3A", borderRadius:6, padding:"7px 10px",
                color:"#D8E8F0", fontSize:14, fontWeight:600, outline:"none", width:"100%", boxSizing:"border-box",
                fontFamily:"inherit", fontVariantNumeric:"tabular-nums" };

// ─── UI Primitives ────────────────────────────────────────────────────────────
function SecHead({ icon, children }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14, paddingBottom:10, borderBottom:"1px solid #1C2B3A" }}>
      <span style={{ fontSize:13 }}>{icon}</span>
      <span style={{ fontSize:10, fontWeight:800, letterSpacing:"0.14em", color:"#3A8FBF", textTransform:"uppercase" }}>
        {children}
      </span>
    </div>
  );
}

function FieldInput({ label, value, onChange, placeholder, suffix, accentColor, note }) {
  const active = value !== "";
  return (
    <div style={{ marginBottom:11 }}>
      <label style={{ ...lbl, color: accentColor && active ? accentColor : "#4A6A80" }}>
        {label}
        {accentColor && active && <span style={{ marginLeft:6, color:accentColor, fontWeight:400 }}>← your value</span>}
      </label>
      <div style={{ display:"flex", alignItems:"center", background:"#070D14",
                    border:"1px solid " + (accentColor && active ? accentColor + "55" : "#1C2B3A"),
                    borderRadius:6, overflow:"hidden" }}>
        <input type="number" value={value} onChange={e => onChange(e.target.value)}
          placeholder={placeholder != null ? String(placeholder) : ""}
          style={{ ...inp, border:"none", background:"transparent", flex:1,
                   color: accentColor && active ? accentColor : "#D8E8F0" }}/>
        {suffix && <span style={{ padding:"0 10px", fontSize:11, color:"#2A4A5A", whiteSpace:"nowrap" }}>{suffix}</span>}
      </div>
      {note && <div style={{ fontSize:9, color:"#2A3A4A", marginTop:3, lineHeight:1.5 }}>{note}</div>}
    </div>
  );
}

function TokenInput({ label, value, onChange, color }) {
  return (
    <div>
      <label style={{ ...lbl }}>{label}</label>
      <div style={{ display:"flex", alignItems:"center", background:"#070D14",
                    border:"1px solid " + color + "55", borderRadius:6, overflow:"hidden" }}>
        <input type="number" value={value} min={0}
          onChange={e => onChange(Math.max(0, Number(e.target.value)))}
          style={{ ...inp, border:"none", background:"transparent", flex:1, color }}/>
      </div>
      <div style={{ fontSize:9, color:"#243040", marginTop:3 }}>
        {Number(value).toLocaleString() + " tokens / request"}
      </div>
    </div>
  );
}

function Slider({ label, value, onChange, min, max, step=1, suffix="" }) {
  return (
    <div style={{ marginBottom:14 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
        <label style={lbl}>{label}</label>
        <span style={{ fontSize:12, fontWeight:700, color:"#C8D8E0" }}>{value}{suffix}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width:"100%", accentColor:"#3A6080", cursor:"pointer" }}/>
    </div>
  );
}

function RateBox({ label, cpm, tokSec, measured, color }) {
  return (
    <div style={{ background:"#070D14", border:"1px solid " + color + "33", borderRadius:8, padding:"14px 16px" }}>
      <div style={{ fontSize:10, color:color + "88", textTransform:"uppercase", letterSpacing:"0.08em", marginBottom:6 }}>{label}</div>
      <div style={{ fontSize:28, fontWeight:800, color, fontVariantNumeric:"tabular-nums", letterSpacing:"-0.02em" }}>
        {fCPM(cpm)}
      </div>
      <div style={{ fontSize:11, color:"#304050", marginTop:4 }}>per million tokens</div>
      <div style={{ fontSize:10, color:"#2A4050", marginTop:6, borderTop:"1px solid #1A2A38", paddingTop:6 }}>
        {tokSec.toLocaleString() + " tokens/s " + (measured ? "✓ measured" : "(estimated)")}
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {

  // Infrastructure
  const [gpuId,      setGpuId]      = useState("l40s");
  const [numGpus,    setNumGpus]    = useState("8");
  const [custPrice,  setCustPrice]  = useState("");
  const [custIn,     setCustIn]     = useState("");
  const [custOut,    setCustOut]    = useState("");
  const [startupPct,   setStartupPct]   = useState(2);
  const [interruptPct, setInterruptPct] = useState(1);

  // Batch job
  const [reqIn,        setReqIn]        = useState(2000);
  const [reqOut,       setReqOut]       = useState(500);
  const [numRequests,  setNumRequests]  = useState(1000);
  const [batchesPerDay,setBatchesPerDay]= useState(10);

  // API config
  const [cacheHit, setCacheHit] = useState(50);
  const defaultSel = ["gpt4o","gpt4o-mini","haiku45","doubao-pro","qwen3max","qwen-flash","qwen3-32b","fw-70b","fw-moe-s"];
  const [selAPIs, setSelAPIs] = useState(defaultSel);
  const toggle = id => setSelAPIs(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id]);

  // ── Derived infra values ─────────────────────────────────────────────────────
  const gpu = GPU_BY_ID[gpuId];
  const n   = Math.max(1, parseInt(numGpus) || 1);
  const pricePerHr   = custPrice !== "" ? Number(custPrice) : gpu.pricePerHr;
  const inputTokSec  = custIn    !== "" ? Number(custIn)    : gpu.defaultInputTokSec;
  const outputTokSec = custOut   !== "" ? Number(custOut)   : gpu.defaultOutputTokSec;
  const overheadMult = (1 + startupPct / 100) * (1 + interruptPct / 100);

  const { inputCPM, outputCPM, gpuCostPerHr } = useMemo(
    () => deriveRates({ pricePerHr, numGpus: n, inputTokSec, outputTokSec, overheadMult }),
    [pricePerHr, n, inputTokSec, outputTokSec, overheadMult]
  );

  // ── Per-request cost ─────────────────────────────────────────────────────────
  const selfPerReq  = useMemo(() => reqCostAt(reqIn, reqOut, inputCPM, outputCPM), [reqIn, reqOut, inputCPM, outputCPM]);
  const selfJobCost = selfPerReq * numRequests;
  const selfDayCost = selfJobCost * batchesPerDay;
  const selfMooCost = selfDayCost * 30;

  const apiRows = useMemo(() =>
    selAPIs.map(id => {
      const api = APIS.find(a => a.id === id);
      if (!api) return null;
      const eIn    = effInCPM(api, cacheHit);
      const perReq = reqCostAt(reqIn, reqOut, eIn, api.outCPM);
      return { id, api, eIn, perReq, jobCost: perReq * numRequests };
    }).filter(Boolean),
    [selAPIs, reqIn, reqOut, numRequests, cacheHit]
  );

  const allJobCosts = [selfJobCost, ...apiRows.map(r => r.jobCost)];
  const maxJobCost  = Math.max(...allJobCosts, 0.0001);

  // Per-request timing
  const reqInSec  = inputTokSec  > 0 ? reqIn  / inputTokSec  : 0;
  const reqOutSec = outputTokSec > 0 ? reqOut / outputTokSec : 0;

  return (
    <div style={{ fontFamily:"'DM Mono','Fira Code',monospace", background:"#080F18",
                  minHeight:"100vh", color:"#B8CCE0", padding:"20px 16px" }}>

      {/* Header */}
      <div style={{ marginBottom:20, borderBottom:"1px solid #1C2B3A", paddingBottom:12 }}>
        <div style={{ fontSize:9, letterSpacing:"0.25em", color:"#3A8FBF", textTransform:"uppercase", marginBottom:2 }}>LLM Infrastructure</div>
        <h1 style={{ margin:0, fontSize:22, fontWeight:800, color:"#E0EEF8", letterSpacing:"-0.03em" }}>AIBrix Batch Cost Calculator</h1>
        <p style={{ margin:"4px 0 0", fontSize:10, color:"#2A4A5A" }}>
          Configure GPU cluster → benchmark throughput → price any job vs cloud APIs
        </p>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"300px 1fr", gap:16, alignItems:"start" }}>

        {/* ═══ LEFT: Infrastructure ═══ */}
        <div>

          {/* ① GPU Setup */}
          <div style={panel}>
            <SecHead icon="①">GPU Setup</SecHead>
            <label style={lbl}>GPU Type</label>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:3, marginBottom:14 }}>
              {GPUS.map(g => (
                <div key={g.id} onClick={() => setGpuId(g.id)}
                  style={{ cursor:"pointer", padding:"6px 8px", borderRadius:6,
                           background: gpuId === g.id ? "rgba(58,143,191,0.15)" : "#070D14",
                           border: gpuId === g.id ? "1px solid rgba(58,143,191,0.5)" : "1px solid #131F2A",
                           transition:"all 0.12s" }}>
                  <div style={{ fontSize:10, fontWeight: gpuId === g.id ? 700 : 400,
                                color: gpuId === g.id ? "#90C8E8" : "#607080",
                                overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                    {g.label}
                  </div>
                  <div style={{ display:"flex", justifyContent:"space-between", marginTop:2 }}>
                    <span style={{ fontSize:8, color:"#243040" }}>{g.memGB + "GB"}</span>
                    <span style={{ fontSize:9, fontWeight:700, color: gpuId === g.id ? "#3A8FBF" : "#304050" }}>
                      {"$" + g.pricePerHr + "/hr"}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
              <FieldInput label="GPU Count" value={numGpus} onChange={setNumGpus}
                placeholder={8} suffix="GPUs"/>
              <FieldInput label="$/GPU/hr override" value={custPrice} onChange={setCustPrice}
                placeholder={gpu.pricePerHr} accentColor="#A8E6CF"/>
            </div>
            <div style={{ fontSize:10, color:"#243040", marginBottom:12 }}>
              {"Total rental: $" + (pricePerHr * n * overheadMult).toFixed(2) + "/hr (incl. overhead)"}
            </div>

            <Slider label="Startup overhead" value={startupPct} onChange={setStartupPct} min={0} max={20} suffix="%"/>
            <Slider label="Interruption rate" value={interruptPct} onChange={setInterruptPct} min={0} max={20} suffix="%"/>
          </div>

          {/* ② Benchmark */}
          <div style={panel}>
            <SecHead icon="②">Benchmark Results</SecHead>
            <div style={{ background:"#060C14", border:"1px solid #131F2A", borderRadius:6,
                          padding:"9px 11px", marginBottom:12, fontSize:9, lineHeight:2, color:"#243040" }}>
              <div style={{ color:"#3A8FBF", fontWeight:700, marginBottom:2 }}>Run on your cluster (vLLM):</div>
              <div style={{ color:"#F0C060" }}>Input tokens/s — prefill:</div>
              <div style={{ fontFamily:"monospace", fontSize:8, color:"#304050", marginBottom:4 }}>
                {"python benchmark_throughput.py --input-len 2048 --output-len 1"}
              </div>
              <div style={{ color:"#C09EFF" }}>Output tokens/s — decode:</div>
              <div style={{ fontFamily:"monospace", fontSize:8, color:"#304050" }}>
                {"python benchmark_throughput.py --input-len 1 --output-len 256"}
              </div>
            </div>
            <FieldInput label="Input tokens/s (prefill)" value={custIn} onChange={setCustIn}
              placeholder={gpu.defaultInputTokSec} accentColor="#F0C060"
              note={custIn === "" ? "Rough 70B estimate — replace with your benchmark" : ""}/>
            <FieldInput label="Output tokens/s (decode)" value={custOut} onChange={setCustOut}
              placeholder={gpu.defaultOutputTokSec} accentColor="#C09EFF"
              note={custOut === "" ? "Decode is ~8–15x slower than prefill" : ""}/>
          </div>

          {/* ③ Derived rates */}
          <div style={{ ...panel, border:"1px solid #1E3A50" }}>
            <SecHead icon="③">Effective Price / M Tokens</SecHead>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:10 }}>
              <RateBox label="Input $/M"  cpm={inputCPM}  tokSec={inputTokSec}  measured={custIn !== ""}  color="#F0C060"/>
              <RateBox label="Output $/M" cpm={outputCPM} tokSec={outputTokSec} measured={custOut !== ""} color="#C09EFF"/>
            </div>
            <div style={{ fontSize:10, color:"#243040", lineHeight:1.8, borderTop:"1px solid #131F2A", paddingTop:8 }}>
              {"$/s = $" + (pricePerHr * n * overheadMult / 3600).toFixed(5) + "  ·  $/token = $/s ÷ tokens/s"}
              <div style={{ color:"#1E3A50", marginTop:2 }}>{"Faster benchmark → lower $/M. GPU rental stays fixed."}</div>
            </div>
          </div>

          {/* ④ Cache */}
          <div style={panel}>
            <SecHead icon="④">API Cache Hit Rate</SecHead>
            <Slider label="Cache hit rate (input tokens)" value={cacheHit} onChange={setCacheHit} min={0} max={100} suffix="%"/>
            <div style={{ fontSize:10, color:"#304050", lineHeight:2 }}>
              <div>{"🟠 Anthropic · 90% off input on hit"}</div>
              <div>{"🔵 Doubao · 60% off input on hit"}</div>
              <div>{"🟡 Qwen · ~60% off input on hit"}</div>
              <div>{"🟣 Fireworks · 50% off input on hit"}</div>
            </div>
          </div>

        </div>

        {/* ═══ RIGHT: Job + Comparison ═══ */}
        <div>

          {/* ⑤ Define job */}
          <div style={panel}>
            <SecHead icon="⑤">Define Your Job</SecHead>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12 }}>
              <TokenInput label="Input tokens / request"  value={reqIn}  onChange={setReqIn}  color="#F0C060"/>
              <TokenInput label="Output tokens / request" value={reqOut} onChange={setReqOut} color="#C09EFF"/>
              <div>
                <label style={lbl}>Number of requests</label>
                <div style={{ display:"flex", alignItems:"center", background:"#070D14",
                              border:"1px solid #1C2B3A", borderRadius:6, overflow:"hidden" }}>
                  <input type="number" value={numRequests} min={1}
                    onChange={e => setNumRequests(Math.max(1, Number(e.target.value)))}
                    style={{ ...inp, border:"none", background:"transparent", flex:1 }}/>
                </div>
                <div style={{ fontSize:9, color:"#243040", marginTop:3 }}>
                  {numRequests === 1 ? "single request" : numRequests.toLocaleString() + " total"}
                </div>
              </div>
            </div>

            {/* Timing + totals summary */}
            <div style={{ marginTop:12, background:"#070D14", borderRadius:8, padding:"10px 14px",
                          display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10 }}>
              <div>
                <div style={{ fontSize:9, color:"#1A2A38", marginBottom:3, textTransform:"uppercase", letterSpacing:"0.06em" }}>Per request time</div>
                <div style={{ fontSize:12, color:"#F0C06099" }}>{reqInSec.toFixed(2) + "s prefill"}</div>
                <div style={{ fontSize:12, color:"#C09EFF99" }}>{reqOutSec.toFixed(2) + "s decode"}</div>
              </div>
              <div>
                <div style={{ fontSize:9, color:"#1A2A38", marginBottom:3, textTransform:"uppercase", letterSpacing:"0.06em" }}>Total tokens</div>
                <div style={{ fontSize:12, color:"#F0C06099" }}>{(reqIn * numRequests / 1e6).toFixed(2) + "M input"}</div>
                <div style={{ fontSize:12, color:"#C09EFF99" }}>{(reqOut * numRequests / 1e6).toFixed(2) + "M output"}</div>
              </div>
              <div>
                <div style={{ fontSize:9, color:"#1A2A38", marginBottom:3, textTransform:"uppercase", letterSpacing:"0.06em" }}>Self-hosted</div>
                <div style={{ fontSize:15, fontWeight:700, color:"#4FAAFF" }}>{f$(selfPerReq) + " / req"}</div>
                <div style={{ fontSize:12, color:"#3A6A90" }}>{fM(selfJobCost) + " total"}</div>
              </div>
            </div>
          </div>

          {/* ⑥ APIs to compare — moved above cost comparison */}
          <div style={panel}>
            <SecHead icon="⑥">APIs to Compare</SecHead>
            {VENDORS.map(vendor => {
              const apis = APIS.filter(a => a.vendor === vendor);
              const vc = VENDOR_COLOR[vendor];
              return (
                <div key={vendor} style={{ marginBottom:10 }}>
                  <div style={{ fontSize:9, color:vc, textTransform:"uppercase", fontWeight:700,
                                letterSpacing:"0.1em", marginBottom:5, display:"flex", gap:6, alignItems:"center" }}>
                    {vendor}
                    {vendor === "Fireworks" && <span style={{ color:"#304050", fontWeight:400, textTransform:"none" }}>{"· open-source weights · flat $/M"}</span>}
                    {vendor === "Qwen-CN"   && <span style={{ color:"#304050", fontWeight:400, textTransform:"none" }}>{"· ¥→USD"}</span>}
                    {vendor === "Qwen-Intl" && <span style={{ color:"#304050", fontWeight:400, textTransform:"none" }}>{"· Singapore"}</span>}
                  </div>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:4 }}>
                    {apis.map(a => {
                      const sel = selAPIs.includes(a.id);
                      return (
                        <div key={a.id} onClick={() => toggle(a.id)}
                          style={{ cursor:"pointer", padding:"4px 10px", borderRadius:4, fontSize:10,
                                   background: sel ? vc + "18" : "#070D14",
                                   border: "1px solid " + (sel ? vc + "66" : "#131F2A"),
                                   color: sel ? "#C8D8E8" : "#304050", transition:"all 0.12s" }}>
                          {a.oss && <span style={{ color:"#B06EFF", marginRight:3 }}>{"⬡"}</span>}
                          {a.name}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
            <div style={{ fontSize:9, color:"#1A2A38", marginTop:4, borderTop:"1px solid #131F2A", paddingTop:6 }}>
              <span style={{ color:"#B06EFF" }}>{"⬡"}</span>
              {" = open-source weights — same model you can self-host"}
            </div>
          </div>

          {/* ⑦ Cost per job — main comparison */}
          <div style={panel}>
            <SecHead icon="⑦">
              {"Cost Comparison · " + numRequests.toLocaleString() + " request" + (numRequests === 1 ? "" : "s")}
            </SecHead>

            {/* Self-hosted — prominent */}
            <div style={{ background:"rgba(58,143,191,0.08)", border:"1px solid rgba(58,143,191,0.25)",
                          borderRadius:10, padding:"14px 16px", marginBottom:12 }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:10 }}>
                <div>
                  <div style={{ fontSize:13, fontWeight:700, color:"#90C8E8", marginBottom:2 }}>
                    {"🖥  Self-hosted · " + n + "× " + gpu.label}
                  </div>
                  {(custIn === "" || custOut === "") &&
                    <div style={{ fontSize:10, color:"#3A6080", fontStyle:"italic" }}>using estimated throughput</div>}
                </div>
                <div style={{ textAlign:"right" }}>
                  <div style={{ fontSize:30, fontWeight:800, color:"#4FAAFF", fontVariantNumeric:"tabular-nums", lineHeight:1 }}>
                    {fM(selfJobCost)}
                  </div>
                  <div style={{ fontSize:11, color:"#2A4A5A", marginTop:3 }}>{"total · " + f$(selfPerReq) + " per request"}</div>
                </div>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8 }}>
                <div style={{ background:"#060C14", borderRadius:6, padding:"8px 10px" }}>
                  <div style={{ fontSize:10, color:"#4A6A80", marginBottom:4 }}>Input cost</div>
                  <div style={{ fontSize:18, fontWeight:700, color:"#F0C060" }}>
                    {fM(reqIn * numRequests * inputCPM / 1e6)}
                  </div>
                  <div style={{ fontSize:11, color:"#5A4A20", marginTop:3 }}>
                    {fCPM(inputCPM) + " / M tokens"}
                  </div>
                  <div style={{ fontSize:11, color:"#3A3020", marginTop:1 }}>
                    {f$(reqIn * inputCPM / 1e6) + " / request"}
                  </div>
                </div>
                <div style={{ background:"#060C14", borderRadius:6, padding:"8px 10px" }}>
                  <div style={{ fontSize:10, color:"#4A6A80", marginBottom:4 }}>Output cost</div>
                  <div style={{ fontSize:18, fontWeight:700, color:"#C09EFF" }}>
                    {fM(reqOut * numRequests * outputCPM / 1e6)}
                  </div>
                  <div style={{ fontSize:11, color:"#503A70", marginTop:3 }}>
                    {fCPM(outputCPM) + " / M tokens"}
                  </div>
                  <div style={{ fontSize:11, color:"#3A2A50", marginTop:1 }}>
                    {f$(reqOut * outputCPM / 1e6) + " / request"}
                  </div>
                </div>
                <div style={{ background:"#060C14", borderRadius:6, padding:"8px 10px" }}>
                  <div style={{ fontSize:10, color:"#4A6A80", marginBottom:4 }}>GPU rental</div>
                  <div style={{ fontSize:18, fontWeight:700, color:"#8A9BB0" }}>
                    {f$(gpuCostPerHr) + "/hr"}
                  </div>
                  <div style={{ fontSize:11, color:"#304050", marginTop:3 }}>
                    {"fixed · " + n + " GPU" + (n === 1 ? "" : "s")}
                  </div>
                  <div style={{ fontSize:11, color:"#243040", marginTop:1 }}>
                    {"regardless of load"}
                  </div>
                </div>
              </div>
            </div>

            {/* API rows */}
            {apiRows.map(({ id, api, eIn, perReq, jobCost }) => {
              const vc      = VENDOR_COLOR[api.vendor];
              const barPct  = Math.max((jobCost / maxJobCost) * 100, 1.5);
              const selfRatio = jobCost / selfJobCost;
              const cheaper = selfRatio < 1;
              const vLabel  = (api.vendor === "Qwen-CN" || api.vendor === "Qwen-Intl") ? "Qwen" : api.vendor;
              return (
                <div key={id} style={{ borderBottom:"1px solid #131F2A", paddingTop:10, paddingBottom:10 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:6 }}>
                    <div style={{ display:"flex", alignItems:"center", gap:6, flex:1, minWidth:0 }}>
                      <span style={{ fontSize:12, color:"#7A9AB0", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", maxWidth:240 }}>
                        {api.oss && <span style={{ color:"#B06EFF", marginRight:3 }}>{"⬡"}</span>}
                        {vLabel + " · " + api.name}
                      </span>
                      {api.flat && <span style={{ fontSize:9, padding:"1px 5px", borderRadius:3, flexShrink:0,
                                                  background:"#9060CC22", color:"#9060CC", border:"1px solid #9060CC44" }}>{"flat"}</span>}
                    </div>
                    <div style={{ display:"flex", alignItems:"center", gap:12, flexShrink:0 }}>
                      <span style={{ fontSize:11, fontWeight:700, color: cheaper ? "#6EE7B7" : "#F87171" }}>
                        {cheaper
                          ? (((1 - selfRatio) * 100).toFixed(0) + "% cheaper")
                          : ("+" + (((selfRatio - 1) * 100).toFixed(0)) + "% vs self")}
                      </span>
                      <span style={{ fontSize:18, fontWeight:700, color:"#C8D8E8", fontVariantNumeric:"tabular-nums", minWidth:80, textAlign:"right" }}>
                        {fM(jobCost)}
                      </span>
                    </div>
                  </div>
                  <div style={{ height:5, background:"#0D1820", borderRadius:3, marginBottom:5 }}>
                    <div style={{ width: barPct + "%", height:"100%", background:vc, borderRadius:3, opacity:0.7 }}/>
                  </div>
                  <div style={{ display:"flex", gap:14, fontSize:11, color:"#304050" }}>
                    <span>{"in: " + fCPM(eIn) + "/M"}</span>
                    <span>{"out: " + fCPM(api.outCPM) + "/M"}</span>
                    <span>{"per req: " + f$(perReq)}</span>
                    {api.note && <span style={{ color:"#1A2A38" }}>{api.note}</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* ⑧ Scale */}
          <div style={panel}>
            <SecHead icon="⑧">Cost at Scale</SecHead>

            {/* Batches/day input */}
            <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:14 }}>
              <label style={{ ...lbl, marginBottom:0, whiteSpace:"nowrap" }}>Batches / day</label>
              <div style={{ display:"flex", alignItems:"center", background:"#070D14",
                            border:"1px solid #1C2B3A", borderRadius:6, overflow:"hidden", width:120 }}>
                <input type="number" value={batchesPerDay} min={1}
                  onChange={e => setBatchesPerDay(Math.max(1, Number(e.target.value)))}
                  style={{ ...inp, border:"none", background:"transparent", width:"100%" }}/>
              </div>
              <span style={{ fontSize:10, color:"#304050" }}>
                {"→ " + (batchesPerDay * 30).toLocaleString() + " / month"}
              </span>
            </div>

            {/* Scale cards */}
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8, marginBottom:14 }}>
              {[
                { label:"Per request",  val: selfPerReq,             note:"1 prompt → 1 response" },
                { label:"Per job",      val: selfJobCost,            note: numRequests.toLocaleString() + " requests" },
                { label:"Per month",    val: selfMooCost,            note: (batchesPerDay * 30).toLocaleString() + " jobs" },
              ].map(({ label, val, note }) => (
                <div key={label} style={{ background:"#070D14", border:"1px solid #1C2B3A", borderRadius:8, padding:"10px 12px" }}>
                  <div style={{ fontSize:9, color:"#304050", textTransform:"uppercase", letterSpacing:"0.08em", marginBottom:4 }}>{label}</div>
                  <div style={{ fontSize:18, fontWeight:700, color:"#4FAAFF", fontVariantNumeric:"tabular-nums" }}>{fM(val)}</div>
                  <div style={{ fontSize:9, color:"#1A2A38", marginTop:2 }}>{note}</div>
                </div>
              ))}
            </div>

            {/* Comparison table */}
            <table style={{ width:"100%", borderCollapse:"collapse", fontSize:11 }}>
              <thead>
                <tr style={{ borderBottom:"1px solid #1C2B3A" }}>
                  {["Provider", "Per request", "Per job", "Per month", "vs Self"].map((h, i) => (
                    <th key={h} style={{ padding:"6px 8px", color:"#304050", fontWeight:600,
                                        fontSize:9, textTransform:"uppercase", letterSpacing:"0.05em",
                                        textAlign: i === 0 ? "left" : "right", whiteSpace:"nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr style={{ background:"rgba(58,143,191,0.05)", borderBottom:"1px solid #131F2A" }}>
                  <td style={{ padding:"7px 8px", color:"#4FAAFF", fontWeight:700, fontSize:11 }}>Self-hosted</td>
                  <td style={{ padding:"7px 8px", textAlign:"right", color:"#A8C8E0", fontSize:12 }}>{f$(selfPerReq)}</td>
                  <td style={{ padding:"7px 8px", textAlign:"right", color:"#A8C8E0", fontSize:12 }}>{fM(selfJobCost)}</td>
                  <td style={{ padding:"7px 8px", textAlign:"right", color:"#A8C8E0", fontSize:12 }}>{fM(selfMooCost)}</td>
                  <td style={{ padding:"7px 8px", textAlign:"right", color:"#304050" }}>—</td>
                </tr>
                {apiRows.map(({ id, api, perReq, jobCost }) => {
                  const ratio   = jobCost / selfJobCost;
                  const cheaper = ratio < 1;
                  return (
                    <tr key={id} style={{ borderBottom:"1px solid #0D1820" }}>
                      <td style={{ padding:"7px 8px", color:"#7A9AB0", fontSize:11 }}>
                        {api.oss && <span style={{ color:"#B06EFF", marginRight:3 }}>{"⬡"}</span>}
                        {api.name}
                      </td>
                      <td style={{ padding:"7px 8px", textAlign:"right", color:"#A8C8E0", fontSize:12 }}>{f$(perReq)}</td>
                      <td style={{ padding:"7px 8px", textAlign:"right", color:"#A8C8E0", fontSize:12 }}>{fM(jobCost)}</td>
                      <td style={{ padding:"7px 8px", textAlign:"right", color:"#A8C8E0", fontSize:12 }}>{fM(jobCost * batchesPerDay * 30)}</td>
                      <td style={{ padding:"7px 8px", textAlign:"right", fontWeight:700, fontSize:11,
                                   color: cheaper ? "#6EE7B7" : "#F87171" }}>
                        {cheaper
                          ? (((1 - ratio) * 100).toFixed(0) + "% cheaper")
                          : ("+" + ((ratio - 1) * 100).toFixed(0) + "%")}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize:9, color:"#131F2A", textAlign:"center", marginTop:4 }}>
            {"Lambda/RunPod/CoreWeave rates · Feb–Mar 2026 · Qwen CNY @7.28 · Fireworks flat rate"}
          </div>
        </div>
      </div>
    </div>
  );
}
