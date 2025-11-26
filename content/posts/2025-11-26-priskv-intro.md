---
date: '2025-11-26T12:00:00-00:00'
draft: false
title: 'PrisKV: An Colocated Tiered KVCache Store for LLM Serving'
author: ["Xu Wang", "Jinlong Xuan", "Yi Wang", "Haiyang Shi", "Bo Liu", "Jiaxin Shan"]

disableShare: true
hideSummary: true
searchHidden: false
ShowReadingTime: false
ShowWordCount: false
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowRssButtonInSectionTermList: false
UseHugoToc: true
ShowToc: true
tocopen: true
---

In recent years, large language models (LLMs) such as GPT, DeepSeek, Doubao and Qwen have advanced rapidly and are reshaping a wide range of industries. As the Scaling Law continues to be validated and pushed to its limits, LLM capabilities keep improving and are now playing a critical role in scenarios like enterprise knowledge management, intelligent customer support, code generation, and content creation.

However, this surge in demand is putting unprecedented pressure on the underlying AI infrastructure. The key question for the industry has shifted to: how can we deliver high-throughput, low-latency inference in a stable and cost-efficient way? Most modern models are built on Transformer architecture, and performance optimization has evolved from simply "adding more compute" to a more holistic, system-level effort focused on inference frameworks, scheduling strategies, and high-performance KVCache designs.

## KV Cache Introduction & Challenges

### What is KV Cache and Why Is It So Costly?

KV Cache (Key-Value Cache) is a technique used during large language model (LLM) inference to reuse previously computed intermediate results, significantly reducing computation and speeding up generation. In Transformer-based autoregressive models, each layer's attention mechanism computes Key and Value tensors for all past tokens. Instead of recomputing these Key/Value tensors from scratch every time a new token is generated, the model with KV Cache only computes them for newly arrived tokens, appends them to the cache, and reuses the existing history for subsequent attention calculations. Put simply, KV Cache works like a "notebook" for the model: when generating the next word, it doesn't need to rethink everything it has said before—it just looks up that notebook and continues from there.

However, this efficiency comes at a high memory cost. By default, KV Cache is stored in GPU memory, and as the sequence grows longer, the cache size grows linearly and can quickly consume most of the available VRAM.

A rough formula for KV cache size is:

```
cache_size ≈ 2 (Key and Value) × batch_size × seq_len × hidden_dim × num_layers × dtype_size
```

For large models such as 70B, when generating long sequences or serving high traffic, KV Cache alone can easily take up tens of gigabytes of GPU memory. This becomes one of the main bottlenecks limiting context length and concurrency, and is the reason why so many KV cache offloading / compression / pruning techniques are being proposed.

### KVCache Use Cases

KVCache appears in a variety of serving patterns rather than a single, monolithic design. In practice, we observe three representative use cases: prefix reuse, KVCache-centric prefill/decode disaggregation, and more advanced scenarios such as request live migration and sequence parallelism.

1. **Prefix Cache**

In many multi-round use cases (e.g., chatbots and agents) and in workloads that rely on prompt templates, large parts of the prompt share the same prefix across requests. For example, the system prompt and the initial turns of the conversation. The figure above shows a multi-turn chat where the blue segments represent these shared prefixes that can be reused between turns.

<p align="center">
  <img src="/images/priskv/prompt-cache-multi-turn.png" width="80%" style="display:inline-block; margin-right:1%" />
  <br />
  <a href="https://proceedings.mlsys.org/paper_files/paper/2024/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf" target="_blank" rel="noopener noreferrer">MLSys'24 - Prompt Cache</a>
</p>


Today, each inference request typically carries a long system prompt plus repeated conversational context so the model can better understand and respond to the user. During the prefill phase, the model repeatedly recomputes KV states for these identical prefixes, which wastes compute and reduces overall throughput. By reusing KV cache for these shared prefixes, we avoid redundant prefill computation, which lowers inference cost and boosts effective system throughput while still meeting the target SLOs.

2. **KVCache Centric P/D Disaggregation**

A KVCache-centric prefill/decode architecture treats KVCache as a first-class schedulable resource rather than a by-product of a fixed prefill–decode pair. Instead of directly wiring each prefill worker to a dedicated decode worker (as in Nixl-style P↔D designs), this approach introduces a logically shared KV layer and lets a global scheduler independently choose prefill and decode instances while routing KV blocks through the KV pool. Although this adds one extra hop on the critical path, it enables cluster-wide reuse of KVCache across requests, sessions, and even different applications (e.g., multi-round chat and prompt-template workloads), substantially reducing redundant prefill computation and improving effective throughput under TTFT/TBT SLO constraints. This KVCache-centric disaggregation has already been adopted in large-scale MaaS systems such as Kimi's Mooncake architecture, illustrating a practical production use case for distributed KV caching.

<p align="center">
  <img src="/images/priskv/mooncake-arch.png" width="80%" style="display:inline-block; margin-right:1%" />
  <br />
  <a href="https://www.usenix.org/system/files/fast25-qin.pdf" target="_blank" rel="noopener noreferrer">Fast'25-Mooncake</a>
</p>


Beyond the mainstream P/D and prefix-reuse scenarios, KVCache is also a powerful enabler for **request live migration** and **sequence parallelism**. Feel free to check out these papers [OSDI'24-ServerlessLLM](https://www.usenix.org/system/files/osdi24-fu.pdf) and [Infinite-LLM](https://arxiv.org/pdf/2401.02669) for more details.

## PrisKV Architecture

PrisKV follows a client-server architecture designed around three core principles: performance, flexibility, and operational simplicity. PrisKV integrates with popular inference frameworks like vLLM and SGLang through the AIBrix KVCache Offloading Framework, providing a simple Python API while the underlying transport leverages RDMA for maximum performance.  
<p align="center">
  <img src="/images/priskv/priskv-arch.png" width="100%" style="display:inline-block; margin-right:1%" />
</p>

***Performance at the Core***
PrisKV's architecture is built around a single premise: RDMA should not be an afterthought. The entire data path—from client request to server response—is designed around RDMA primitives. Both client and server implement dedicated RDMA connection management, completion queue processing, and scatter-gather list handling for efficient multi-buffer transfers.  
For AI workloads, this matters because PrisKV supports GPU Direct RDMA (GDR), allowing KV cache data to flow directly between GPU memory and the network fabric without staging through CPU memory. This zero-copy path is what makes sub-millisecond offload/reload operations possible, keeping GPU utilization high during inference.

***Tiered Storage for Cost Efficiency***
Not all cached data deserves expensive DRAM. PrisKV implements a pluggable backend system that enables automatic data movement between storage tiers. The in-memory KV engine—built on hash tables with slab and buddy allocators—serves hot data with minimal latency. When memory pressure rises, LRU eviction moves cold entries to lower-cost backends.  
Currently, the system supports local filesystem storage and Redis-compatible services. This last point is particularly useful for cloud deployments: teams can point the cold tier at managed services like AWS ElastiCache or GCP Memorystore, leveraging existing infrastructure.

***Scalability Through Cluster Design***
PrisKV's cluster architecture is designed with large-scale inference in mind. The system uses consistent hashing for automatic key distribution across nodes, enabling horizontal scaling as workloads grow. Combined with AIBrix's orchestration capabilities, PrisKV can support large-scale deployments where hundreds of inference instances share a distributed KV cache pool—turning what would be isolated per-node caches into a unified, cluster-wide resource.

***Operational Visibility***  
PrisKV exposes an HTTP interface for runtime management and monitoring. Operators can query memory statistics, connection counts, and KV engine metrics. Access control lists provide basic multi-tenancy support, and liveness endpoints enable health checking for load balancer integration.  
While the control plane handles monitoring and configuration well, features like automated deployment orchestration and rolling upgrades would likely require external tooling—the current implementation focuses on the data plane performance story.

## How does PrisKV work with existing frameworks?

From an architectural point of view, PrisKV integrates with AIBrix KVCache offloading framework and Inference engines in three layers:

1. **Engine-side KVCache Manager** (e.g., SGLang HiCache, vLLM's KV management)  
2. **AIBrix KVCache Offloading Framework**, which bridges engines and L2 external KV stores such as PrisKV  
3. **PrisKV + AIBrix orchestration**, which turns multiple KV nodes into a cluster-wide, shared in-memory pool

<p align="center">
  <img src="/images/priskv/aibrix-offloading-framework.png" width="60%" style="display:inline-block; margin-right:1%" />
</p>

### AIBrix KVCache Offloading: making PrisKV a shared L2 cache in vLLM and SGLang

Modern inference engines such as vLLM and SGLang already ship with built-in KVCache mechanisms that use spare HBM and DRAM to cache context and avoid redundant compute. However, single-node KVCache still faces three fundamental limitations:

* **Limited capacity** — long contexts and high concurrency quickly exhaust GPU and host memory.  
* **No sharing across instances** — each engine process typically owns its own private cache.  
* **Hard to support advanced patterns** such as KV migration, KVCache Centric prefill/decode (P/D) disaggregation.

To address this, AIBrix provides a production-grade **[KVCache Offloading Framework](https://aibrix.readthedocs.io/latest/designs/aibrix-kvcache-offloading-framework.html)**, which exposes **PrisKV** as a general-purpose remote KV cache layer:

* By default, it enables an **L1 host-side DRAM cache**, offloading KVCache from GPU HBM to CPU memory and significantly relieving GPU pressure with minimal added latency.

* For larger-scale reuse and multi-node sharing, users can optionally enable an **L2 remote cache**, offloading KVCache into a distributed KVCache pool built on PrisKV. This pool offers much higher capacity and better hit rates, improving throughput and stability under real workloads.

### PrisKV clusters: from individual nodes to a shared memory pool

In the above examples, PrisKV is deployed as a single node. On a larger scale, you can make it as **cluster-level shared KV memory pool**.AIBrix's orchestration layer takes care of turning multiple PrisKV servers into a coherent cluster:

* Cluster specs (capacity, number of nodes, tiers) are described declaratively via CRDs.  
* PrisKV servers are sharded using a consistent-hash–style scheme, and membership/routing metadata is kept in a small control component.  
* The offloading layer uses this metadata to route KV operations to the correct PrisKV node and to support scaling and rebalancing.

<p align="center">
  <img src="/images/priskv/priskv-cluster-orchestration.png" width="80%" style="display:inline-block; margin-right:1%" />
</p>

## How to use it?

This guide helps you launch a complete KVCache offloading setup with PrisKV and an inference engine (vLLM or SGLang) using Docker Compose.

**Prerequisites**

* A Linux host with Docker and Docker Compose installed
* NVIDIA GPU drivers and runtime set up (required for vLLM/SGLang GPU inference)
* Privileged mode enabled for Docker (required for PrisKV server and engine containers)

**PrisKV Server Image**

* `aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/priskv:v0.0.2`

**Inference Engine Image**

* vLLM + aibrix_kvcache + nixl + PrisKV: `aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/vllm-openai:v0.10.2-aibrix0.5.1-nixl0.7.1-priskv0.0.2-20251121`
* SGLang + aibrix_kvcache + nixl + PrisKV: `aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/sglang:v0.5.5.post3-aibrix0.5.1-nixl0.7.1-priskv0.0.2-20251121`

Note: If you'd like to build all images from source or tweak the compose file, please refer to the [detailed guide](https://github.com/aibrix/PrisKV/tree/main/samples/kvcache-offloading) in the documentation.

**Deployment with Docker Compose**

1. Save the following yaml as deploy.yaml

```
services:
  redis:
    image: redis:7.4.2
    container_name: redis
    ports:
      - "16379:6379"
    entrypoint: ["redis-server"]
    command: ["--appendonly", "yes"]
    
  init:
    image: redis:7.4.2
    container_name: init
    network_mode: "host"
    environment:
      PRISKV_CLUSTER_META: '{"version":1,"nodes":[{"name":"node0","addr":"33.2.58.130","port": 9000,"slots":[{"start":0,"end":4095}]}]}'
    depends_on:
      - redis
    restart: "no"
    command: |
      sh -c "
        sleep 5
        echo $$PRISKV_CLUSTER_META > /tmp/priskv.json
        cat /tmp/priskv.json | redis-cli -h 0.0.0.0 -p 16379 -x SET priskv_cluster_metadata
        redis-cli -h 0.0.0.0 -p 16379 CONFIG SET requirepass kvcache-redis
      "

  priskv:  # capacity: 512 GB
    image: aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/priskv:v0.0.2
    container_name: priskv
    network_mode: "host"
    shm_size: 32gb
    privileged: true
    cap_add:
      - IPC_LOCK
    command: ['-a', '33.2.58.130', '-p', '9000', '-v', '1048576', '-b', '524288', '-k', '2097152', '-K', '256', '-t', '16', '--acl', 'any', '-L', 'stdout', '-l', 'notice']

  engine:
    image: aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/vllm-openai:v0.10.2-aibrix0.5.1-nixl0.7.1-priskv0.0.2-20251121
    container_name: engine
    runtime: nvidia
    network_mode: "host"
    shm_size: 32gb
    cap_add:
      - IPC_LOCK
    privileged: true
    volumes:
      - /data01:/data01
    depends_on:
      - init
      - priskv
    environment:
      VLLM_KV_CONFIG: '{"kv_connector":"AIBrixOffloadingConnectorV1Type3","kv_role":"kv_both"}'
      CUDA_VISIBLE_DEVICES: 0,1,2,3
      AIBRIX_KV_CACHE_OL_BLOCK_SIZE: 64
      AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED: 0
      AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND: PRISKV
      AIBRIX_KV_CACHE_OL_PRISKV_REMOTE_ADDR: 127.0.0.1
      AIBRIX_KV_CACHE_OL_PRISKV_REMOTE_PORT: 16379
      AIBRIX_KV_CACHE_OL_PRISKV_PASSWORD: kvcache-redis
    entrypoint: []
    command: |
      sh -c "
        sleep 30
        
        python3 -m vllm.entrypoints.openai.api_server --port=8000 --uvicorn-log-level=warning --model=/data01/models/Qwen3-32B --served-model-name=Qwen3-32B --trust-remote-code --disable-log-requests --disable-fastapi-docs --swap-space=0 --no-enable-prefix-caching --kv-transfer-config=$$VLLM_KV_CONFIG --tensor-parallel-size=4
      "
```

2. Modify `PRISKV_CLUSTER_META` to describe the consistent hashing topology for your PrisKV cluster. For a single-server setup:  
```
{"version": 1,"nodes": [{"name": "node0","addr": "<REPLACE_WITH_SERVER_IP>","port": 9000,"slots": [{ "start": 0, "end": 4095 }]}]}
```
Tips:
  - Replace `<REPLACE_WITH_SERVER_IP>` with the reachable IP of your PrisKV server and update `-a` in the PrisKV server command.
  - If port 9000 is not available, change it to a free port in PRISKV_CLUSTER_META and update `-p` in the PrisKV server command.

3. Ensure your model files exist on the host at `/data01/models/<MODEL>`. If not, download or place your model there. Alternatively, update the `volumes:` and `--model` path in the compose file to match your host path.

4. Start services: `docker compose -f deploy.yaml up -d`.

5. Verify services:  
   - Verify Redis initialization: `docker compose -f deploy.yaml logs -f init`.
   - Check PrisKV logs: `docker compose -f deploy.yaml logs -f priskv` (ensure it reports listening on your chosen address/port).
   - Check engine logs: `docker compose -f deploy.yaml logs -f engine`
6. Issue request: once the engine is up, use the following curl command to issue a request.

```
curl localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-32B", "messages": [{"role": "user", "content": "Tell me a joke"}]}'
```

8. Stop services: `docker compose -f deploy.yaml stop`.
9. Cleanup: `docker compose -f deploy.yaml down -v` to remove containers and Redis data.

**Common issues**

* Engine startup errors about model path: ensure `/data01/models/<MODEL>` exists and is mounted; otherwise adjust `volumes:` and `--model` path.  
* GPU visibility: ensure NVIDIA drivers and runtime are installed; test with `nvidia-smi` inside the engine container.  
* PrisKV bind error: update `-a` to your host IP or `0.0.0.0`; ensure port `9000` is not in use.  
* Redis authentication errors: confirm the password matches `AIBRIX_KV_CACHE_OL_PRISKV_PASSWORD` everywhere.

For cluster setup, feel free to refer to [documentation](https://github.com/aibrix/PrisKV/tree/main/samples/cluster) for more references.

## How PrisKV performs?

Before diving into end-to-end engine benchmarks, we micro-benchmark PrisKV with **value sizes 512KB, 1MB, 2MB, 4MB, and 8MB** on H20 with 400Gbps RDMA Network, which roughly match the KV footprint of 16-64 tokens for 8B/30B/70B-class models (such as Llama-8B, Qwen-32B, and Llama-70B). Under this setting, a single PrisKV node sustains **tens of thousands of QPS with sub-millisecond average latency** over RDMA, indicating that the KV store itself has ample headroom and is unlikely to be the bottleneck for the L2 KVCache path.  

<p align="center">
  <img src="/images/priskv/priskv-benchmark.png" width="100%" style="display:inline-block; margin-right:1%" />
</p>

### End-to-End Benchmarking

Across end-to-end vLLM benchmarking on Nvidia H20 GPUs with Qwen3-32B (TP=4) on 8k-token prompts and 200-token outputs at 16 and 32 concurrent requests, PrisKV-powered KVCache offloading consistently delivers substantial throughput and latency improvements over the baseline: at 16 concurrency, request and token throughputs increase by about 4.8x while mean TTFT drops by \~90%; TPOT also falls by \~75%. At 32 concurrency, gains are even larger: throughput improves by roughly 6.35x mean TTFT decreases by \~90.7% (4842ms→450ms)with TPOT reductions of 83–84%, as shown in following figures.  

<p align="center">
  <img src="/images/priskv/priskv-e2e-benchmark-throughput.png" width="30%" style="display:inline-block; margin-right:1%" />
  <img src="/images/priskv/priskv-e2e-benchmark-ttft.png" width="30%" style="display:inline-block; margin-right:1%" />
  <img src="/images/priskv/priskv-e2e-benchmark-tpot.png" width="30%" style="display:inline-block; margin-right:1%" />
</p>

These results highlight PrisKV's ability to significantly accelerate time-to-first-token, improve tail latencies, and scale inference under higher concurrency, translating directly into faster, more predictable user experiences and higher system throughput.

## Future Plan

Looking ahead, we plan to evolve PrisKV along three main directions.

* First, we aim to position PrisKV as a cluster-wide KV cache substrate and extend its tiering interface to integrate with more managed services, such as AWS ElastiCache and Google Memorystore, so that colder KV data can seamlessly spill into external tiers.
* Second, we plan to pursue true zero-copy integration for intra-node communications between inference engines and PrisKV, with a particular focus on optimizing low-end GPU deployments to lower the adoption barrier for everyday LLM serving workloads.
* Third, we will further strengthen PrisKV's control capabilities, supporting flexible replication and migration, workload-aware eviction policies, and robust fault tolerance and recovery, to make it a first-class building block for large-scale, production LLM systems.

We warmly invite practitioners and researchers who are interested in KVCache, LLM serving, and tiered storage to join us and help shape the next generation of PrisKV.
