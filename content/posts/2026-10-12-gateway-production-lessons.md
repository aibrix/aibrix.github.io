---
date: '2026-10-12T13:00:00-00:00'
draft: false
title: 'Scaling LLM Inference in Production: Lessons Learned with AIBrix Gateway'
author: ["The AIBrix Team"]

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

## Introduction

Over the past year, we have been running AIBrix in production and learning what it takes to operate LLM inference reliably at scale. Today, AIBrix supports large-scale production workloads across diverse GPU fleets and multiple regions.

AIBrix serves a broad range of open-source and custom models, including Qwen, DeepSeek, Gemma, and other model families, with both vLLM and SGLang as inference engines. These deployments handle high-volume traffic across workloads ranging from chat completions and embeddings to multimodal inference and generation, including image, audio, and video processing.

At this scale, challenges emerge that are difficult to reproduce in benchmarks or smaller deployments. Models can span hundreds—or even thousands—of replicas, traffic changes continuously, capacity scales up and down, and workloads increasingly run across heterogeneous GPU fleets. Under these conditions, seemingly small problems in routing, load balancing, health management, or autoscaling can quickly become system-wide issues.

**This makes the Gateway a critical part of the production inference path.** It is no longer simply forwarding requests to available replicas; it must make fast, inference-aware routing decisions while protecting backend capacity and maintaining availability. For high availability, production deployments typically run multiple AIBrix Gateway instances—often three to five depending on scale—with Redis providing shared state where routing decisions require a global view.

Running the Gateway this way also makes it a distributed system. Multiple Gateway instances must coordinate enough state to make consistent routing and admission decisions without turning that coordination into latency on the critical path. At the same time, the routing layer must handle overloaded or degraded replicas, changing capacity during autoscaling, and large fleets where not every GPU—or every request—has the same performance characteristics.

Many of the improvements we have made to AIBrix came directly from encountering these problems under real production traffic. **In this post, we share what went wrong, what we changed, what we learned, and the challenges we are still working through.**

Our goal is not to present a perfect architecture, but to share practical lessons from operating inference infrastructure at scale. We hope these experiences are useful to others building similar systems—and that they encourage the community to share what they are learning as well.

![AIBrix Gateway production architecture: clients send inference requests to multiple AIBrix Gateway replicas that share state through Redis and route to model deployments across three availability zones](/images/gateway-production-lessons/gateway-architecture.png)

## Every Millisecond in the Routing Path Matters

For latency-sensitive, user-facing inference workloads, every millisecond matters. The routing layer sits directly on the request path, so even a small amount of additional processing can become a meaningful part of end-to-end latency.

One example is a production Qwen3-VL embedding workload running on A100 GPUs that powers user-facing multimodal applications. With approximately peak **120 ms P99 end-to-end latency**, adding even a few milliseconds in the Gateway can directly affect the responsiveness users experience.

Through several rounds of optimization, we reduced AIBrix Gateway routing overhead from **~10 ms to ~2 ms P99** for this workload. For the embedding workload shown above, intelligent load balancing improved backend selection and reduced P99 end-to-end latency by **~10%** compared with simpler routing—saving substantially more latency than the routing layer itself introduces.

![Production latency: AIBrix Gateway adds ~2 ms P99 routing overhead while intelligent routing lowers P99 end-to-end latency by ~10%, from ~120 ms to ~110 ms, compared with baseline routing](/images/gateway-production-lessons/production-latency.png)

Achieving this required treating the entire routing path as a hot path and continuously removing unnecessary work:

- **Minimize parsing and allocations** — parse requests once, avoid unnecessary JSON unmarshalling, allocations, copies, and repeated tokenization.
- **Keep distributed state off the critical path** — reuse Redis state within a request, batch and pipeline operations, cache where possible, and move bookkeeping asynchronously.
- **Build and continuously optimize fast paths** — skip unnecessary scoring for simple cases, cache configuration and Kubernetes state, reduce critical sections, optimize prefix matching and hashing, and use microbenchmarks to catch hot-path regressions.

The broader lesson is simple: **intelligent routing only pays off if the intelligence itself is cheap**. As routing becomes more sophisticated, we evaluate both the quality of the routing decision and the cost of making it.

## Pod Health Checks Aren't Enough

A pod being Ready in Kubernetes does not necessarily mean it should receive inference traffic. In production, replicas can become degraded because of network issues, engine failures, overload, incomplete streaming responses, or temporary control-plane connectivity issues. Continuing to route traffic to these replicas can result in failed requests and increased **4xx and 5xx error rates**.

AIBrix Gateway therefore evaluates replica health using several complementary mechanisms and can temporarily exclude unhealthy or overloaded replicas from routing:

- **Request-driven failure detection** — track upstream errors, failed streams, and incomplete responses. Repeated failures trigger additional health checks before a replica is excluded.
- **Background health probing** — periodically probe replicas to detect failures even when they are not receiving traffic.
- **Load-outlier detection** — temporarily exclude replicas whose running and waiting requests indicate abnormal overload.
- **Progressive recovery** — use backoff and half-open probing to gradually return excluded replicas to the routing pool.

![Pod blacklist in AIBrix Gateway: request failures, background health probing, load-outlier detection, and cluster or network issues feed a per-replica circuit breaker that excludes unhealthy pods from routing and fails open if every replica is excluded](/images/gateway-production-lessons/health-checks-blacklist.png)

Blacklisting is intentionally a routing protection mechanism, not another source of availability failure. If every ready replica is excluded, the Gateway can fall back to the available replicas rather than turning a partial degradation into a complete outage.

The broader lesson is that **readiness, health, and ability to safely serve traffic are different signals**. Kubernetes tells us whether a replica is available; the routing layer must continuously determine whether it is a good destination for the next request.

## Your Gateway Is a Distributed System

Running multiple Gateway replicas is essential for high availability, but it introduces another challenge: **routing decisions depend on state that no single Gateway owns completely**.

Each Gateway receives an independent slice of traffic. Local state alone cannot answer questions such as: *How busy is this replica globally? Where is this session pinned? Which replica owns this async job?*

AIBrix uses Redis to share routing-critical state across Gateway replicas:

- **Running requests and Gateway liveness** — maintain a global view of replica load while preventing stale counts from failed Gateways from influencing routing.
- **Session affinity** — keeps agentic and multi-turn sessions pinned to the same backend regardless of which Gateway receives the next request.
- **Async job routing** — maps long-running jobs back to the replica holding their state.

![Redis in AIBrix: how multiple Gateways share running-request counts, Gateway liveness, session affinity, and async job state through Redis, the key challenges of the running-request counter, and the performance optimizations that keep Redis overhead low](/images/gateway-production-lessons/redis-shared-state.png)

Because this state can participate directly in routing decisions, Redis performance becomes part of Gateway performance. We minimize that cost through batched and pipelined operations, reusing shared state within a request, moving writes off the critical path, and falling back to local state where appropriate.

At the same time, not every routing signal needs to be globally synchronized. Signals that participate in hard admission decisions, such as running-request counts used for per-replica inflight limits, require a shared view across Gateways. Other signals can tolerate weaker consistency. For example, the capacity estimates used by heterogeneous load balancing are learned locally by each Gateway, so temporary differences between Gateway replicas are expected as their observations converge.

This creates an important tradeoff: synchronizing more state can make routing decisions more consistent, but it also adds communication and coordination to the request path. The principle is therefore to keep distributed state **small, short-lived, and limited to decisions that truly require a global view**.

Once the Gateway is replicated, **routing state becomes distributed state—and deciding what must be synchronized is itself part of routing-system design.**

## Protect the Model, Not Just the Gateway

A Gateway can be healthy while the inference replicas behind it are overloaded. In production, we found that the routing layer needs to protect **individual model replicas**, not just Gateway capacity.

This becomes particularly important for large deployments. For one production workload, the model typically runs across hundreds of replicas, but during peak traffic the fleet can grow beyond **1,000 replicas while serving more than 20,000 concurrent requests**. At this scale, even a small traffic imbalance or burst can overload individual replicas. Model protection therefore needs to scale dynamically with the serving fleet rather than rely on a static model-level limit.

AIBrix provides two complementary controls:

- **Per-replica RPS** — users specify the throughput each replica can safely sustain. AIBrix converts this into a model-wide RPS limit based on the number of routable replicas, automatically adjusting as the fleet scales.
- **Per-replica inflight limits** — replicas at their concurrency limit are excluded from routing. After selecting a replica, AIBrix performs an atomic check-and-admit using shared cross-Gateway state, preventing multiple Gateways from collectively exceeding the replica's limit.

Enforcing these limits becomes more challenging when multiple Gateway instances are running for high availability. Per-replica request state must be stored and retrieved from Redis so that every Gateway makes admission decisions against a shared view of backend load. **Maintaining consistency and correctness across concurrent routing decisions, while keeping Redis coordination off the critical path as much as possible, becomes part of the problem itself.**

![Protect the Model, Not Just the Gateway: per-replica RPS scales into a model-wide limit, per-replica inflight limits are enforced with an atomic cross-Gateway check-and-admit, and a request that exceeds either limit is rejected with 429 without consuming capacity](/images/gateway-production-lessons/per-replica-capacity-limits.png)

If all replicas are saturated, AIBrix returns **429** rather than continuing to overload the inference engines. One alternative is to queue requests at the Gateway until capacity becomes available, but in practice this introduces its own tradeoffs. As we discuss in the [Open Challenges](#open-challenges) section, queuing can hide demand from the autoscaler, grow rapidly under sustained load, and consume a request's remaining deadline before inference even begins.

The key lesson is that **rate and concurrency protect different dimensions of inference load**: RPS controls how quickly work enters the system, while inflight limits control how much work can accumulate on each replica.

## Scaling Up Requires Traffic-Aware Routing

Autoscaling inference workloads involves more than adding and removing Pods. A replica has a **traffic lifecycle** that needs to be coordinated with its Kubernetes lifecycle. A newly Ready replica may be technically capable of serving requests, but immediately treating it like an established replica can create unexpected traffic patterns—especially when routing decisions are distributed across multiple Gateway instances.

### Avoiding the New-Replica Stampede

We observed one such failure mode during scale-up. A newly Ready replica starts with zero running requests, making it appear to be the ideal target for least-request routing.

The problem becomes more pronounced in a distributed Gateway deployment. Multiple Gateway instances make routing decisions independently while using Redis to maintain a shared view of per-replica request state. Because request-count updates propagate asynchronously, several Gateways can observe the same replica as the least loaded and independently select it before the shared state catches up.

In production, this caused newly added replicas to receive a disproportionate share of traffic immediately after becoming Ready, producing large queue-depth spikes while existing replicas remained relatively underutilized.

To address this, we introduced **Least-Request Top-K with slow-start ramp-up**:

- **Top-K candidate selection** — instead of always selecting the single least-loaded replica, the router considers up to *K* of the least-loaded replicas within a small load difference of the current minimum. This spreads concurrent routing decisions across similarly loaded replicas without sending traffic to replicas that are already significantly busier.
- **Randomized selection** — the final target is selected uniformly from this candidate set, reducing the chance that multiple Gateway instances observing the same shared state converge on the same replica.
- **Slow-start ramp-up** — newly Ready replicas receive a temporary synthetic load based on the load of established replicas. This prevents a new replica with zero requests from immediately appearing dramatically underloaded. The synthetic load gradually decays over a configurable ramp window, allowing the replica to take an increasing share of traffic as it warms up.

![New replicas no longer receive a disproportionate amount of traffic: with plain least-request routing a newly joined pod receives a queue-depth spike, while Least-Request Top-K with ramp-up distributes traffic across replicas and gradually shifts load to the new pod](/images/gateway-production-lessons/least-request-top-k.png)

Together, these mechanisms address two different aspects of the problem: Top-K prevents distributed Gateways from repeatedly converging on the same minimum, while slow-start prevents a newly added replica from appearing artificially attractive simply because it has no traffic yet.

The broader lesson is that **adding capacity and sending traffic to that capacity are separate operations**. A Pod becoming Ready should make it *eligible* for traffic—not immediately give it a full share of production load.

## Not All GPUs—or Requests—Are Equal

Production GPU fleets are rarely homogeneous. When capacity on one GPU type is exhausted, workloads may need to spill over to hardware with very different performance characteristics.

This breaks a basic assumption of traditional load balancing: **equal request counts do not mean equal load**. Sending the same QPS to a faster and slower GPU can overload one deployment while leaving capacity unused on the other.

AIBrix addresses this by scoring each replica based on its **current load relative to its observed capacity**:

```
score = (running + λ · queued + 1) / capacity  ×  (1 + α · (1 − kvFree)²)
```

**Lower score = better routing candidate**

The first term normalizes current work by how quickly the replica can process it, while the second increasingly penalizes replicas as their available KV-cache capacity decreases.

Rather than relying on a static table of GPU performance, AIBrix **learns capacity from production traffic**:

- **Generative workloads** — observed output-token throughput.
- **Embedding workloads** — inverse request latency, since there are no output tokens.
- **New replicas** — start with the fleet's observed average until enough measurements are available.

Importantly, this learned value represents observed serving capacity under the current workload mix, not a fixed property of the GPU. Request shape can materially affect throughput even on identical hardware, particularly for input-heavy and multimodal workloads. As the workload mix changes, the observed capacity estimate can change with it.

Capacity is also only one part of the routing decision. Signals such as KV-cache availability depend on runtime telemetry from the inference engine, which is refreshed approximately every 50 ms. Even at this frequency, telemetry introduces another practical challenge: **missing, stale, and genuinely low-utilization signals are not equivalent**. A missing or stale signal should not automatically make a replica appear lightly loaded simply because no recent measurement is available. As routing incorporates more runtime signals, defining freshness and fallback behavior becomes increasingly important.

![Capacity-aware routing aligns GPU utilization across different GPU types: before, similar QPS sent to a slower and a faster GPU left the faster GPU underutilized; after, more traffic goes to the faster GPU and both deployments reach similar utilization](/images/gateway-production-lessons/capacity-aware-routing.png)

In production, the difference was clear. Before capacity-aware routing, similar QPS was sent to both GPU types, creating a large utilization gap. After rollout, AIBrix naturally sent **more traffic to the faster GPU**, bringing utilization across the two deployments much closer together.

The broader lesson is that **balanced load does not mean equal request counts**. In heterogeneous fleets, routing should distribute work according to the capacity available to process it—not simply divide traffic evenly.

## If You Can't Explain a Routing Decision, You Can't Debug It

As routing becomes more intelligent, observability of the **routing decision itself** becomes critical. Knowing that a request was slow or went to a particular replica is not enough—we need to understand **why that replica was selected**.

A routing decision can depend on many signals: running requests, prefix-cache locality, replica health, RPS and inflight limits, session affinity, hardware characteristics, and routing strategy. We therefore use two complementary layers of observability:

- **Metrics provide a fleet-level view** — selection and traffic distribution, routing latency, load imbalance, exclusions, errors, and routing signals.
- **Distributed traces provide a request-level view** — allowing operators to follow an individual request through routing and inference.

![Routing decision pipeline from incoming request through candidate pods, eligibility filters, routing signals, and top-K ranking to the selected pod, with the metrics emitted at each stage](/images/gateway-production-lessons/routing-decision-observability.png)

For example, if Pod A has fewer running requests but Pod B receives the request, that does not necessarily indicate a routing failure. Pod B might have stronger prefix-cache locality or session affinity, while Pod A may have been excluded by an RPS or inflight limit.

Aggregate metrics tell us *that* routing behavior looks unusual. To understand *what happened to a particular request*, AIBrix also provides OpenTelemetry-based distributed tracing across the request path.

Trace context is propagated through Envoy → AIBrix Gateway → inference backend. Gateway spans capture request processing and target-pod selection, while `process.select_target_pod` records attributes such as the routing strategy, candidate and ready replicas, selected pod, and routing time. `llm.inference` and its child spans capture the Gateway-observed inference lifecycle, including time to the first response chunk and streaming through the last response chunk.

![AIBrix distributed trace for one inference request, propagated across Envoy, the AIBrix Gateway plugin, and the inference backend, with a real Jaeger trace showing the span hierarchy and timing](/images/gateway-production-lessons/distributed-trace.png)

Together, these provide a practical debugging workflow: **metrics tell us where to look; traces tell us what happened.**

The lesson is simple: **a smarter routing algorithm also needs to be a more explainable routing algorithm.** When something goes wrong in production, operators need to reconstruct not only where a request went, but why it went there.

## Open Challenges

Production scale continues to expose new problems. Three areas we are actively exploring are **traffic-aware scale-down**, **scalable routing across increasingly large replica fleets**, and **intelligent queuing and admission control that accounts for autoscaling, capacity, and request deadlines**.

### Traffic-Aware Scale-Down and Proactive Draining

Scale-down is more complicated than simply terminating a Pod. Inference requests can run for seconds or minutes, so a replica selected for termination may still have significant work in flight.

Today, the lifecycle is largely reactive:

```
Autoscaler scales down → Pod starts terminating → Gateway observes → Traffic stops
```

We want to move toward a more coordinated, traffic-aware lifecycle where **routing naturally creates a safe termination candidate before the autoscaler removes capacity**.

When the system determines that capacity can be reduced, the Gateway can progressively steer new traffic away from a replica that is already lightly loaded. Existing requests continue to completion while new requests are preferentially routed to other replicas. As a result, that replica's inflight load naturally converges toward zero.

The autoscaler can then observe the same load signal and select the **least-loaded or idle replica** for termination:

```
Low-load replica → Traffic steered elsewhere → Inflight approaches 0 → Autoscaler selects replica → Terminate
```

This avoids requiring an explicit handoff between the Gateway and autoscaler. Instead, **routing and autoscaling converge on the same termination candidate through shared load signals**.

The goal is simple: **scale-down should drain traffic before removing capacity, rather than reacting to termination after it has already started.**

**RFC:** [#3187 — Traffic-Aware Scale-Down and Proactive Draining](https://github.com/vllm-project/aibrix/issues/3187)

### Don't Rank the Entire Fleet for Every Request

As deployments grow, **sophisticated routing can itself become expensive**. Evaluating every candidate is inexpensive with tens of replicas; with hundreds of replicas, fetching distributed state and scoring the entire fleet for every request can become a meaningful part of routing latency.

We are already seeing this in production: once replica counts cross **500 replicas**, the cost of routing becomes noticeable. With per-replica inflight limits enabled, routing requires additional Redis reads and writes to maintain shared request counts across Gateway instances. At this scale, we have observed **~5 ms P90 Gateway latency**—significantly higher than we would like as we work to keep routing overhead around **~2 ms P99**.

The problem is that routing cost can grow with the size of the replica fleet:

```
Request → Fetch state for N replicas → Evaluate N → Rank N → Select
```

At sufficiently large N, doing this for every request becomes unnecessarily expensive. We are therefore exploring candidate-based routing:

```
Request → Select K candidates → Fetch/Evaluate K → Select
```

where **K ≪ N**.

The first stage can use inexpensive signals such as health, locality, hardware class, cached state, or randomized sampling to quickly reduce the candidate set. More sophisticated routing—including distributed load signals—can then operate only on those candidates.

The goal is not simply to make the routing algorithm faster, but to make its **cost largely independent of total fleet size**. A deployment growing from 50 to 500 replicas should not require proportionally more work on the critical path of every request.

The broader lesson is that **the routing algorithm itself must scale with the inference fleet**. At large enough scale, the cost of deciding where a request should go becomes part of inference performance.

**RFC:** [#3188 — Don't Rank the Entire Fleet for Every Request](https://github.com/vllm-project/aibrix/issues/3188)

### Queuing, Autoscaling, and Request Deadlines

Queuing can absorb short traffic bursts, but in production we found that it can also **hide demand from the autoscaler and amplify overload when traffic continues to increase**.

We experimented with Gateway-side queuing for a workload serving approximately **1,500+ QPS**. During periods of high load, queuing reduced the amount of traffic reaching the inference replicas. As a result, the autoscaler did not always observe enough load to scale out as aggressively as needed. If incoming QPS continued to increase, the queue could then grow rapidly—**crossing 100k+ queued requests** in a short time—with requests waiting **hundreds of milliseconds or even seconds** before reaching a replica.

Long queues introduce another problem: **the remaining client deadline matters more than simply finding capacity**. For example, with a 30-second client timeout, a request that spends 10 seconds waiting in the queue has only 20 seconds left to complete inference. If the request is unlikely to finish within that remaining budget, dequeuing and executing it may waste GPU capacity—the client can cancel before receiving the result.

In those situations, **returning 429 early and allowing the client to retry** may be more efficient than accepting work that is unlikely to complete.

The open challenge is building **deadline- and capacity-aware admission control**. Instead of deciding only whether to queue or reject a request, the system could consider queue depth, expected wait time, estimated execution time, remaining client deadline, and expected near-term capacity:

```
Request → Can it start and finish within its deadline? → Admit / Queue / 429
```

The broader question is: **when should the system absorb a temporary burst, when should queuing trigger additional capacity, and when should it fail fast rather than spend GPU resources on work that is unlikely to complete?**

**RFC:** [#3189 — Queuing, Autoscaling, and Request Deadlines](https://github.com/vllm-project/aibrix/issues/3189)

## Closing Thoughts

Running AIBrix at production scale has reinforced a simple lesson: **many of the hardest inference problems only become visible under real traffic**—when fleets are large, workloads are constantly changing, and small routing or reliability issues can quickly become system-wide problems.

At this scale, the Gateway is no longer just a request proxy. It becomes a **distributed control point for admission, load balancing, health, traffic lifecycle, and inference-aware routing**. Its latency, scalability, and reliability become part of inference performance itself.

The lessons in this post came from operating these systems in production: keep the routing path fast, protect individual replicas from overload, distinguish readiness from serving health, account for heterogeneous capacity, coordinate distributed routing state carefully, and make intelligent routing decisions observable and explainable.

There is still plenty to solve—from traffic-aware scale-down and scalable routing across increasingly large fleets to deadline- and capacity-aware admission control. We hope sharing both **what worked and what we are still figuring out** helps others building large-scale inference infrastructure, and we look forward to learning from the production experiences of the broader community.

From the AIBrix team, thank you to our **community members and contributors**. Your contributions, feedback, bug reports, and real-world experiences continue to make AIBrix a stronger platform for production inference.
