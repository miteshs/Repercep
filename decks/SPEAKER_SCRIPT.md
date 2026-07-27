# Repercep — Seed Deck Speaker Script

**CONFIDENTIAL.** Companion to `repercep-seed-deck-2026-07.html` (20 slides).

- **Target: 14 minutes**, leaving 30+ for questions. In a seed meeting the Q&A is
  the meeting; the deck is the setup for it. If you are at slide 10 at minute 12,
  skip 12–15 and go straight to 16.
- **This is a spine, not a script to read.** Bold lines are the ones worth saying close
  to verbatim, because they are load-bearing or easy to get wrong. Everything else is
  yours to say naturally.
- **The one habit that decides this pitch:** when a number is soft, say so *before*
  they ask. Every claim you volunteer a limit on makes the next claim cost less to
  believe. That is the whole strategy of this deck.

---

## The 30-second version (if you get one elevator, use this)

> Every inference cloud — Fireworks, Together, Baseten — is an LLM company that grew a
> container service for everything else. Bring them a vision model, a diffusion model, a
> robot policy, and the answer is "here's a GPU-hour and a Dockerfile." We built the
> serving runtime from the hard end of that tail first: six models, three regimes, four
> silicon targets, one engine — and we're the only ones with production numbers on AMD,
> which is 26–39% cheaper per unit of work. We're raising $8–10M to turn that runtime
> into a cloud.

---

## Slide 1 · Cover — 20 sec

Say who you are and the one line. Do not explain the company yet.

> "Repercep. We're building the inference cloud for every model class — not just LLMs.
> Raising an $8–10M seed."

Then move. The cover is not where you win.

---

## Slide 2 · The observation — 60 sec

**This slide has to land or nothing after it matters.** It is the whole thesis in one
observation, and it is one most investors have not heard put this way.

> "Here's the thing we noticed. Every inference platform on the market is an LLM company
> that grew a container service for everything else. The per-token meter, the model
> catalog, the autoscaler — all of it is shaped around a stateless text decoder.
>
> **Bring them anything else and the answer is the same: here's a GPU-hour and a
> Dockerfile.** No per-call price. No workload-aware scheduling. You rent a whole GPU to
> run a forty-millisecond model and you pay for every idle second."

Then the size point, because "tail" sounds small and it isn't:

> "That tail is every production vision system, every industrial inspection line, every
> AV perception stack, every robotics fleet. It's growing faster than text-only
> inference, because it's where physical AI lives."

---

## Slide 3 · What we do — 45 sec

Three products, fast. Do not linger.

> "Repercep Cloud is the business — serverless per token or per call, dedicated per
> GPU-minute. Runtime is the same engine licensed for people who can't use a hosted
> cloud: sovereign, on-prem, defense. Link is the 2027 product — hybrid edge-to-cloud for
> robotics, and I'll come back to it."

**If asked "which one is real today?"** — Runtime is real and running. Cloud is in build.
Link is a design. Say that immediately; don't let them discover it.

---

## Slide 4 · Why now — 60 sec

> "The category question is answered. Fireworks, Together and Baseten all crossed roughly
> a billion in annualized revenue inside eighteen months. **We are not funding market
> creation — we're funding entry**, at the moment a second wave of model classes arrives."

Then the gap, which is the actual point of the slide:

> "And notice what none of them do: **none of them price AMD at all.** Modal's rate card
> doesn't have an AMD line. The one competitor with a real multi-silicon runtime supports
> NVIDIA, Trainium and TPU — and skips ROCm."

---

## Slide 5 · The moat — 45 sec

> "Cost leadership is only a moat if it comes out of the cost base, not the margin. Ours
> comes from two independent levers on the same identity: how many GPU-seconds a unit of
> work takes, and what a GPU-second costs. **Neither depends on the other**, which is what
> makes the position robust rather than one bet."

---

## Slide 6 · Lever A — 60 sec

Six measured techniques. **Do not read the table** — they can read.

> "Six techniques, all measured on both vendors, all with the raw result lines archived."

Then say the thing that buys you credibility for the rest of the deck:

> "And the bottom line matters as much as the table: **PagedAttention, continuous
> batching, speculative decoding, prefix caching — those are table stakes. We ride the
> same open engines everyone else does.** We'd rather tell you that than have you find
> it."

---

## Slides 7–8 · What we killed — 2 min · **the most important two minutes**

Slide 7 is a full-stop. Let it sit for a beat before speaking.

> "We ran the experiment that would kill our best claim. It killed it."

Then slide 8, and tell it as a story, not a retraction:

> "Our robot planner decodes sixteen candidate action sequences that share a long visual
> prefix and diverge over about seven tokens. Batching those instead of looping is a
> **five to ten times** serving win — measured on both vendors, bit-identical outputs.
>
> That shape — long shared prefix, short divergent tails, N of them, under a deadline —
> **is** best-of-N sampling. It's agentic tree search. So we had a thesis that our
> robotics work transferred to the fastest-growing LLM workload.
>
> We tested it rather than asserting it. **The transfer is real but much smaller: one
> and a half times, not five to ten.** Then we tested whether our gateway could capture
> even that for a customer who doesn't change their code — and **it can't.** Fusing
> measured slower than doing nothing.
>
> So we withdrew it. There's a slide later where a margin table used to be."

**The close, and say this deliberately:**

> "I'm telling you this because it's the fastest way for you to calibrate everything else
> in this deck. Every remaining number has been through the same treatment."

---

## Slide 9 · Lever B — 90 sec · **this is now the core of the pitch**

Since Lever A stopped being a differentiator, **this slide carries the economics.** Slow
down here.

> "Read the price table as a business plan. The platform layer charges one-point-six to
> two-point-seven times the cost of the capacity underneath it. Baseten's dedicated H100
> is six-fifty an hour; the capacity costs about two-forty-five. **And none of them price
> AMD**, where a 192-gig part rents for two to three dollars."

Then the part people miss:

> "Cheaper silicon only counts if the customer believes their outputs don't change. On a
> 7B model we reproduce the reference implementation **bit-identically — zero maximum
> error — on both vendors.** 'AMD is 30% cheaper' is a procurement argument. 'AMD is 30%
> cheaper and here's the parity gate on your model' is a product."

**Volunteer the time-box before they raise it:**

> "This is worth twelve to twenty-four months. If Fireworks ships ROCm it compresses to
> the raw price delta. The plan is to be the entrenched AMD-native platform before that
> closes."

---

## Slide 10 · Unit economics — 90 sec

Left table first, and own the weak number:

> "On plain LLM tokens, silicon arbitrage buys about 25% under the incumbent, at a gross
> margin roughly eleven points thinner than theirs. **That's real but modest — it wins a
> switch, it doesn't win a war.** Which is exactly why we cap the discount there."

Right side — the withdrawal:

> "A second table used to sit here claiming half price at 68% margin on agentic traffic.
> We measured it and removed it. Not just because our gateway couldn't deliver it —
> **because even if it had, it wouldn't differentiate us.** `n=N` is stock in every
> engine. A customer who sends it gets the same efficiency from Fireworks.
>
> What survived is smaller and actually ours: SGLang serves this shape 9 to 17% faster
> than vLLM, and **engine choice is a lever we control as the operator.** The customer
> never sees which engine runs."

---

## Slide 11 · Proof — 60 sec

> "Six models, three serving regimes, four silicon targets, one unmodified engine — built
> and benchmarked before we raised anything. The efficiency levers in this deck are
> outputs of that work, not projections."

Then the cultural asset, which after slides 7–8 they will actually believe:

> "Two GPU-verified negative findings in a single ADR. A disclosed distribution shift in
> our own best latency lever. A public note that a model's own published latency number
> wasn't full compute. **In a category where everyone benchmarks and nobody discloses,
> that's the fastest path to design-partner trust.**"

---

## Slide 12 · Repercep Link — 60 sec

> "Physical AI is a hybrid workload and nobody sells the seam. A robot can't put its
> reflexes on the network, and it can't fit long-horizon reasoning on its own processor.
> The partner's processor runs the tight loop; our cloud runs the heavy reasoning; Link
> decides what runs where and degrades gracefully to edge-only.
>
> **We don't build edge silicon.** We go through robotics companies — they own the
> silicon choice and the customer."

**Say the status without being asked:**

> "This is a design, not code. It's a 2027 product. It's the option the rest of the plan
> pays for."

---

## Slide 13 · Segments — 45 sec

Don't read five rows. Say the shape:

> "Startups are volume. **Production vision teams are the wedge** — nobody prices their
> workload per call today. Robotics is the future. Regulated and on-prem is margin, and
> hosted-only competitors structurally can't serve it. Neoclouds are the channel."

One asymmetry worth landing:

> "Neoclouds with AMD fleets have a demand problem, not a supply problem. We're one of
> very few teams who can move production inference onto idle MI300X. **The channel and
> the cost moat are the same relationship.**"

---

## Slide 14 · Competition — 45 sec

> "We lose the bottom row decisively — scale, brand, capital. They raised $800M to $1.5B.
> **The plan is to never compete where that row decides the outcome.** No brand-spend
> war, no fight for plain chat tokens. We compete on published, reproducible numbers,
> which is the one asset money can't instantly buy."

---

## Slide 15 · GTM — 45 sec

> "Self-serve is the funnel. Benchmark-led content is our unfair advantage — we publish
> cost-per-unit-of-work across model classes and vendors with the methodology and the
> negative results, which nobody else does. AMD's ecosystem is the leverage: they have a
> physical-AI story with no serving story, and we hold the only published world-model
> numbers on Instinct."

---

## Slide 16 · Roadmap — 45 sec

> "Four phases. Private beta by October. Ten paying customers and $25–50k MRR by January.
> The tail wedge proven by June. Series A at $400k MRR.
>
> **These are deliberately modest against the comparables' curves** — they're what a
> seed-stage team should be held to, not what makes a deck look good."

---

## Slide 17 · Risks — 60 sec · **don't rush this one**

> "Every risk has a named kill signal. The two that matter: **if we can't secure
> committed MI300X under $2.20 an hour by November, half the moat is gone** and we replan
> on NVIDIA. And utilization is the quiet one — every margin assumes 70%, and a young
> platform runs well below that.
>
> Company-level: no paying customers by March 2027 and the hosted thesis is wrong. The
> fallback — runtime licensing plus benchmark reports — is a real, capital-light business
> the same assets support. **That's why the downside is bounded.**"

---

## Slide 18 · Team — 45 sec

⚠️ **Fill in the founder bio before this deck is circulated.** The slide currently has a
placeholder.

> "What exists is a shipped runtime on the hardest workloads in the market, built before
> raising anything. What can't be shortcut is a year of production ROCm — not 'we support
> AMD,' but six models running with cross-vendor parity gates and a written ledger of
> every trap. A funded competitor needs quarters to reach where we start.
>
> Hiring with this round: six to eight engineers, two GTM, DevRel first — this is a
> benchmark-led, developer-adopted business."

---

## Slide 19 · The ask — 45 sec

> "Eight to ten million, twenty-eight to thirty months. **What it buys, in one sentence:
> it converts a runtime already proven on the hardest workloads into a metered cloud with
> paying customers — while the AMD window is open.**
>
> Why not bootstrap: capacity commitments and multi-tenant operations both need capital
> ahead of revenue, and the window is time-boxed."

---

## Slide 20 · Close — 20 sec

Let the slide do it. One line, then stop talking.

> "More work per GPU-second, on GPU-seconds that cost less — and we can prove your
> outputs didn't change."

**Then be silent.** The first question tells you what they actually care about.

---

## Delivery notes

**Three things to never do in this pitch:**

1. **Never quote a speedup without its workload.** 5–10× is robot action decode. 1.5× is
   LLM. They are the same technique and wildly different sizes. Getting this wrong once
   undoes slides 7–8 entirely.
2. **Never say "cheapest inference"** unqualified. Per workload class, with the cost stack
   shown.
3. **Never name a cloud partner.** Contact is informal. Named partners appear in the
   confidential plan only.

**If you only get 5 minutes:** slides 2, 9, 10, 11, 19. The observation, the silicon
lever, the economics with the withdrawal, the proof, the ask.

**If the room goes quiet after slides 7–8:** that's good. Don't fill it. Move to 9.
