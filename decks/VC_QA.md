# Repercep — Investor Q&A Prep

**CONFIDENTIAL.** Companion to `repercep-seed-deck-2026-07.html` and `SPEAKER_SCRIPT.md`.

Written from the other side of the table: these are the questions I would ask if I were
deciding whether to write this cheque, phrased the blunt way they actually get asked.
Each has **the answer** (what to say), **the evidence** (what backs it), and **the trap**
(how this question is usually lost).

**§6 is the most useful section in this document** — the five questions we currently
lose. Read it first.

---

## 0. What the investor is actually trying to decide

Behind every question below are three real ones. Answer these and the meeting is won;
miss them and no individual answer helps.

1. **Is the market real?** Already yes — three comparables near $1B ARR. You will not be
   asked to prove this, and you should not spend time on it.
2. **Why this team?** The only honest answer is the year of production ROCm and the
   measurement discipline. Everything else is copyable.
3. **Why don't you get crushed?** *This is the meeting.* Three incumbents with
   $800M–$1.5B rounds are in this market. Every hard question below is a version of this
   one.

---

## 1. The existential question

### "Fireworks, Together and Baseten are all around $1B ARR with billions in the bank. Why does the world need a fourth one?"

**Answer.** It doesn't need a fourth LLM API, and we're not building one. All three are
LLM companies that grew a container service for everything else — the per-token meter,
the catalog, the scheduler are all shaped around a stateless text decoder. We start from
the other end: stateful, deadline-bound, multi-modal serving, where their answer is "rent
a GPU-hour." And we run it on silicon none of them price at all.

**Evidence.** None of the four comparables has an AMD line on its rate card. Six models,
three serving regimes, four silicon targets on one engine, all benchmarked.

**Trap.** Do not answer this by listing features. The answer is a *different starting
point*, not a longer feature list. If you find yourself comparing API ergonomics with
Fireworks, you have lost the question.

---

### "You just told me your main technical differentiator failed. What's actually left?"

**Answer.** Three things, and I'd rather be precise about them than broad. One: we can
run production inference on AMD, which is 26–39% cheaper per unit of work, with
bit-identical parity gates so a customer believes the move is safe. Two: we serve model
classes the incumbents treat as a Docker problem. Three: a measurement practice that
turns into design-partner trust faster than marketing does — which you just watched
work, since I told you about the failure before you found it.

**Evidence.** Parity 0.0 on both vendors for a 7B model; 26–39% measured cost delta;
`docs/LLM_BESTOFN_RESULT.md` for the failure.

**Trap.** Don't be defensive, and don't oversell what's left. The failed claim was one of
several and it died *before* it was sold to anyone. That is the process working, and
saying so plainly is more persuasive than minimising it.

---

## 2. The moat

### "Your entire cost story is AMD. What happens the day Fireworks ships ROCm?"

**Answer.** The lever compresses to the raw price delta, and we've time-boxed it at
12–24 months in writing. Two things make that survivable. First, "supporting ROCm" isn't
the hard part — running six models across three serving regimes with cross-vendor parity
gates is, and that took us a year of failures we've written down. Second, by the time
they ship it we intend to be the incumbent in AMD-native inference, which is a position
someone is going to own.

**Trap.** Do not claim this is a permanent moat. Claiming permanence on a copyable
advantage is how you lose credibility for the whole deck. Volunteer the time-box first.

---

### "Isn't 'we support AMD' just an arbitrage that disappears when AMD prices rise?"

**Answer.** Partly, and that's why it's a lever rather than a moat. But the second half
is structural: 192 GB versus 80 GB changes what fits, and memory density is the
denominator of per-session cost. We measured six resident sessions on MI300X against one
on H100 for a 16.5B model. That doesn't disappear with a price move.

---

## 3. Unit economics — expect the most pressure here

### "Your gross margin is 35% against the incumbent's 46%. You're entering a scale business with worse unit economics and no scale. Explain that."

**Answer.** That comparison is our *entry* price against their *mature* price, and it's
deliberately conservative — we modelled ourselves pricing 25% under them to win switches.
At the same price we'd have the better margin, because our cost base is lower. The
question isn't whether we out-margin them today; it's whether the cost base is
structurally cheaper, and AMD plus session density says it is.

**Trap.** This is the question most likely to be pressed twice. Have the arithmetic at
hand and do not fudge the utilization assumption — see the next question, which is the
real one.

---

### "You assume 70% utilization. A new platform runs at 20–30%. At 30% you're underwater. What's the plan?"

**Answer.** You're right, and it's flagged as the most likely early failure mode in our
own sensitivity table. Three mitigations: a batch tier priced at 50% to fill troughs,
dedicated deployments as a contracted utilization floor, and multi-tenant packing across
model classes — vision and LLM demand curves aren't correlated, which is a genuine
advantage of breadth that an LLM-only fleet can't copy.

**Trap.** **Do not defend 70%.** Concede it immediately. A founder who argues for a
utilization assumption looks naive; one who names it as their top risk unprompted looks
operational.

---

### "Is $2.20/hr for MI300X a quote or a hope?"

**Answer.** Today it's an assumption, tagged as one in the plan. Observed on-demand is
$1.99–$3.00. Getting a committed quote is an explicit pre-raise item, and if we can't get
under $2.20 by November the plan says replan on NVIDIA and lean on software alone.

**Trap.** Don't imply you have pricing you don't. This is checkable in one phone call.

---

## 4. The pivot

### "You spent a year on world models and now you're pivoting to LLM serving. Why should I believe this and not the next pivot?"

**Answer.** Because it isn't a change of technology — it's a change of buyer. The engine
is the same one; the ports, the kernels and the parity gates all carry. What changed is
that we stopped asking a 2027 market to buy today. Our own strategy doc set a kill
criterion, the criterion fired, and we acted on it instead of arguing with it.

**Trap.** Don't present it as a strategic epiphany. Present it as a pre-committed
decision rule that triggered. That's a much stronger signal, and it's true.

---

### "Is this a pivot or a retreat?"

**Answer.** On the revenue thesis, a retreat — we were early and we said so. On the
technology, nothing retreated: the world-model work is the reason we can serve the model
classes nobody else serves, and it's the only published cross-vendor closed-loop
benchmark anywhere. We're monetising the capability now instead of waiting for the market
that capability was built for.

---

## 5. The tail thesis

### "You say the non-LLM tail is the wedge. Name five customers who told you they'd pay per call for a CNN."

**Answer.** I can't, and that's the honest gap. What I have is a structural argument —
these teams currently rent GPU-hours to run 40 ms models and eat the idle — and a phase
in the plan whose entire purpose is to test whether they'll pay for it, with a kill
signal attached: fewer than five paying tail customers by June 2027 and we're an
AMD-native LLM cloud instead, which is a smaller but real business.

**Trap.** Do not invent demand signal. See §6 — this is one of the five we lose, and
inventing an answer here is the single most dangerous thing you could do in the meeting.

---

### "If the tail is so valuable, why hasn't Baseten done it? They have 87 clusters and $1.5B."

**Answer.** Because it's not where their next $100M is. When per-token LLM serving is
compounding, the rational move is to pour everything into it — the tail is smaller,
messier, and each model class needs real work. That's exactly why it's available to
someone starting there. It also isn't free for them: the reason we can serve six model
families on one engine is a year of unglamorous porting.

**Trap.** Don't claim they *can't*. They can. Argue that they *won't*, and why the
incentive holds.

---

## 6. The five questions we currently lose

**Read this section before every meeting.** These are real gaps. In each case the right
move is to concede fast, say what you're doing about it, and move — never to improvise.

| # | Question | Status | What to say | What to go get |
|---|---|---|---|---|
| 1 | **"What's your revenue / pipeline?"** | Zero. No LOIs. | "None. We have a runtime and benchmarks, not customers. That's what this round buys, and I'd rather show you a kill criterion than a pipeline I made up." | 3–5 discovery calls with vision/robotics teams; even unsigned interest notes |
| 2 | **"Which cloud partners are signed?"** | Informal contact only. | "None signed. We have working relationships with providers we benchmark on, and partnership conversations are early. I won't name people who haven't agreed to be named." | One signed capacity agreement or LOI |
| 3 | **"Who else is on the team?"** | Solo technical founder. | "Just me today. First three hires are serving/platform, ROCm, and an SRE — the round is sized around that." | At least one named advisor or committed first hire |
| 4 | **"Has anyone validated per-call pricing for the tail?"** | No. | "No. It's the P2 hypothesis and it has a kill signal on it." | Same as #1 |
| 5 | **"What's your committed silicon price?"** | Assumption. | "$2.20/hr is modelled, not contracted. Observed range is $1.99–3.00." | A written quote |

**The meta-answer if several land at once** — and they will:

> "You're describing a pre-seed company with a Series-A-quality technical asset. That's
> accurate. What I've built is the part that takes a year and can't be shortcut; what I
> haven't built is the commercial proof, because it needs the cloud to exist. If the gap
> between those two is too wide for you today, the honest thing is for me to come back
> with three paying customers."

That answer will lose some meetings. It will also make the ones you win much easier to
close, and it's the same posture the rest of the deck takes.

---

## 7. Technical diligence

### "Walk me through why fusing failed. I want to see if you actually understand it."

**Answer.** The lever comes from a cache-timing race, not prefill arithmetic. When N
prefix-sharing requests arrive simultaneously they all miss the prefix cache together —
none has finished prefilling to populate it. `n=N` sidesteps that by sharing the prefill
inside one request. Our gateway tried to reconstruct `n=N` by coalescing requests in a
time window, and the same property that makes the lever valuable defeats it: requests a
client issues simultaneously don't *arrive* simultaneously. A 30× longer window bought
1.8× more batching and never got the mean batch past 4.45 out of 16.

**This is a question you want.** It's where the technical depth shows.

---

### "Your 5–10× is on a robot model. Isn't quoting it near LLM numbers misleading?"

**Answer.** It would be, which is why both numbers appear with their workloads
everywhere — deck, report and website. The 5–10× is token-VLA action decode. The LLM
figure is 1.5×. Same technique, different sizes, and we publish the smaller one next to
the larger.

---

### "What have you not tested that could kill the rest of this?"

**Answer — and this is the best question anyone can ask us.** Three. One: MI300X token
throughput versus H100 — we've never measured it on our own stack, and the whole silicon
lever assumes rough parity. Two: whether the tail will pay per call. Three: whether we
can operate a multi-tenant cloud at all — we've never run one.

**Trap.** Don't answer "nothing." Anyone who has read the deck knows there's something,
and naming three is far stronger than being caught on one.

---

## 8. The ask

### "Why $8–10M? What does $4M not buy?"

**Answer.** $4M gets a beta and no capacity position. The two things that need capital
ahead of revenue are committed silicon — which is the moat — and multi-tenant operations,
which is table stakes for anyone trusting us with production traffic. $8–10M is
28–30 months, which covers both plus the Series A milestone at $400k MRR.

---

### "What happens if you don't hit $25–50k MRR by January?"

**Answer.** The plan says no paying customers by March 2027 means the hosted thesis is
wrong, and we fall back to runtime licensing plus benchmark reports — capital-light, real,
supported by the same assets. I'd rather tell you the fallback now than discover it with
your money.

---

## 9. Questions to ask them

Ending with questions signals you're choosing too. Pick two.

- "Where does your conviction on inference infrastructure sit right now — is the layer
  consolidating, or is there room below the incumbents?"
- "What would you need to see in six months to lead the A?"
- "You've seen more of this category than I have. What's the failure mode you'd expect
  for a company like ours, that I might be blind to?"

**The third one is the highest-value question in the meeting.** It's genuine, it's
flattering without being obsequious, and the answer is frequently the most useful thing
you get out of the room.

---

## 10. One-line reminders

- Every speedup travels with its workload. Every time.
- Concede utilization, the partner gap and the revenue gap **before** being pushed.
- Never name an unsigned partner.
- "Cheapest inference" is never said unqualified.
- The failed claim is an asset in this pitch. Lead with it on slide 7; don't bury it.
