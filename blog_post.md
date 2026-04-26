# Teaching AI to Cooperate: How We Built an RL Environment for NGO Disaster Coordination

*A Meta PyTorch OpenEnv Hackathon submission by Team Tech Wizards — Scaler School of Technology*

---

## The Problem We Kept Thinking About

Imagine a flood just hit a city. Three NGOs show up with volunteers, supplies, and good intentions — but different priorities. One wants to deploy fast and cover the most ground. Another is careful with resources and wants nothing wasted. The third is trying to negotiate so everyone gets a fair slice of help.

In real life, this coordination happens over phone calls, WhatsApp messages, and gut decisions made under pressure. It's messy, slow, and nobody wins quite as much as they could if everyone just... worked together better.

We asked ourselves: *what if an AI could learn how to navigate this?* Not by being told the rules, but by figuring them out through experience — the way humans do.

That question became **CommunityPulse**, our NGO Multi-Agent Coordination Environment.

---

## What We Built

We built a Reinforcement Learning environment using **OpenEnv (Gymnasium)** where three AI agents represent NGO coordinators. They share a limited pool of volunteers and must decide — every step — how to allocate resources, whether to cooperate, and what to offer in negotiation.

The agents don't start knowing anything. They start completely random and learn through trial and error.

### What Each Agent Sees

Every agent observes:

- **Urgency level** — how critical is the current situation (1–10)
- **Available resources** — volunteers currently on hand
- **People affected** — how many need help right now
- **Other agents' last actions** — what their counterparts did
- **Coalition status** — who's currently allied with whom
- **Communication signals** — cooperation intent from others

### What Each Agent Can Do

Each agent outputs three values per step:

- `allocation %` — how much of the resource pool to deploy
- `cooperation signal` — willingness to work with others (0 = go it alone, 1 = full cooperation)
- `negotiation bid` — an offer to reach a compromise

### Four Scenarios, One Environment

Rather than just one task, the environment cycles through four distinct challenge types:

**Cooperation** — agents must coordinate their allocations tightly to maximize total impact. Reward goes up when variance between agents is low and total coverage is high.

**Competition** — each agent is scored individually. Being too greedy (allocating far above average) gets penalized. A Nash equilibrium naturally emerges.

**Negotiation** — fairness and efficiency are both rewarded. The agent that negotiates well, not just the one that grabs the most, comes out ahead.

**Coalition Formation** — agents that form alliances and commit high allocations together get a 1.5× bonus. Agents learn when it's worth joining a coalition and when it's better to stay independent.

---

## How Agents Actually Learn

We used an **epsilon-greedy policy gradient** approach:

- At episode 1, agents act completely randomly (ε = 1.0)
- Over training, exploration decays to just 5% (ε = 0.05)
- High-reward actions get reinforced; low-reward ones get discouraged
- Agents share a common environment but each maintains its own policy parameters

This isn't a simulation with pre-scripted outcomes. The strategies that emerge — cooperation, defection, coalition timing — aren't programmed in. They're learned.

---

## What Changed After Training

We ran 100 episodes (5,000 total environment steps) and tracked episode rewards across all four task types.

The numbers:

| | Reward |
|---|---|
| First 10 episodes (avg) | 2,353 |
| Last 10 episodes (avg) | 5,103 |
| **Improvement** | **+116.8%** |

A few things we observed as training progressed:

- In **cooperation tasks**, allocation variance dropped — agents naturally converged on similar contribution levels
- In **competition tasks**, extreme greedy behavior disappeared; a stable equilibrium formed
- In **coalition tasks**, agents learned to signal cooperation *before* high-urgency steps, not after
- The negotiation bid values gradually became more informative — agents started using them to actually communicate intent

The learning curves (generated automatically by `train.py`) show a clear upward trend from noisy early episodes to smoother, higher-reward later ones.

---

## Why It Matters

Multi-agent coordination is one of the hardest open problems in AI. Most RL work focuses on single-agent settings. But the real world is full of situations where multiple decision-makers — with different goals, different information, and different incentives — have to find a way to work together.

NGO disaster coordination is just one frame. The same environment structure applies to:

- **Hospital resource allocation** during surges (ICU beds, ventilators, staff)
- **Budget negotiations** between departments with competing priorities
- **Supply chain coalition formation** across logistics partners
- **Training LLMs** to handle multi-stakeholder conversations

If you're working on multi-agent systems and want a clean, fast, OpenEnv-compatible testbed for cooperation/competition/negotiation dynamics — this was built with that in mind.

---

## Try It Yourself

The environment is built on standard Gymnasium and runs in under 30 seconds on a CPU.

```bash
pip install -r requirements.txt
python train.py
```

Or run the full training notebook in Google Colab — no setup, no API keys, just hit **Runtime → Run all**.

**[Open in Colab →](https://github.com/Sayali5115/communitypulse-env)**
**[Hugging Face Space →](#)** *(link to be added post-deployment)*

---

## One More Thing

We named this project CommunityPulse because that's what we were going for — an AI that can feel the rhythm of a community in crisis and respond in a way that's not just efficient, but fair.

It's a small environment. A hundred episodes. Three agents. But the thing it's trying to learn — how to cooperate under pressure without being told to — feels like something worth working on.

We hope you think so too.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2025 · Theme #1: Multi-Agent Interactions*
*Team: Tech Wizards, Scaler School of Technology*
*GitHub: [github.com/Sayali5115/communitypulse-env](https://github.com/Sayali5115/communitypulse-env)*
