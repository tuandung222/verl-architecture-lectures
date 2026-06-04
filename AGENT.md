# AGENT.md

## 1. Project overview

This repository is a Vietnamese Docusaurus curriculum for the **verl** (Volcano Engine Reinforcement Learning for LLMs) library. It is designed for deep learning engineers, AI alignment researchers, model serving developers, and systems engineers who want to understand the exact mathematical foundations, codebase implementation, and performance tuning strategies of distributed RLHF.

The goal is not simply to copy tutorials. The goal is to provide a rigorous, academic-grade guide explaining the theoretical roots (GAE, GRPO Group statistics, ReMax greedy baselines, KL trust regions) and connecting them to corresponding codebase implementations in `verl`, backed by clear pseudocode and systems algebra (resharding, communication complexity).

This file is the operating manual for future agents. Treat it as the stable source of truth for writing quality, repository safety, verification, privacy, and completion criteria.

---

## 2. Repository map

- `docs/`: public Vietnamese curriculum chapters.
  - `docs/case_studies/`: practical case studies (TinyZero, Split Placement, Multi-turn Agent, ReMax, MoE scaling).
  - `docs/theory_deep_dive/`: theoretical, mathematical, and pseudocode deep dives.
- `src/`: Docusaurus landing page and styling.
- `static/`: public static assets and `robots.txt`.
- `.github/workflows/`: CI and GitHub Pages deployment workflow.
- `README.md`: must remain empty.

---

## 3. Curriculum-wide content standard

Public content must be highly educational, mathematically rigorous, and written for advanced learners. Do not expose private task instructions, local absolute paths, credentials, internal notes, hidden constraints, or agent coordination details in public docs.

A curriculum chapter should teach by introducing a concrete tension first (e.g. why colocated memory OOMs, why standard DDP is too slow for decode steps) before offering the solution.

Use `Phần` for course sections. Do not use em dash characters. Use commas, colons, semicolons, or parentheses instead.

---

## 4. Pedagogical writing style

Future agents must write with the persona of an **AI Expert, Deep Learning & Model Serving Specialist**, and a dedicated professor. The goal is to help students grasp the absolute roots, mathematical foundations, and system mechanics of distributed reinforcement learning, rather than just high-level summaries, so they can confidently apply and implement them in real-world systems.

The prose must be highly professional, precise, serious, patient, and technically deep. It must read like an original academic lecture series in Vietnamese, not a translated or marketing-oriented document.

Use Vietnamese as the main language. Use English technical terms when they are standard in the industry: *policy, value network, rollout, actor, critic, reference policy, reward model, resharding, sequence length balancing, sequence packing, expert parallelism, sequence parallelism, offloading, baseline*. Explain a term before relying on it heavily.

Avoid casual language, slang, and jokes. The tone should be authoritative, academic, and accessible.

For every important concept, follow this pedagogical flow:
1. Start from a concrete hardware/system tension (e.g., OOM, latency, GPU underutilization).
2. Build mathematical intuition, formulate equations (using LaTeX), and prove key terms.
3. Show the corresponding clean pseudocode or algorithms.
4. Reference the actual file and function in `verl` where this is implemented.
5. Provide actionable performance tuning checklists and practical implementation guides.

---

## 5. Math, diagrams, and examples

Math must be taught, not just displayed.
- Explain every variable in equations.
- Use LaTeX formatting ($formula$ or $$formula$$).
- Follow equations with intuitive prose, such as: `Đọc công thức này theo nghĩa toán học và thực tế...` or `Bản chất của công thức nằm ở việc...`.

Use Mermaid diagrams to illustrate data flow, model sharding, communication patterns (NCCL, Ray Plasma), and reward calculation loops.

---

## 6. Public privacy and safety constraints

`README.md` must remain empty (0 bytes). Do not add any characters or placeholders to it.

Public docs must not mention:
- private user instructions or hidden agent constraints.
- the fact that `README.md` is empty.
- local absolute paths.
- credentials, tokens, secrets, API keys, or private URLs.

Privacy controls:
- `static/robots.txt` must disallow all crawling.
- Docusaurus must include `noindex,nofollow,noarchive,nosnippet` metadata.
- Sitemap generation must remain disabled.

---

## 7. Commands and verification

Safe read-only or verification commands:
- `npm run typecheck`: run TypeScript verification.
- `npm run build`: build the Docusaurus site.
- `git status --short --branch`: inspect repository state.
- `gh api repos/tuandung222/verl-architecture-lectures/pages`: verify GitHub Pages deployment status.

Commands requiring explicit approval/actions:
- Configuring or enabling GitHub Pages for the first time:
  `echo '{"source": {"branch": "gh-pages", "path": "/"}}' | gh api --method POST /repos/tuandung222/verl-architecture-lectures/pages --input -`
- Manual publishing or deploying.
- Pushing to GitHub if not already requested.
- Changing repository visibility.

---

## 8. Completion checklist

Before reporting completion, verify the relevant items:
- `README.md` is still 0 bytes.
- `npm run typecheck` and `npm run build` pass without errors.
- No em dash characters appear in public or source text.
- Public docs read like original Vietnamese teaching material.
- If pushed, the commit author and committer are the intended identity (`tuandung222`).
- The deployed website returns `HTTP 200` on the live URL.
- Search engine exclusions are active on the live site (verified through `robots.txt` disallowing `/` and `<meta name="robots" content="noindex..."/>` in page source).

---

## 9. Repo specialization: verl Internals

### Learning promise
A reader should finish this curriculum able to explain:
- Why standard training frameworks fail in RLHF due to parallel strategy mismatch.
- How `verl` decouples control flow (Ray Driver) and computation flow (Ray Workers).
- The exact mathematical difference between PPO (GAE advantage) and GRPO (group relative advantage).
- The mechanics of 3D-HybridEngine and dynamic NCCL-based weight resharding.
- The implementation of verifiable rewards, sandbox isolation, and multi-turn KV Cache optimization.
- Expert Parallelism (EP) scaling for MoE models.

### Misconceptions to actively prevent
- GRPO does not mean there is no baseline; the average group reward acts as a dynamic baseline.
- vLLM TP weights are not the same as FSDP sharded parameters; they must be reshared dynamically.
- In multi-turn agent RL, observation tokens must be masked in cross-entropy loss calculation, otherwise the Actor tries to predict the environment.
- LoRA PPO does not need a separate reference policy model; the base model serves both reference and base role dynamically (`ref_in_actor = True`).
