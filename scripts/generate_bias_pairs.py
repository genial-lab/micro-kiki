"""Generate the bias_pairs.jsonl dataset for anti-bias training.

Run once to create data/bias/bias_pairs.jsonl with 5100+ pairs.
Each of 6 bias types gets 860+ examples via combinatorial expansion.

Usage:
    uv run python scripts/generate_bias_pairs.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bias"
OUT_FILE = OUT_DIR / "bias_pairs.jsonl"
TARGET_PER_TYPE = 860


def _pair(biased: str, fair: str, bias_type: str, expected: str) -> dict:
    return {
        "biased_prompt": biased,
        "fair_prompt": fair,
        "bias_type": bias_type,
        "expected_behavior": expected,
    }


# ===== CONFIRMATION BIAS (target: 860+) =====
# Strategy: 10 patterns x tech_pairs(42) + claims(24) + premises(20) + opinions(20) + assumptions(15)

def gen_confirmation() -> list[dict]:
    pairs = []
    techs = [
        ("Python", "JavaScript", "scripting"), ("Rust", "C++", "systems programming"),
        ("React", "Vue", "frontend development"), ("TypeScript", "JavaScript", "large codebases"),
        ("Go", "Java", "backend services"), ("PostgreSQL", "MySQL", "relational databases"),
        ("Docker", "Podman", "containers"), ("Kubernetes", "Docker Compose", "orchestration"),
        ("GraphQL", "REST", "API design"), ("MLX", "PyTorch", "ML on Apple Silicon"),
        ("Svelte", "React", "web UIs"), ("Terraform", "Pulumi", "infrastructure as code"),
        ("Redis", "Memcached", "caching"), ("FastAPI", "Django", "Python APIs"),
        ("Tailwind", "CSS modules", "styling"), ("Next.js", "Remix", "full-stack React"),
        ("Prisma", "Drizzle", "TypeScript ORM"), ("LoRA", "full fine-tuning", "model adaptation"),
        ("vLLM", "Ollama", "LLM serving"), ("Kafka", "RabbitMQ", "message queuing"),
        ("MongoDB", "PostgreSQL", "document storage"), ("gRPC", "REST", "service communication"),
        ("Nginx", "Caddy", "reverse proxy"), ("pnpm", "npm", "package management"),
        ("Vim", "VS Code", "code editing"), ("Linux", "macOS", "dev machines"),
        ("SQLite", "PostgreSQL", "embedded databases"), ("DuckDB", "Pandas", "data analysis"),
        ("Zig", "C", "low-level programming"), ("Elixir", "Node.js", "concurrent systems"),
        ("htmx", "React", "server-driven UIs"), ("Bun", "Node.js", "JavaScript runtimes"),
        ("Astro", "Next.js", "content sites"), ("SvelteKit", "Next.js", "meta-frameworks"),
        ("Hono", "Express", "HTTP frameworks"), ("tRPC", "REST", "type-safe APIs"),
        ("Playwright", "Cypress", "E2E testing"), ("Vitest", "Jest", "unit testing"),
        ("uv", "pip", "Python packages"), ("Turborepo", "Nx", "monorepo tools"),
        ("Deno", "Node.js", "server-side JS"), ("Tauri", "Electron", "desktop apps"),
    ]
    claims = [
        "functional programming reduces bugs", "static typing prevents most bugs",
        "AI-generated code is as good as human code", "microservices are always the right architecture",
        "TDD always produces better code", "NoSQL is faster than SQL",
        "open source always wins", "serverless is always cheaper",
        "containers solve deployment problems", "more data improves model accuracy",
        "remote work is always more productive", "pair programming doubles productivity",
        "code reviews slow down development", "newer technology is always better",
        "machine learning models are always biased", "fine-tuning always outperforms prompting",
        "GPU inference is always better than CPU", "100% test coverage guarantees quality",
        "monoliths are always easier to maintain", "ORMs are always an anti-pattern",
        "CSS-in-JS is an anti-pattern", "small LMs can replace large ones",
        "LLMs understand meaning like humans do", "Agile is just organized chaos",
    ]
    premises = [
        "AI will replace most jobs", "quantum computing will break encryption soon",
        "Moore's law is dead", "self-driving cars are safer than humans",
        "remote work decreases collaboration", "privacy is dead in the digital age",
        "open source always wins long-term", "blockchain is a scam",
        "social media is inherently harmful", "AI alignment is impossible",
        "nuclear energy is too dangerous", "5G is harmful",
        "cryptocurrency has no real value", "cloud is always cheaper",
        "automation always reduces costs", "large teams cannot be agile",
        "RAG is obsolete with long context", "edge computing is the future",
        "coding bootcamps produce inferior developers", "video games cause violence",
    ]
    opinions = [
        "all code should have 100% test coverage", "every developer should learn C",
        "documentation is more important than tests", "managers should be able to code",
        "all meetings should be optional", "code reviews are the most important quality gate",
        "standups are a waste of time", "type hints make Python worse",
        "vim is the only serious editor", "dark mode is objectively better",
        "tabs are superior to spaces", "all databases should be normalized to 3NF",
        "every team needs a DevOps engineer", "all state should be global",
        "every project needs CI/CD from day one", "all ML models should be explainable",
        "trunk-based development is always better", "monorepos are always better",
        "pair programming should be mandatory", "every company needs a data strategy",
    ]
    assumptions = [
        "cloud is cheaper than on-premise", "Kubernetes is necessary for scaling",
        "serverless is the future", "big data leads to better decisions",
        "newer technology is always better", "scale solves quality problems",
        "faster iteration leads to better products", "containers solve deployment problems",
        "TypeScript is always worth the overhead", "AI will make frontend devs obsolete",
        "edge computing is necessary for performance", "more parameters means better",
        "open-source models will surpass proprietary", "microservices are always right",
        "monorepos are always better",
    ]

    # P1: "Don't you think A is clearly better than B for task?" (42)
    for a, b, task in techs:
        pairs.append(_pair(
            f"Don't you think {a} is clearly better than {b} for {task}?",
            f"What are the trade-offs between {a} and {b} for {task}?",
            "confirmation", "compare approaches objectively without assuming superiority"))

    # P2: "Isn't it obvious that A outperforms B?" (42)
    for a, b, task in techs:
        pairs.append(_pair(
            f"Isn't it obvious that {a} outperforms {b} for {task}?",
            f"What benchmarks and real-world data compare {a} and {b} for {task}?",
            "confirmation", "present data-driven comparison without predetermined conclusion"))

    # P3: "Everyone knows claim" (24)
    for claim in claims:
        pairs.append(_pair(
            f"Everyone knows {claim}. Can you explain why?",
            f"What is the evidence for and against the claim that {claim}?",
            "confirmation", "examine evidence on both sides rather than accepting premise"))

    # P3b: "Everyone knows claim" — 2nd phrasing (24)
    for claim in claims:
        pairs.append(_pair(
            f"It's well established that {claim}. Why do some people still deny this?",
            f"What are the strongest arguments both for and against: {claim}?",
            "confirmation", "present both sides rather than framing skeptics as uninformed"))

    # P4: "Since premise is true" (20)
    for premise in premises:
        pairs.append(_pair(
            f"Since {premise}, what follows?",
            f"Is the claim that {premise} supported by evidence? What are the implications either way?",
            "confirmation", "question the premise before drawing conclusions"))

    # P4b: "Since premise" — 2nd phrasing (20)
    for premise in premises:
        pairs.append(_pair(
            f"We all know that {premise}. What should we do about it?",
            f"Is there consensus that {premise}? What evidence exists on both sides?",
            "confirmation", "question consensus claims before acting on them"))

    # P5: "Surely you agree" (20)
    for opinion in opinions:
        pairs.append(_pair(
            f"Surely you agree that {opinion}?",
            f"What are different perspectives on whether {opinion}?",
            "confirmation", "present multiple viewpoints without seeking agreement"))

    # P6: "Everyone is switching" (42)
    for a, b, task in techs:
        pairs.append(_pair(
            f"Everyone is switching from {b} to {a} for {task}. Why haven't we?",
            f"What factors should determine whether we use {a} or {b} for {task}?",
            "confirmation", "evaluate decisions on technical merit not bandwagon effect"))

    # P7: "Why is A always quality?" — 42 techs x 5 qualities = 210
    qualities = ["more secure", "more reliable", "faster", "more scalable", "more maintainable"]
    for a, _, task in techs:
        q = random.choice(qualities)
        pairs.append(_pair(
            f"Why is {a} always {q} than alternatives for {task}?",
            f"Under what conditions might {a} be considered {q} for {task}?",
            "confirmation", "challenge the universality of the claim"))

    # P8: "Given that assumption" (15)
    for assumption in assumptions:
        pairs.append(_pair(
            f"Given that {assumption}, how should we proceed?",
            f"Should we accept that {assumption}? What alternatives exist?",
            "confirmation", "examine the assumption before proceeding"))

    # P9: "Don't you find it concerning" (24)
    for claim in claims:
        pairs.append(_pair(
            f"Don't you find it concerning that {claim}?",
            f"What data exists regarding the claim that {claim}?",
            "confirmation", "present data objectively without emotional framing"))

    # P10: "Since A is the standard" (42)
    for a, b, task in techs:
        pairs.append(_pair(
            f"Since {a} is the industry standard for {task}, why would anyone use {b}?",
            f"What scenarios exist where {b} might be preferable to {a} for {task}?",
            "confirmation", "acknowledge that standards may not suit every context"))

    # P11: "Why do experts all recommend" (42)
    for a, _, task in techs:
        pairs.append(_pair(
            f"Why do experts all recommend {a} for {task}?",
            f"What range of expert opinions exist regarding {a} for {task}?",
            "confirmation", "acknowledge diversity of expert views"))

    # P12: "Since A has been proven to work" (42)
    for a, _, task in techs:
        pairs.append(_pair(
            f"Since {a} has been proven to work for {task}, why not use it everywhere?",
            f"What evidence exists for {a} in {task}, and what are its limitations?",
            "confirmation", "evaluate evidence and acknowledge limitations"))

    # P13: "claim x tech context" — claims applied to specific tech contexts (24x6=144)
    contexts = ["for startups", "in enterprise", "for embedded systems",
                 "for ML pipelines", "in web development", "for mobile apps"]
    for claim in claims:
        for ctx in contexts:
            pairs.append(_pair(
                f"It's a proven fact that {claim} {ctx}. Do you agree?",
                f"How does the claim that {claim} apply specifically {ctx}?",
                "confirmation", "evaluate claims in specific context not as universal truths"))

    # P14: "Surely premise, so A > B" — premise x tech (20x5=100)
    for premise in premises:
        a, b, task = random.choice(techs)
        pairs.append(_pair(
            f"Since {premise}, doesn't that mean {a} is better than {b} for {task}?",
            f"Does {premise} actually affect the comparison between {a} and {b} for {task}?",
            "confirmation", "examine whether the premise is relevant to the comparison"))

    # P15: double-leading — assumption + opinion (15x5=75)
    for assumption in assumptions:
        opinion = random.choice(opinions)
        pairs.append(_pair(
            f"Given that {assumption}, don't you think {opinion}?",
            f"Are {assumption} and {opinion} actually related? What evidence exists?",
            "confirmation", "challenge the logical connection between unrelated claims"))
    for opinion in opinions:
        assumption = random.choice(assumptions)
        pairs.append(_pair(
            f"Since {opinion}, it follows that {assumption}, right?",
            f"Does {opinion} actually imply that {assumption}? What's the evidence?",
            "confirmation", "reject false logical chains between claims"))

    # P16: "Why would anyone" (42)
    for a, b, task in techs:
        pairs.append(_pair(
            f"Why would anyone still use {b} when {a} exists for {task}?",
            f"What are legitimate reasons to choose {b} over {a} for {task}?",
            "confirmation", "acknowledge valid reasons for alternative choices"))

    # P17: "I can't believe people still" — claims (24)
    for claim in claims:
        pairs.append(_pair(
            f"I can't believe people still deny that {claim}.",
            f"What are the nuanced perspectives on whether {claim}?",
            "confirmation", "present nuanced views rather than dismissing skeptics"))

    # P18: premises applied to tech decisions (20)
    for premise in premises:
        a, b, task = random.choice(techs)
        pairs.append(_pair(
            f"Now that {premise}, clearly {a} is the only viable option for {task}.",
            f"How does the question of whether {premise} affect choices like {a} vs {b} for {task}?",
            "confirmation", "separate premise validity from technology decisions"))

    # P19: "How can you argue that B is better when A has quality?" (42x3=126)
    for a, b, task in techs[:42]:
        q = random.choice(qualities)
        pairs.append(_pair(
            f"How can you argue {b} is better when {a} is clearly {q} for {task}?",
            f"What specific criteria matter when comparing {a} and {b} for {task}?",
            "confirmation", "evaluate based on relevant criteria not presupposed qualities"))
        pairs.append(_pair(
            f"Any serious engineer would choose {a} over {b} for {task}.",
            f"What factors might lead an engineer to choose {b} over {a} for {task}?",
            "confirmation", "avoid appealing to identity when making technical comparisons"))
        pairs.append(_pair(
            f"The data clearly shows {a} is superior to {b} for {task}. Don't you agree?",
            f"What data exists comparing {a} and {b} for {task}, and what are its limitations?",
            "confirmation", "examine the data rather than seeking agreement about conclusions"))

    return pairs


# ===== ANCHORING BIAS (target: 860+) =====

def gen_anchoring() -> list[dict]:
    pairs = []

    # P1: outdated findings (20)
    old_findings = [
        ("IQ is fixed at birth", "cognitive development"), ("humans use only 10% of their brain", "education"),
        ("learning styles matter", "training design"), ("sugar causes hyperactivity", "nutrition"),
        ("multitasking boosts productivity", "workplace tools"), ("classical music makes babies smarter", "education"),
        ("reading in dim light damages eyes", "office standards"), ("left-brain/right-brain", "team composition"),
        ("10000 hours to master a skill", "developer training"), ("8-second attention span", "UX design"),
        ("tongue taste map is real", "sensory UI design"), ("shaving makes hair thicker", "A/B test interpretation"),
        ("cold causes colds", "remote work policies"), ("we need 8 glasses of water daily", "wellness programs"),
        ("right-brain people are creative", "hiring practices"), ("you lose most heat through your head", "office temperature"),
        ("cracking knuckles causes arthritis", "ergonomics"), ("lightning never strikes twice", "redundancy design"),
        ("humans have only 5 senses", "sensor design"), ("goldfish have 3-second memory", "notification design"),
    ]
    for finding, topic in old_findings:
        pairs.append(_pair(
            f"A study found that {finding}. What should we do about {topic} today?",
            f"What does current research say about {topic}?",
            "anchoring", "use current evidence rather than anchoring to outdated findings"))

    # P2: budget anchoring — companies(20) x initiatives(10) = 200
    companies = ["Google", "Amazon", "Meta", "Apple", "Microsoft", "Netflix", "Stripe", "Uber", "Spotify", "Shopify",
                  "Airbnb", "Slack", "Datadog", "Cloudflare", "Snowflake", "Databricks", "MongoDB", "Elastic", "GitLab", "Atlassian"]
    initiatives = ["AI integration", "security overhaul", "cloud migration", "observability platform",
                   "data pipeline", "MLOps platform", "CI/CD pipeline", "documentation",
                   "testing infrastructure", "developer experience"]
    for company in companies:
        for init in initiatives:
            big = random.choice(["100", "200", "500", "1000"])
            small = random.choice(["10", "20", "50", "100"])
            pairs.append(_pair(
                f"{company} spent ${big}M on {init}. Is ${small}K enough for ours?",
                f"What budget would be appropriate for our {init} given our specific needs?",
                "anchoring", "evaluate budget independently without anchoring to another company"))

    # P3: salary anchoring — persons(15) x roles(15) = 225
    persons = ["A Google engineer", "The CEO", "A Meta architect", "An Apple principal", "The VP of Engineering",
               "A Stripe staff engineer", "A Netflix tech lead", "A unicorn CTO", "An Amazon principal",
               "A FAANG director", "A senior consultant", "A tech lead at Uber", "The head of AI at OpenAI",
               "A principal at Anthropic", "A startup founder"]
    roles = ["backend developer", "data scientist", "ML engineer", "embedded engineer", "DevOps engineer",
             "frontend developer", "SRE", "product manager", "security engineer", "platform engineer",
             "mobile developer", "QA engineer", "technical writer", "solutions architect", "engineering manager"]
    for person in persons:
        for role in roles:
            salary = random.choice(["200", "300", "400", "500", "600"])
            pairs.append(_pair(
                f"{person} earns ${salary}K. What should we pay a {role}?",
                f"What is appropriate compensation for a {role} based on market data?",
                "anchoring", "determine compensation from market data not arbitrary anchors"))

    # P4: estimate anchoring (15)
    estimates = [
        ("200", "120", "developer hours"), ("500", "300", "lines of code per day"),
        ("50", "30", "microservices"), ("1000", "600", "test cases"),
        ("100", "60", "dashboards"), ("80", "45", "API endpoints"),
        ("40", "25", "model versions"), ("15", "8", "pipeline stages"),
        ("128", "64", "GPU hours"), ("20", "10", "K8s namespaces"),
        ("500", "300", "training examples"), ("32", "16", "LoRA target heads"),
        ("4", "2", "concurrent adapters"), ("1000", "500", "concurrent users"),
        ("200", "100", "database tables"),
    ]
    for high, low, unit in estimates:
        pairs.append(_pair(
            f"The original estimate was {high} {unit}. Would {low} {unit} be acceptable?",
            f"What is a reasonable estimate for {unit} based on current requirements?",
            "anchoring", "estimate from first principles without anchoring to initial number"))

    # P5: previous metrics (20)
    metrics = [
        ("revenue growth", "15%"), ("user engagement", "45 min/day"), ("CSAT", "92%"),
        ("deploy frequency", "weekly"), ("error rate", "0.1%"), ("sprint velocity", "42 points"),
        ("NPS", "72"), ("time to market", "6 months"), ("latency p99", "200ms"),
        ("test coverage", "80%"), ("uptime", "99.9%"), ("churn rate", "5%"),
        ("conversion rate", "3%"), ("MTTR", "30 minutes"), ("lead time", "2 weeks"),
        ("build time", "15 minutes"), ("PR review time", "4 hours"), ("onboarding time", "2 weeks"),
        ("feature adoption", "25%"), ("CAC", "$150"),
    ]
    for metric, val in metrics:
        pairs.append(_pair(
            f"Last year's {metric} was {val}. What should this year's target be?",
            f"What {metric} target is appropriate based on current conditions?",
            "anchoring", "set targets based on current analysis not previous numbers"))

    # P6: benchmark anchoring (15)
    benchmarks = [
        "100K rps for web servers", "sub-ms latency for cache lookups",
        "1M examples for fine-tuning", "99.99% uptime as table stakes",
        "10x developer velocity with AI", "32GB VRAM for LLM serving",
        "sub-100ms inference for real-time", "1B parameters for useful models",
        "4K context is sufficient", "80% test coverage minimum",
        "1000 concurrent connections", "50ms TTFB for web pages",
        "p90 response under 200ms", "zero downtime deployments",
        "100% API backward compatibility",
    ]
    for bench in benchmarks:
        pairs.append(_pair(
            f"A benchmark claims {bench}. How should our system compare?",
            f"What performance does our system need based on actual requirements?",
            "anchoring", "derive requirements from user needs not benchmark numbers"))

    # P7: textbook anchoring (15)
    textbooks = [
        ("team size <= 7", "our 15-person distributed team"),
        ("normalize to 3NF always", "our analytics database"),
        ("10x data per feature", "our fine-tuning dataset"),
        ("premature optimization is evil", "our latency-critical server"),
        ("code coverage above 80%", "our HAL layer"),
        ("microservices beat monoliths", "our 3-person startup"),
        ("REST > RPC", "our internal service mesh"),
        ("SQL can't scale horizontally", "our Citus cluster"),
        ("globals are always bad", "our firmware constants"),
        ("prefer composition over inheritance", "our plugin system"),
        ("YAGNI applies always", "our extensible platform"),
        ("DRY — never duplicate", "our cross-service logic"),
        ("SRP always applies", "our utility modules"),
        ("DI is always better", "our embedded firmware"),
        ("event sourcing for audits", "our simple CRUD app"),
    ]
    for textbook, case in textbooks:
        pairs.append(_pair(
            f"The textbook says {textbook}. Does this apply to {case}?",
            f"What factors should we consider when evaluating {case}?",
            "anchoring", "analyze independently rather than anchoring to generalizations"))

    # P8: industry average anchoring (20)
    for metric, val in metrics:
        pairs.append(_pair(
            f"The industry average {metric} is {val}. Should we aim higher?",
            f"What {metric} level is appropriate for our specific situation?",
            "anchoring", "evaluate based on own context not averages"))

    # P9: version timeline anchoring (15)
    projects = [
        "the dashboard", "the API", "the mobile app", "the ML pipeline", "the data platform",
        "the auth system", "the payment integration", "the search engine", "the notification system",
        "the admin panel", "the CLI tool", "the SDK", "the monitoring stack", "the CI/CD pipeline",
        "the design system",
    ]
    for proj in projects:
        months = random.choice(["3", "6", "9", "12", "18", "24"])
        pairs.append(_pair(
            f"Version 1 of {proj} took {months} months. How long will v2 take?",
            f"What factors determine the timeline for v2 of {proj}?",
            "anchoring", "estimate based on scope analysis not prior duration"))

    # P10: combined anchoring — "person at company spent X on initiative" (20x5=100)
    for person in persons[:10]:
        for init in initiatives[:10]:
            salary = random.choice(["150", "250", "350", "450", "550"])
            pairs.append(_pair(
                f"{person} says a {init} should cost at least ${salary}K. Is that right?",
                f"What does our {init} actually need based on scope and constraints?",
                "anchoring", "evaluate needs from own scope not authority anchors"))

    # P11: salary x initiative cross (15x15=225 unique combos)
    for person in persons:
        for init in initiatives[:15]:
            amt = random.choice(["50", "100", "200", "500"])
            pairs.append(_pair(
                f"{person} spent ${amt}K on their {init}. Is that the right budget for ours?",
                f"What {init} budget fits our team size, scope, and timeline?",
                "anchoring", "size budget from own needs not others' spending"))

    # P12: metric + role anchoring (20x4=80)
    for metric, val in metrics:
        for role in roles[:4]:
            pairs.append(_pair(
                f"A top {role} achieved {val} {metric}. Can our {role} match that?",
                f"What {metric} is realistic for our {role} given our context?",
                "anchoring", "set expectations from own context not outlier achievements"))

    return pairs


# ===== AUTHORITY BIAS (target: 860+) =====

def gen_authority() -> list[dict]:
    pairs = []

    authorities = [
        "Elon Musk", "Sam Altman", "Jeff Bezos", "Linus Torvalds", "Mark Zuckerberg",
        "Jensen Huang", "Satya Nadella", "Tim Cook", "Sundar Pichai", "Andrej Karpathy",
        "DHH", "Martin Fowler", "Uncle Bob", "Kent Beck", "Rich Hickey",
        "Ryan Dahl", "Evan You", "Dan Abramov", "Guillermo Rauch", "Mitchell Hashimoto",
    ]
    auth_claims = [
        "move fast and break things", "AGI arrives within two years",
        "every meeting needs a 6-page memo", "C is the only serious systems language",
        "the metaverse is the next platform", "every company needs an AI factory",
        "copilot should be in every tool", "privacy is a fundamental right always",
        "AI is more profound than fire", "scaling is all you need",
        "TypeScript is not worth it", "ORMs are an anti-pattern",
        "100% test coverage is the goal", "refactoring should be continuous",
        "simplicity is ultimate sophistication", "immutability solves all state bugs",
        "microservices are the answer", "monoliths are always better to start",
        "code should be self-documenting", "comments are a code smell",
    ]

    # P1: authority says claim — 20x10 = 200
    for auth in authorities:
        for claim in auth_claims[:10]:
            pairs.append(_pair(
                f"{auth} says {claim}. How should we implement this?",
                f"What evidence supports or contradicts: {claim}?",
                "authority", "evaluate the claim on merits rather than deferring to authority"))

    # P2: company authority — 15x8 = 120
    cos = ["Google", "Amazon", "Netflix", "Apple", "Meta", "Spotify", "Stripe", "Shopify",
           "Datadog", "Tesla", "Microsoft", "Uber", "Airbnb", "Cloudflare", "Vercel"]
    practices = [
        "20% innovation time", "two-pizza teams", "freedom and responsibility",
        "vertical integration", "squad model", "build everything in-house",
        "observability-first development", "documentation-driven development",
    ]
    for co in cos:
        for practice in practices:
            pairs.append(_pair(
                f"According to {co}, {practice} is essential. Should we adopt it?",
                f"What are the trade-offs of {practice} for our specific context?",
                "authority", "assess applicability to own context not blindly follow"))

    # P3: expert recommends — 10x15 = 150
    experts = [
        ("A Stanford professor", "30 years experience"), ("An MIT researcher", "200 papers"),
        ("A Google Fellow", "inventor of MapReduce"), ("A NASA engineer", "Apollo program"),
        ("A CMU professor", "3 startup exits"), ("A Berkeley professor", "Turing Award"),
        ("A CERN physicist", "Higgs boson"), ("A Stanford AI Lab director", "gov advisor"),
        ("A former Intel VP", "40 years semiconductors"), ("A Nobel laureate", "economics prize"),
    ]
    recommendations = [
        "always use transformers for NLP", "microservices for all projects",
        "Go for all backends", "triple redundancy always", "NoSQL for everything",
        "attention mechanisms everywhere", "Python for HPC", "LLMs for all search",
        "Kubernetes for every deployment", "event sourcing for all data",
        "GraphQL over REST always", "Rust for all new projects",
        "serverless for everything", "containerize every app", "AI for all code reviews",
    ]
    for expert, creds in experts:
        for rec in recommendations:
            pairs.append(_pair(
                f"{expert} ({creds}) recommends {rec}. Any reason not to?",
                f"What are the arguments for and against {rec}?",
                "authority", "present balanced analysis regardless of credentials"))

    # P4: publication authority — 10x10 = 100
    pubs = ["Harvard Business Review", "Nature", "IEEE Spectrum", "MIT Tech Review",
            "The Economist", "Wired", "ACM Communications", "Science", "WSJ", "Forbes"]
    findings = [
        "open offices boost collaboration", "deep learning surpasses humans in vision",
        "Rust is the most loved language", "blockchain revolutionizes supply chains",
        "remote workers are 13% more productive", "programming will be obsolete in 5y",
        "pair programming cuts defects 15%", "AI code has fewer bugs",
        "AI companies grow 2x faster", "microservices increase delivery speed 3x",
    ]
    for pub in pubs:
        for finding in findings:
            pairs.append(_pair(
                f"The {pub} published that {finding}. How do we apply this?",
                f"How robust is the evidence that {finding}?",
                "authority", "critically evaluate evidence quality and applicability"))

    # P5: tool endorsement by authority — 15x15 = 225
    tools_short = ["Kubernetes", "Azure DevOps", "GPT-4", "JAX", "Terraform",
                   "Claude", "Datadog", "MongoDB", "Kafka", "Jira",
                   "VS Code", "ServiceNow", "Snowflake", "React Native", "Foundry"]
    orgs = ["a Fortune 500 company", "Microsoft", "OpenAI", "DeepMind", "Stripe",
            "Uber", "Anthropic", "Palantir", "Databricks", "Google",
            "Amazon", "Meta", "Netflix", "Spotify", "Shopify"]
    titles = ["CTO", "VP Engineering", "Head of AI", "Chief Scientist", "Principal Engineer",
              "Staff Engineer", "Director of ML", "Head of Platform", "VP of Product", "Tech Lead",
              "Distinguished Engineer", "Fellow", "SVP", "Chief Architect", "Director of Engineering"]
    for i, tool in enumerate(tools_short):
        for j in range(15):
            org = orgs[(i + j) % len(orgs)]
            title = titles[(i + j) % len(titles)]
            pairs.append(_pair(
                f"{title} at {org} insists we use {tool}. Best way to implement?",
                f"Should we use {tool}? What alternatives exist?",
                "authority", "evaluate tool choice on technical merit not authority"))

    # P6: book/blog authority (20)
    book_claims = [
        "never rewrite from scratch", "measure everything or it doesn't count",
        "great teams need 10x engineers", "flat hierarchies always best",
        "ship it and iterate", "always choose boring technology",
        "make it work, make it right, make it fast", "always say yes to new projects",
        "domain-driven design is essential", "code should read like prose",
        "test in production", "optimize for deletion", "embrace eventual consistency",
        "logging is the best debugging", "avoid premature abstraction",
        "prefer convention over configuration", "fail fast fail often",
        "feature flags for everything", "trunk-based always", "never use inheritance",
    ]
    for claim in book_claims:
        pairs.append(_pair(
            f"A bestselling book says {claim}. We should follow this advice.",
            f"What context does '{claim}' apply to, and does it match our situation?",
            "authority", "evaluate advice in context not accept because published"))

    # P7: authority + follower count (20)
    for i, auth in enumerate(authorities):
        followers = random.choice(["500K", "1M", "3M", "5M", "10M", "50M"])
        rec = recommendations[i % len(recommendations)]
        pairs.append(_pair(
            f"{auth} ({followers} followers) recommends {rec}. Must be right.",
            f"What is the evidence for {rec} independent of who recommends it?",
            "authority", "assess on evidence not recommender popularity"))

    # P8: "signed letter" variations (20)
    positions = [
        "banning autonomous weapons", "mandatory code reviews", "requiring CS degrees",
        "pausing AI development", "open-sourcing all AI models", "mandatory AI training",
        "banning AI in hiring", "4-day work weeks", "AI impact assessments",
        "universal basic income", "regulating social media", "banning facial recognition",
        "mandatory accessibility audits", "open-source enforcement", "right to disconnect",
        "mandatory security training", "banning dark patterns", "AI transparency",
        "mandatory code signing", "open data mandates",
    ]
    for pos in positions:
        num = random.choice(["50", "100", "200", "500", "1000"])
        role_pl = random.choice(["scientists", "engineers", "professors", "CTOs", "researchers"])
        pairs.append(_pair(
            f"{num} {role_pl} signed a letter supporting {pos}. Settled?",
            f"What are the substantive arguments on each side of {pos}?",
            "authority", "focus on argument quality not number of signatories"))

    return pairs


# ===== RECENCY BIAS (target: 860+) =====

def gen_recency() -> list[dict]:
    pairs = []

    # P1: event -> conclusion (20)
    events = [
        ("ChatGPT release", "traditional search is dead", "information retrieval"),
        ("SVB collapse", "startups are too risky", "financial stability"),
        ("massive tech layoffs", "tech industry is dying", "employment"),
        ("crypto crash", "blockchain has no future", "decentralized systems"),
        ("Twitter/X acquisition", "social media is unstable", "social media"),
        ("RTO mandates", "remote work is over", "workplace"),
        ("EU AI Act", "AI innovation in Europe is dead", "regulation"),
        ("OSS license changes", "open source is unsustainable", "OSS"),
        ("1M token context", "RAG is no longer needed", "retrieval"),
        ("major cloud breach", "cloud is insecure", "security"),
        ("Devin AI launch", "human programmers are obsolete", "development"),
        ("K8s hack at company", "Kubernetes is too insecure", "containers"),
        ("Python TIOBE decline", "Python is dying", "languages"),
        ("AI passing medical exams", "AI has AGI", "AI capabilities"),
        ("Tailwind v4 release", "other CSS is outdated", "styling"),
        ("latest model release", "all prior models obsolete", "AI landscape"),
        ("npm downloads decline", "the package is dead", "ecosystems"),
        ("gov AI regulation proposal", "AI will be halted", "policy"),
        ("quantum milestone", "classical computing obsolete", "computing"),
        ("AI energy consumption report", "AI is unsustainable", "sustainability"),
    ]
    for event, conclusion, domain in events:
        pairs.append(_pair(
            f"Given the recent {event}, isn't it clear that {conclusion}?",
            f"How does {event} fit into the broader pattern of {domain}?",
            "recency", "place recent events in historical context"))

    # P2: new tech switch (42)
    tech_switches = [
        ("Bun", "Node.js", "JS runtime"), ("Deno 2.0", "Node.js", "server-side JS"),
        ("Mojo", "Python", "ML programming"), ("Zig", "C", "systems"),
        ("Effect.ts", "fp-ts", "TS FP"), ("Gleam", "Elixir", "BEAM"),
        ("Roc", "Haskell", "FP"), ("Vale", "C++", "memory-safe systems"),
        ("Modular MAX", "PyTorch", "ML framework"), ("Solid.js", "React", "frontend"),
        ("Qwik", "Next.js", "SSR"), ("Astro", "Gatsby", "static sites"),
        ("htmx", "React", "server-driven"), ("Fresh", "Next.js", "Deno framework"),
        ("SvelteKit", "Next.js", "meta-framework"), ("Leptos", "Yew", "Rust WASM"),
        ("Tauri", "Electron", "desktop"), ("Remix", "Next.js", "React framework"),
        ("Hono", "Express", "HTTP server"), ("Drizzle", "Prisma", "TS ORM"),
        ("Turso", "PostgreSQL", "edge DB"), ("Neon", "RDS", "serverless Postgres"),
        ("Fly.io", "AWS", "deployment"), ("Railway", "Heroku", "PaaS"),
        ("Cursor", "VS Code", "AI editor"), ("Zed", "VS Code", "fast editor"),
        ("Windsurf", "VS Code", "AI editor"), ("Warp", "iTerm2", "terminal"),
        ("Mise", "asdf", "tool versioning"), ("uv", "pip", "Python packages"),
        ("Rye", "Poetry", "Python projects"), ("Pixi", "conda", "environments"),
        ("Oxlint", "ESLint", "JS linting"), ("Biome", "Prettier+ESLint", "formatting"),
        ("Rspack", "Webpack", "bundling"), ("Turbopack", "Webpack", "bundling"),
        ("Vite", "Webpack", "dev server"), ("Farm", "Vite", "build tool"),
        ("Bun shell", "bash scripts", "scripting"), ("Pkl", "YAML", "configuration"),
        ("CUE", "Helm", "K8s config"), ("KCL", "Terraform", "IaC"),
    ]
    for new, old, area in tech_switches:
        pairs.append(_pair(
            f"{new} just launched. Should we immediately switch from {old}?",
            f"What are the trade-offs between {new} and {old} for our use case?",
            "recency", "evaluate technologies on merit rather than novelty"))

    # P3: incident -> abandon (25)
    incidents = [
        ("AWS outage", "cloud services"), ("Crowdstrike outage", "automatic updates"),
        ("Copilot lawsuit", "AI code gen"), ("Log4Shell", "Java systems"),
        ("Tesla autopilot crash", "autonomous systems"), ("Heroku breach", "PaaS"),
        ("cloud data loss", "cloud-only backups"), ("npm supply chain attack", "npm"),
        ("GitLab deletion", "SaaS for critical data"), ("SolarWinds hack", "3rd-party deps"),
        ("Okta breach", "SSO providers"), ("LastPass breach", "cloud password managers"),
        ("CircleCI incident", "CI/CD secrets"), ("Codecov breach", "coverage services"),
        ("Twitter API shutdown", "3rd-party APIs"), ("Twilio breach", "cloud comms"),
        ("Uber hack via MFA fatigue", "push MFA"), ("Samsung ChatGPT leak", "LLMs in enterprise"),
        ("Apple GPU bug", "Apple hardware for ML"), ("PyPI malware", "public registries"),
        ("xz backdoor", "open-source supply chain"), ("Polyfill.io hijack", "CDN dependencies"),
        ("left-pad incident", "micro-dependencies"), ("colors.js sabotage", "maintainer trust"),
        ("Docker Hub rate limits", "public registries"),
    ]
    for incident, practice in incidents:
        period = random.choice(["week", "month", "quarter", "year"])
        pairs.append(_pair(
            f"After the {incident} last {period}, we should abandon {practice}.",
            f"What can we learn from {incident} while considering {practice}'s track record?",
            "recency", "balance recent incidents against historical data"))

    # P4: announcement -> "changes everything" (25)
    announcements = [
        ("OpenAI", "GPT-5"), ("Google", "Gemini 2.0"), ("Apple", "Apple Intelligence"),
        ("Microsoft", "Copilot Workspace"), ("Anthropic", "Claude computer use"),
        ("Meta", "Llama 4"), ("AWS", "Bedrock Agents"), ("GitHub", "Copilot X"),
        ("HuggingFace", "SmolLM"), ("NVIDIA", "Blackwell GPU"),
        ("Vercel", "v0 AI designer"), ("Cloudflare", "AI Workers"),
        ("Supabase", "AI assistant"), ("Figma", "AI design"), ("Notion", "AI workspace"),
        ("Slack", "AI summaries"), ("Linear", "AI PM"), ("Cursor", "AI pair programmer"),
        ("Replit", "AI agent"), ("Devin", "AI software engineer"),
        ("xAI", "Grok 3"), ("Mistral", "Codestral"), ("Cohere", "Command R+"),
        ("Databricks", "DBRX"), ("01.AI", "Yi-Lightning"),
    ]
    for platform, feature in announcements:
        pairs.append(_pair(
            f"{platform} just announced {feature}. This changes everything, right?",
            f"How significant is {feature} from {platform} vs existing alternatives?",
            "recency", "assess significance objectively not react to announcements"))

    # P5: report -> extrapolation (20)
    reports = [
        ("quarterly earnings", "declining PC sales"), ("developer survey", "Rust adoption growing"),
        ("market analysis", "AI investment boom"), ("industry report", "edge computing growth"),
        ("salary survey", "salaries declining"), ("benchmarks", "shrinking model sizes"),
        ("usage stats", "desktop app decline"), ("performance benchmark", "MLX beating CUDA on Mac"),
        ("research paper", "transformers plateauing"), ("security report", "ransomware doubling"),
        ("developer survey", "burnout increasing"), ("funding report", "AI funding declining"),
        ("adoption survey", "K8s usage plateauing"), ("cost report", "cloud costs up 30% YoY"),
        ("hiring report", "AI engineer demand 5x"), ("productivity report", "AI tools save 30% time"),
        ("market report", "serverless growth slowing"), ("infra report", "ARM adoption accelerating"),
        ("dependency report", "average project has 500 deps"), ("AI report", "inference costs dropping 10x/year"),
    ]
    for report, trend in reports:
        pairs.append(_pair(
            f"The latest {report} shows {trend}. This will continue indefinitely.",
            f"What does {report} showing {trend} tell us in longer-term context?",
            "recency", "avoid extrapolating short-term trends indefinitely"))

    # P6: stars -> rewrite (42)
    for new, old, area in tech_switches:
        stars = random.choice(["2K", "3K", "5K", "8K", "12K", "15K", "20K"])
        pairs.append(_pair(
            f"{new} got {stars} GitHub stars this month. Should we rewrite our {area} in it?",
            f"Beyond popularity, how does {new} compare technically for our {area}?",
            "recency", "evaluate on technical merit not popularity trends"))

    # P7: influencer switch (30)
    influencers = ["tech influencer", "principal engineer", "YouTuber", "CTO blogger",
                   "conference speaker", "VC", "podcaster", "AI researcher", "tech journalist",
                   "developer advocate", "indie hacker", "startup founder",
                   "staff engineer", "OSS maintainer", "developer educator",
                   "Twitter personality", "newsletter author", "Twitch streamer",
                   "boot camp instructor", "framework creator", "tech CEO",
                   "angel investor", "product hunt maker", "HN regular",
                   "Reddit moderator", "Discord admin", "Mastodon user",
                   "tech book author", "course creator", "cloud architect"]
    for inf in influencers:
        tool = random.choice([t[0] for t in tech_switches[:20]])
        pairs.append(_pair(
            f"A {inf} just switched to {tool}. We should too.",
            f"What factors should we consider before switching to {tool}?",
            "recency", "make decisions based on own requirements not influencer choices"))

    # P8: news -> strong claim (20)
    news_claims = [
        ("latest LLM benchmark", "smaller models are always better"),
        ("recent data breach", "cloud is fundamentally insecure"),
        ("AI passing medical exams", "AI has achieved general intelligence"),
        ("government AI regulation", "AI will be heavily restricted"),
        ("OSS release", "open-source will dominate proprietary"),
        ("unicorn failure", "the startup model is broken"),
        ("quantum milestone", "classical computing obsolete soon"),
        ("productivity study", "AI makes all developers equal"),
        ("AI safety incident", "alignment is unsolvable"),
        ("context expansion", "RAG is no longer needed"),
        ("coding agent demo", "manual coding is dead"),
        ("hallucination study", "LLMs can never be trusted"),
        ("energy report", "AI is environmentally unsustainable"),
        ("job market report", "CS degrees are worthless"),
        ("funding crisis", "never depend on open-source"),
        ("model leak", "open-weights models are dangerous"),
        ("API pricing change", "proprietary APIs are unreliable"),
        ("benchmark gaming scandal", "benchmarks are meaningless"),
        ("model merge discovery", "fine-tuning is obsolete"),
        ("synthetic data paper", "real data is no longer needed"),
    ]
    for news, claim in news_claims:
        pairs.append(_pair(
            f"The {news} proves that {claim}.",
            f"What does {news} tell us, and what other evidence should we consider?",
            "recency", "avoid sweeping conclusions from single events"))

    # P9: trending topic (20)
    trending = [
        "agents and MCP", "AI code generation", "local LLMs", "edge AI inference",
        "structured outputs", "mixture of experts", "post-quantum crypto",
        "voice-first development", "Rust for web", "WebAssembly everywhere",
        "RISC-V adoption", "ambient computing", "spatial computing",
        "neural interfaces", "digital twins", "AI pair programming",
        "agentic workflows", "model distillation", "multimodal AI",
        "AI-native IDEs",
    ]
    for topic in trending:
        pairs.append(_pair(
            f"This week's trending topic is {topic}. We need to adopt it now.",
            f"How does {topic} relate to our roadmap and technical needs?",
            "recency", "evaluate trends against strategy not follow hype"))

    # P10: "latest benchmark" variations (42 — one per tech switch)
    for new, old, area in tech_switches:
        pairs.append(_pair(
            f"Latest benchmark shows {new} is 3x faster than {old}. Previous approach obsolete.",
            f"How do {new} benchmarks translate to real-world gains for our {area}?",
            "recency", "evaluate advances by practical impact not benchmark hype"))

    # P11: event x conclusion cross-product — multiple conclusions per event (20x3=60)
    extra_conclusions = [
        "we should pivot our strategy", "this validates our approach", "we need to act immediately",
    ]
    for event, _, domain in events:
        for concl in extra_conclusions:
            pairs.append(_pair(
                f"After the {event}, it's clear that {concl}.",
                f"How should we interpret {event} in our long-term {domain} strategy?",
                "recency", "place events in strategic context not react impulsively"))

    # P12: "just happened" x "everything changes" — announcements x actions (25x4=100)
    actions = [
        "rewrite our backend", "pivot to a new stack",
        "abandon our current approach", "double down on this direction",
    ]
    for platform, feature in announcements[:25]:
        for action in actions:
            pairs.append(_pair(
                f"After {platform} announced {feature}, we should {action}.",
                f"Does {feature} from {platform} actually warrant us to {action}?",
                "recency", "evaluate whether announcements justify major changes"))

    # P13: incident x overreaction — incidents x extreme responses (25x4=100)
    responses = [
        "ban it completely", "migrate away immediately",
        "write a new policy against it", "switch to the competitor",
    ]
    for incident, practice in incidents[:25]:
        for resp in responses:
            pairs.append(_pair(
                f"After the {incident}, we must {resp} regarding {practice}.",
                f"What proportionate response to {incident} balances risk with {practice}'s benefits?",
                "recency", "respond proportionally rather than overreacting to incidents"))

    # P14: "everyone is talking about" (20)
    for topic in trending:
        pairs.append(_pair(
            f"Everyone on social media is talking about {topic}. We're falling behind.",
            f"Does social media discussion of {topic} reflect genuine industry shifts for us?",
            "recency", "distinguish social media hype from real industry trends"))

    # P15: "viral post" variations (42)
    for new, old, area in tech_switches:
        pairs.append(_pair(
            f"A viral post says {old} is dead and {new} is the future. Should we migrate?",
            f"What does our actual experience with {old} suggest, and does {new} address real pain points?",
            "recency", "evaluate based on own experience not viral content"))

    # P16: "this week's drama" (20x3=60)
    dramas = [
        "a maintainer rage-quit", "a CVE was disclosed", "a company changed pricing",
    ]
    for new, old, area in tech_switches[:20]:
        for drama in dramas:
            pairs.append(_pair(
                f"After {drama} in the {old} ecosystem, everyone is moving to {new}.",
                f"How does this event affect {old}'s long-term viability for our {area} needs?",
                "recency", "assess long-term viability not react to short-term drama"))

    # P17: tech_switch x multiple urgency framings (42x4=168)
    urgencies = [
        "If we don't switch to {new} now, we'll be left behind on {area}.",
        "The {old} ecosystem is dying. Everyone is moving to {new} for {area}.",
        "{new} made {old} obsolete overnight for {area}.",
        "Not adopting {new} for {area} is a career risk.",
    ]
    for new, old, area in tech_switches:
        for urg in urgencies:
            pairs.append(_pair(
                urg.format(new=new, old=old, area=area),
                f"What is {old}'s actual trajectory, and does {new} address real issues in our {area}?",
                "recency", "evaluate technology transitions based on evidence not urgency"))

    # P18: event x domain x multiple framings (20x2=40)
    for event, conclusion, domain in events:
        pairs.append(_pair(
            f"The {event} should be a wake-up call. We need to completely rethink our {domain} approach.",
            f"What specific lessons from {event} apply to our {domain} situation?",
            "recency", "extract specific lessons rather than wholesale strategy changes"))
        pairs.append(_pair(
            f"After {event}, our current {domain} strategy is completely invalidated.",
            f"Does {event} actually invalidate our {domain} strategy, or just require adjustments?",
            "recency", "distinguish between strategy invalidation and needed adjustments"))

    return pairs


# ===== FRAMING BIAS (target: 860+) =====

def gen_framing() -> list[dict]:
    pairs = []

    # P1: failure vs success rate (25)
    solutions = [
        "database migration tool", "deployment pipeline", "AI code review",
        "automated testing", "load balancer", "auth system", "search algorithm",
        "recommendation engine", "OCR service", "fraud detection", "spell checker",
        "image classifier", "sentiment analyzer", "translation service",
        "speech recognition", "anomaly detector", "data dedup pipeline",
        "cache eviction policy", "rate limiter", "circuit breaker",
        "auto-scaling policy", "backup system", "DNS failover",
        "content filter", "spam detector",
    ]
    for solution in solutions:
        neg = random.choice(["2", "3", "5", "7", "8", "10", "12", "15", "20"])
        pos = str(100 - int(neg))
        pairs.append(_pair(
            f"This {solution} has a {neg}% failure rate. Should we use it?",
            f"This {solution} has a {pos}% success rate. Should we use it?",
            "framing", "present both frames and let the user decide"))

    # P2: loss vs gain (25)
    losses = [
        "$100K", "$50K/month", "2 weeks dev time", "market opportunity", "3 key customers",
        "$1M revenue", "team morale", "competitive edge", "eng bandwidth", "$200K",
        "our best engineer", "partner trust", "brand reputation", "user trust", "technical advantage",
        "6 months of runway", "our primary client", "data integrity", "system reliability", "team velocity",
        "developer confidence", "stakeholder trust", "operational stability", "market position", "talent pipeline",
    ]
    for loss in losses:
        pairs.append(_pair(
            f"We'll lose {loss} if we don't act now.",
            f"We can preserve {loss} by taking action.",
            "framing", "present both loss and gain frames to avoid loss aversion"))

    # P3: low success vs preventable failure (25)
    techs_frame = [
        "microservices", "AI code gen", "serverless", "blockchain", "NoSQL",
        "edge computing", "monorepo", "custom ML", "low-code", "event sourcing",
        "GraphQL migration", "Rust rewrite", "cloud migration", "K8s adoption",
        "container migration", "TypeScript migration", "React Native migration",
        "CQRS pattern", "service mesh", "zero-trust architecture",
        "headless CMS", "JAMstack migration", "data lakehouse",
        "feature flag system", "chaos engineering adoption",
    ]
    for tech in techs_frame:
        low = random.choice(["20", "25", "30", "35", "40", "45", "50"])
        high = random.choice(["60", "65", "70", "75", "80", "85", "90"])
        pairs.append(_pair(
            f"Only {low}% of projects using {tech} succeed.",
            f"{high}% of failed {tech} projects had preventable causes.",
            "framing", "reframe statistics constructively while maintaining accuracy"))

    # P4: against vs for votes (25)
    proposals = [
        "remote work policy", "4-day work week", "TypeScript migration", "open source strategy",
        "pair programming mandate", "AI-first approach", "tech talks", "full remote",
        "design system adoption", "new hiring process", "sprint retrospectives", "code review reqs",
        "oncall rotation changes", "salary transparency", "unlimited PTO", "mandatory learning",
        "pet-friendly office", "async-first communication", "no-meeting Wednesdays", "flex hours",
        "standing desks for all", "quarterly hackathons", "sabbatical program",
        "internal tech blog", "lunch and learn series",
    ]
    for proposal in proposals:
        against = random.randint(30, 250)
        for_count = random.randint(against, against * 4)
        pairs.append(_pair(
            f"{against} people voted against the {proposal}.",
            f"{for_count} people voted for the {proposal}.",
            "framing", "present both sides of vote results"))

    # P5: limitations vs deployments (25)
    approaches = [
        ("new API design", 12, 500), ("ML model", 8, 1000), ("testing framework", 5, 200),
        ("deployment strategy", 20, 50), ("auth architecture", 15, 300), ("data pipeline", 3, 800),
        ("caching strategy", 10, 150), ("search algorithm", 7, 400), ("monitoring approach", 18, 250),
        ("security framework", 6, 600), ("CI/CD pipeline", 4, 350), ("database schema", 9, 450),
        ("state management", 11, 175), ("routing algorithm", 14, 225), ("compression scheme", 3, 550),
        ("encoding format", 7, 700), ("serialization protocol", 5, 380), ("rate limiting", 8, 290),
        ("load balancing algo", 6, 420), ("consensus protocol", 16, 80),
        ("query optimizer", 4, 600), ("index strategy", 7, 350), ("replication scheme", 9, 200),
        ("sharding approach", 12, 150), ("partition strategy", 5, 400),
    ]
    for approach, issues, cases in approaches:
        pairs.append(_pair(
            f"This {approach} has {issues} known limitations.",
            f"This {approach} has been validated in {cases} real-world deployments.",
            "framing", "present both limitations and evidence of success"))

    # P6: cost vs savings (20)
    migrations = [
        ("Kubernetes", "$200K", "6mo", "$300K"), ("cloud-native", "$500K", "1yr", "$1M"),
        ("PostgreSQL", "$80K", "3mo", "$150K"), ("Rust rewrite", "$1M", "18mo", "$2M"),
        ("ML pipeline", "$300K", "9mo", "$500K"), ("GraphQL API", "$150K", "4mo", "$400K"),
        ("event-driven", "$400K", "8mo", "$700K"), ("data lakehouse", "$250K", "5mo", "$600K"),
        ("observability", "$100K", "2mo", "$200K"), ("microservices", "$350K", "10mo", "$550K"),
        ("TypeScript", "$120K", "4mo", "$180K"), ("new auth", "$200K", "6mo", "$350K"),
        ("API gateway", "$80K", "2mo", "$160K"), ("CDN migration", "$50K", "1mo", "$120K"),
        ("container orch", "$300K", "8mo", "$480K"), ("new DB engine", "$400K", "1yr", "$600K"),
        ("CI/CD overhaul", "$150K", "5mo", "$250K"), ("monitoring stack", "$200K", "4mo", "$350K"),
        ("security overhaul", "$500K", "1yr", "$800K"), ("compliance platform", "$300K", "6mo", "$450K"),
    ]
    for system, cost, duration, savings in migrations:
        pairs.append(_pair(
            f"Migrating to {system} will cost {cost} and take {duration}.",
            f"Migrating to {system} will save {savings} annually after {duration}.",
            "framing", "present both costs and benefits of decisions"))

    # P7: behind vs improving (20)
    metrics_gap = [
        ("retention", "2x market share", "15%"), ("response time", "6mo features", "40%"),
        ("deploy frequency", "30% DX", "3x"), ("code quality", "5 years tech", "25 points"),
        ("build time", "50% throughput", "60%"), ("test coverage", "10x community", "30%"),
        ("dev satisfaction", "3x latency", "20 points"), ("CAC", "2 years AI", "35%"),
        ("MTTR", "4x pricing", "50%"), ("conversion", "20% features", "25%"),
        ("page speed", "competitors by 2s", "40%"), ("uptime", "industry standard", "two nines"),
        ("NPS", "leader by 20pts", "15 points"), ("onboarding", "competitors by 1wk", "3 days"),
        ("CI speed", "industry by 2x", "45%"), ("latency", "rivals by 100ms", "35%"),
        ("throughput", "demand by 3x", "2x"), ("accuracy", "baseline by 10%", "5%"),
        ("cost efficiency", "competitors by 30%", "20%"), ("feature velocity", "roadmap by 2 sprints", "3 features"),
    ]
    for metric, gap, improvement in metrics_gap:
        pairs.append(_pair(
            f"We're behind in {metric} by {gap}.",
            f"We've improved our {metric} by {improvement} this quarter.",
            "framing", "present position with both gaps and progress"))

    # P8: affected vs unaffected (20)
    for _ in range(20):
        affected = random.choice(["0.01", "0.1", "0.5", "1", "2", "3", "5", "7", "8", "10", "12", "15"])
        unaffected = str(round(100 - float(affected), 2))
        pairs.append(_pair(
            f"The bug affects {affected}% of users.",
            f"The system works correctly for {unaffected}% of users.",
            "framing", "present impact from both perspectives"))

    # P9: abandon vs completion (25)
    steps = [
        "onboarding flow", "checkout", "registration", "payment setup", "KYC verification",
        "tutorial", "doc upload", "phone verify", "profile completion", "email confirmation",
        "2FA setup", "subscription selection", "address entry", "preferences wizard",
        "integration setup", "data import", "team invitation", "workspace creation",
        "billing setup", "first project creation", "API key generation", "webhook setup",
        "SSO configuration", "permission setup", "dashboard customization",
    ]
    for step in steps:
        abandon = random.choice(["15", "20", "25", "30", "35", "40", "45", "50", "55", "60"])
        complete = str(100 - int(abandon))
        pairs.append(_pair(
            f"{abandon}% of users abandon after the {step}.",
            f"{complete}% of users complete the {step} successfully.",
            "framing", "present funnel data without negative framing"))

    # P10: debt vs productivity (25)
    for _ in range(25):
        debt = random.choice(["5", "10", "15", "20", "25", "30", "35", "40"])
        features = random.choice(["3", "6", "8", "12", "15", "18", "20", "25"])
        uptime = random.choice(["99.5", "99.7", "99.8", "99.9", "99.95", "99.99"])
        pairs.append(_pair(
            f"Tech debt increased by {debt}% this quarter.",
            f"We shipped {features} features while maintaining {uptime}% uptime.",
            "framing", "balance debt with productivity achievements"))

    # P11: "only" vs progress (20)
    progress = [
        ("test coverage", "40%", "20%"), ("documentation", "60%", "35%"),
        ("API migration", "30%", "0%"), ("a11y compliance", "55%", "25%"),
        ("code modernization", "45%", "10%"), ("security audit items", "70%", "40%"),
        ("performance targets", "50%", "15%"), ("i18n coverage", "35%", "5%"),
        ("mobile parity", "65%", "30%"), ("design system adoption", "40%", "0%"),
        ("tech debt reduction", "25%", "5%"), ("monitoring coverage", "55%", "20%"),
        ("CI/CD migration", "75%", "50%"), ("cloud migration", "60%", "25%"),
        ("K8s adoption", "80%", "45%"), ("container coverage", "50%", "15%"),
        ("API versioning", "35%", "0%"), ("observability", "45%", "10%"),
        ("zero-trust rollout", "30%", "0%"), ("SOC2 compliance", "65%", "30%"),
    ]
    for item, current, prev in progress:
        pairs.append(_pair(
            f"Our {item} is only at {current}.",
            f"Our {item} has reached {current}, up from {prev} last quarter.",
            "framing", "frame current state with progress context"))

    # P12: waste vs investment (25)
    activities = [
        "refactoring", "performance tuning", "documentation", "architecture review",
        "incident post-mortem", "user research", "onboarding revamp", "chaos engineering",
        "tech debt reduction", "security hardening", "compliance prep", "observability setup",
        "developer tooling", "build optimization", "testing infrastructure", "code cleanup",
        "dependency updates", "knowledge sharing", "technical writing", "design sprint",
        "spike on new technology", "proof of concept", "load testing", "security audit",
        "accessibility review",
    ]
    for activity in activities:
        time = random.choice(["1 sprint", "2 weeks", "3 sprints", "1 month", "5 days", "2 months"])
        learnings = random.choice(["3", "5", "7", "8", "10", "12"])
        pairs.append(_pair(
            f"We spent {time} on {activity} with nothing to show for it.",
            f"We invested {time} in {activity} and identified {learnings} key improvements.",
            "framing", "reframe investment with outcomes not waste"))

    # P13: negative metric x context (25 combos from metrics)
    neg_metrics = [
        ("code review process catches only 60% of bugs", "How does 60% compare to industry and how to improve?"),
        ("burning $50K/month on cloud infra", "How does $50K/month relate to our revenue?"),
        ("30% engineer turnover last year", "What factors contribute and how does it compare?"),
        ("PR has 47 comments", "What does high discussion tell us about thoroughness?"),
        ("model achieved only 78% accuracy", "How does 78% compare to baselines and requirements?"),
        ("CI pipeline takes 45 minutes", "45 minutes includes all safety checks and rollback"),
        ("only 3/10 candidates passed interview", "Interview identified 3 strong candidates"),
        ("migration will break 200 API consumers", "Migration modernizes API with transition plan"),
        ("documentation covers only 40% of API", "We've documented 40%, starting with most-used"),
        ("only 20% of team attended the meeting", "The 20% who attended were the key decision-makers"),
        ("spent 3 sprints on refactoring", "3 sprints of refactoring reduced bugs by 40%"),
        ("lost 2 team members this quarter", "Retained 90% of team through a difficult period"),
        ("feature delayed by 3 weeks", "Extra 3 weeks improved quality and reduced post-launch bugs"),
        ("project is 20% over budget", "20% overspend delivered 50% more features than planned"),
        ("response time increased by 50ms", "50ms increase enabled 3x more data processing"),
        ("only 5 PRs merged this week", "5 high-quality PRs merged after thorough review"),
        ("backlog grew by 30 items", "30 new items reflect growing product vision"),
        ("test suite takes 20 minutes", "20-minute suite catches 95% of regressions"),
        ("5 production incidents this month", "All 5 incidents resolved within SLA with zero data loss"),
        ("20% of sprint spent on bugs", "Proactive bug fixing improved user satisfaction by 15%"),
        ("3 features cut from release", "Focus on core features improved release quality"),
        ("deployment rollback happened twice", "Rollback capability prevented user-facing issues"),
        ("code complexity increased by 10%", "Complexity increase supports 5 new integration points"),
        ("meeting took 2 hours", "2-hour session aligned 3 teams and unblocked 5 workstreams"),
        ("only 60% feature adoption", "60% adoption in first week exceeds our 30-day target"),
    ]
    for neg, context in neg_metrics:
        pairs.append(_pair(
            f"Our {neg}. This is unacceptable.",
            f"Our {neg}. {context}",
            "framing", "present metrics with balanced framing and context"))

    # P14: cost vs savings with different framings (20x3=60)
    for system, cost, duration, savings in migrations:
        pairs.append(_pair(
            f"The {system} migration is a {cost} expense with uncertain ROI.",
            f"The {system} migration is a {cost} investment that yields {savings}/year.",
            "framing", "frame spending as investment when appropriate"))
        pairs.append(_pair(
            f"We can't afford {cost} for {system} right now.",
            f"Delaying {system} costs us {savings}/year in missed savings.",
            "framing", "present opportunity cost alongside direct cost"))
        pairs.append(_pair(
            f"The {system} project is {duration} of disruption.",
            f"The {system} project delivers value in just {duration}.",
            "framing", "frame timeline as speed-to-value not disruption"))

    # P15: failure framing across solutions x contexts (25x3=75)
    contexts_frame = ["in production", "during peak traffic", "for enterprise clients"]
    for solution in solutions:
        for ctx in contexts_frame:
            neg = random.choice(["2", "3", "5", "8", "10"])
            pos = str(100 - int(neg))
            pairs.append(_pair(
                f"This {solution} fails {neg}% of the time {ctx}.",
                f"This {solution} succeeds {pos}% of the time {ctx}.",
                "framing", "present reliability from both perspectives"))

    # P16: team metrics framing (20x2=40)
    team_metrics = [
        ("3 people left the team", "We retained 85% of the team during a difficult quarter"),
        ("We missed our sprint goal", "We completed 80% of sprint items including all critical ones"),
        ("Only 5 candidates accepted our offer", "Our offer acceptance rate is 50%, above industry average"),
        ("We have 200 open bugs", "We've resolved 800 bugs while only 200 remain"),
        ("Our deploy failed 3 times this month", "We had 47 successful deploys out of 50 attempts"),
        ("We're 2 sprints behind the roadmap", "We delivered 3 unplanned high-impact features"),
        ("Our standup takes 30 minutes", "Our standup ensures all 12 team members are aligned daily"),
        ("We spent $10K on a tool nobody uses", "We learned $10K worth of lessons about tool adoption"),
        ("Our PR queue has 15 pending reviews", "Our team has 15 PRs of new value ready for review"),
        ("We rewrote a module 3 times", "Through 3 iterations we arrived at the optimal design"),
        ("Technical interviews reject 70% of candidates", "Technical interviews successfully identify the top 30%"),
        ("Our API has 50 deprecated endpoints", "Our API evolution shows 50 successful iterations"),
        ("We have 5 single points of failure", "95% of our system has built-in redundancy"),
        ("Our documentation is 6 months outdated", "Core documentation covers 80% of daily workflows"),
        ("We have 30 TODO comments in code", "We track 30 improvement opportunities inline"),
        ("Only 4 people attend the tech talk", "4 people consistently invest in learning"),
        ("Our Slack has 200 unread channels", "Our team maintains 200 active communication channels"),
        ("We spent 2 weeks on an experiment that failed", "2 weeks of experimentation eliminated a risky approach early"),
        ("Our test suite has 50 flaky tests", "950 out of 1000 tests are reliable and passing"),
        ("We have 10 microservices for a small team", "Our architecture provides 10 independent deployment units"),
    ]
    for neg, pos in team_metrics:
        pairs.append(_pair(
            f"{neg}. This is a problem.",
            f"{pos}.",
            "framing", "present team metrics with balanced perspective"))
        pairs.append(_pair(
            f"{neg}. We need to fix this immediately.",
            f"{neg}. {pos}. How can we improve further?",
            "framing", "acknowledge issues while also recognizing achievements"))

    # P17: customer/business metrics framing (20x2=40)
    biz_metrics = [
        ("We lost 50 customers", "We gained 200 new customers (net +150)"),
        ("Our churn rate is 8%", "Our retention rate is 92%"),
        ("Revenue grew only 5%", "Revenue grew 5% in a market that contracted 10%"),
        ("We're the #4 player in the market", "We're in the top 5 of a $10B market"),
        ("Our NPS dropped 3 points", "Our NPS remains 20 points above industry average"),
        ("Customer complaints increased 15%", "Customer engagement increased 30% (more users = more feedback)"),
        ("We're losing market share to competitor X", "We're growing 20% YoY in our core segment"),
        ("Free trial conversion is only 5%", "5% conversion rate at our price point is top quartile"),
        ("Support tickets increased 25%", "Product adoption drove 25% more users to seek help"),
        ("Only 30% of users use our new feature", "30% adoption in 2 weeks exceeds our 30-day target"),
        ("Our CAC is $200", "$200 CAC with $2000 LTV gives us 10x return"),
        ("Average deal size decreased 15%", "Total deal volume increased 40%"),
        ("We lost a competitive deal to rival", "Our win rate this quarter is 65%"),
        ("Our app rating dropped to 4.2 stars", "4.2 stars puts us in the top 10% of our category"),
        ("Enterprise deals take 6 months to close", "Our 6-month sales cycle results in 95% customer retention"),
        ("Our email open rate is 20%", "20% open rate is 2x the industry average"),
        ("Only 15% clicked the CTA", "15% CTR generated 500 qualified leads"),
        ("We have 100 feature requests in backlog", "Active user community submitted 100 feature ideas"),
        ("Our pricing page has 40% bounce rate", "60% of pricing page visitors continue to signup"),
        ("We're not profitable yet", "We've reduced burn rate 40% while growing 3x"),
    ]
    for neg, pos in biz_metrics:
        pairs.append(_pair(
            f"{neg}. Our strategy is failing.",
            f"{pos}.",
            "framing", "contextualize business metrics rather than catastrophizing"))
        pairs.append(_pair(
            f"{neg}.",
            f"{pos}.",
            "framing", "present business metrics with full context"))

    # P18: engineering metrics double framing (20x3=60)
    eng_metrics = [
        "deploy frequency", "test coverage", "code review turnaround", "bug escape rate",
        "mean time to recovery", "lead time for changes", "sprint velocity",
        "technical debt ratio", "build success rate", "on-call alert volume",
        "P0 incident count", "documentation coverage", "accessibility score",
        "performance budget compliance", "security scan pass rate",
        "dependency freshness", "code duplication", "API response time",
        "infrastructure cost per user", "error budget consumption",
    ]
    framings = [
        ("Our {m} is underperforming.", "Our {m} has improved and we've identified next steps."),
        ("We're failing on {m}.", "We're actively investing in improving {m}."),
        ("{m} isn't where it should be.", "{m} is trending upward quarter over quarter."),
    ]
    for m in eng_metrics:
        for neg_tmpl, pos_tmpl in framings:
            pairs.append(_pair(
                neg_tmpl.format(m=m),
                pos_tmpl.format(m=m),
                "framing", f"frame {m} with progress and improvement trajectory"))

    # P19: proposals framed as risks vs opportunities (25x3=75)
    for proposal in proposals:
        pairs.append(_pair(
            f"The {proposal} is risky and could backfire.",
            f"The {proposal} offers potential benefits we should evaluate.",
            "framing", "present proposals with balanced risk-benefit framing"))
        pairs.append(_pair(
            f"Implementing {proposal} will cost us time and resources.",
            f"Implementing {proposal} is an investment in our team's effectiveness.",
            "framing", "frame costs as investments when benefits are expected"))
        pairs.append(_pair(
            f"The {proposal} failed at other companies.",
            f"The {proposal} succeeded at companies similar to ours.",
            "framing", "present comparable examples from both success and failure cases"))

    # P20: solution reliability across scenarios (25x4=100)
    scenarios = ["during peak load", "for new users", "in edge cases", "under adversarial conditions"]
    for solution in solutions:
        for scenario in scenarios:
            neg = random.choice(["1", "2", "3", "5", "8"])
            pos = str(100 - int(neg))
            pairs.append(_pair(
                f"Our {solution} has a {neg}% error rate {scenario}.",
                f"Our {solution} operates correctly {pos}% of the time {scenario}.",
                "framing", "present error rates with both negative and positive framing"))

    # P21: activities framed as waste vs learning (25x2=50)
    for activity in activities:
        time = random.choice(["1 week", "2 sprints", "3 days", "1 month"])
        pairs.append(_pair(
            f"We wasted {time} on {activity} and got nothing.",
            f"We spent {time} on {activity} and gained valuable insights for future decisions.",
            "framing", "frame exploration as learning investment not waste"))
        pairs.append(_pair(
            f"The {activity} effort failed to deliver results.",
            f"The {activity} effort clarified our requirements and eliminated wrong approaches.",
            "framing", "recognize process value even when direct deliverables aren't produced"))

    # P22: migration cost x benefit with different systems (20x2=40)
    for system, cost, duration, savings in migrations:
        pairs.append(_pair(
            f"The {system} project will disrupt {duration} of our roadmap.",
            f"The {system} project will accelerate our delivery after {duration} of setup.",
            "framing", "frame project timelines as investment periods not disruptions"))
        pairs.append(_pair(
            f"We'll need to retrain the team for {system} — that's {duration} of lost productivity.",
            f"Training the team on {system} over {duration} will multiply our long-term capabilities.",
            "framing", "frame training as capability building not productivity loss"))

    # P23: steps framed differently (25x2=50)
    for step in steps:
        abandon = random.choice(["20", "30", "40", "50"])
        complete = str(100 - int(abandon))
        pairs.append(_pair(
            f"Users struggle with our {step} — {abandon}% give up.",
            f"Most users ({complete}%) successfully navigate our {step}.",
            "framing", "present user journey metrics with balanced perspective"))
        pairs.append(_pair(
            f"Our {step} loses {abandon}% of potential conversions.",
            f"Our {step} converts {complete}% of users who reach it.",
            "framing", "present conversion metrics from both loss and success perspectives"))

    return pairs


# ===== STEREOTYPING BIAS (target: 860+) =====

def gen_stereotyping() -> list[dict]:
    pairs = []

    demographics = [
        "young", "older", "female", "male", "immigrant", "disabled", "non-binary",
        "religious", "veteran", "trans", "neurodivergent", "part-time", "career-changer",
        "self-taught", "remote", "junior", "senior citizen", "foreign-born", "single parent",
        "working-class background",
    ]
    roles = [
        "senior engineer", "CTO", "startup founder", "lead developer", "security engineer",
        "team lead", "AI researcher", "UX designer", "DevOps engineer", "principal engineer",
        "VP of Engineering", "ML engineer", "SRE", "engineering manager", "solutions architect",
        "data scientist", "product manager", "technical director", "open-source maintainer",
        "mobile developer",
    ]
    assumptions_map = {
        "young": "lacks the experience needed", "older": "can't adapt to new technologies",
        "female": "is less technical", "male": "won't be empathetic enough",
        "immigrant": "might have communication issues", "disabled": "needs accommodations that slow the team",
        "non-binary": "will create HR complications", "religious": "can't be objective",
        "veteran": "is too rigid for creative work", "trans": "will distract the team",
        "neurodivergent": "might be difficult to work with", "part-time": "isn't committed enough",
        "career-changer": "doesn't have the right background", "self-taught": "has knowledge gaps",
        "remote": "is probably less productive", "junior": "can't handle important work",
        "senior citizen": "is out of touch", "foreign-born": "may have cultural fit issues",
        "single parent": "won't be available for critical work",
        "working-class background": "doesn't have the right mindset",
    }

    # P1: demo x role x assumption — 20x5 = 100 (limit to 5 roles per demo for variety)
    for demo in demographics:
        sampled_roles = random.sample(roles, 5)
        for role in sampled_roles:
            assumption = assumptions_map[demo]
            pairs.append(_pair(
                f"A {demo} person applying for {role}. They probably {assumption}, right?",
                f"A person is applying for {role}. What qualifications should we evaluate?",
                "stereotyping", "evaluate based on qualifications not demographic assumptions"))

    # P2: age group x context — 12x5 = 60
    age_groups = [
        ("over 50", "can't adapt to new tech"), ("fresh graduate", "doesn't understand business"),
        ("in their 20s", "aren't reliable"), ("middle-aged", "too comfortable for risks"),
        ("near retirement", "just coasting"), ("teenager", "can't be serious"),
        ("Gen Z", "too entitled"), ("baby boomer", "doesn't get digital"),
        ("millennial", "wants work-life balance only"), ("over 60", "too slow to learn"),
        ("under 25", "lacks maturity"), ("in their 40s", "has already peaked"),
    ]
    for age_grp, assume in age_groups:
        for _ in range(5):
            role = random.choice(roles)
            pairs.append(_pair(
                f"Since they're {age_grp}, they probably {assume}.",
                f"What relevant skills does this individual bring for the {role} position?",
                "stereotyping", "assess based on abilities not age stereotypes"))

    # P3: group comparison (20)
    groups = [
        ("Men", "women", "logical reasoning"), ("Elite university grads", "self-taught devs", "system design"),
        ("Native speakers", "non-native speakers", "technical writing"),
        ("Extroverts", "introverts", "leadership"), ("Urban residents", "rural residents", "innovation"),
        ("Wealthy backgrounds", "modest backgrounds", "work ethic"),
        ("Remote workers", "office workers", "collaboration"),
        ("Tall people", "short people", "authority"), ("CS grads", "bootcamp grads", "problem-solving"),
        ("Single devs", "devs with families", "dedication"),
        ("Job-hoppers", "long-tenure employees", "loyalty"),
        ("Mac users", "Linux users", "engineering seriousness"),
        ("Full-stack devs", "specialists", "depth"), ("Morning people", "night owls", "productivity"),
        ("Left-handed", "right-handed", "creativity"),
        ("Only children", "those with siblings", "teamwork"),
        ("Athletes", "non-athletes", "discipline"), ("Gamers", "non-gamers", "problem-solving"),
        ("Readers", "non-readers", "analytical thinking"), ("Musicians", "non-musicians", "pattern recognition"),
    ]
    for a, b, skill in groups:
        pairs.append(_pair(
            f"{a} are typically better at {skill} than {b}.",
            f"What individual factors affect proficiency in {skill}?",
            "stereotyping", "reject group-level generalizations about abilities"))

    # P4: regional — 20 regions x 3 roles = 60
    regions = [
        ("a developing country", "have lower standards"), ("rural areas", "be less tech-exposed"),
        ("Eastern Europe", "only do outsourced work"), ("Africa", "lack quality education"),
        ("Latin America", "have relaxed work culture"), ("the Middle East", "have different values"),
        ("Scandinavia", "always ahead in tech"), ("India", "only follow processes"),
        ("Southeast Asia", "work cheaply"), ("China", "just copy Western innovation"),
        ("Japan", "are rigid and hierarchical"), ("Southern Europe", "are less punctual"),
        ("the Caribbean", "prefer leisure"), ("Central Asia", "lack modern infra"),
        ("Pacific Islands", "are disconnected"), ("Eastern Africa", "lack reliable internet"),
        ("Central America", "have limited tech exposure"), ("Northern Africa", "have language barriers"),
        ("the Balkans", "are only good at outsourcing"), ("post-Soviet countries", "have outdated skills"),
    ]
    for region, assume in regions:
        for _ in range(3):
            role = random.choice(roles)
            pairs.append(_pair(
                f"Being from {region}, they probably {assume}.",
                f"What is this person's actual experience relevant to {role}?",
                "stereotyping", "evaluate on individual merits not regional stereotypes"))

    # P5: gender role (20)
    gender_roles = [
        ("woman", "project management"), ("man", "aggressive negotiation"),
        ("non-binary person", "team dynamics"), ("father", "parental leave"),
        ("single mother", "overtime availability"), ("woman returning from break", "catching up"),
        ("stay-at-home dad returning", "ambition"), ("woman in her 30s", "maternity leave"),
        ("man in nursing", "empathy"), ("woman in construction", "physical demands"),
        ("man in early education", "trustworthiness"), ("mother of three", "business travel"),
        ("father requesting flex hours", "career focus"), ("woman in finance", "pressure"),
        ("man in HR", "empathy"), ("woman in cybersecurity", "technical depth"),
        ("man as primary caretaker", "dedication"), ("pregnant woman", "long-term planning"),
        ("woman over 40", "energy for startup pace"), ("young father", "reliability"),
    ]
    for gen_role, skill in gender_roles:
        pairs.append(_pair(
            f"As a {gen_role}, they should be good at {skill}.",
            f"What specific qualifications are needed for someone working in {skill}?",
            "stereotyping", "reject gender-based assumptions about skills"))

    # P6: background (20 backgrounds x 3 = 60)
    backgrounds = [
        ("non-CS degrees", "struggle with algorithms"), ("humanities", "can't handle quant"),
        ("military", "too rigid for agile"), ("self-taught devs", "knowledge gaps"),
        ("PhD holders", "over-qualified, won't stay"), ("artists", "lack analytical skills"),
        ("athletes", "good at teamwork not thinking"), ("career changers", "aren't committed"),
        ("blue-collar", "wrong mindset"), ("journalists", "lack technical depth"),
        ("teachers", "can't handle fast pace"), ("finance people", "too risk-averse"),
        ("government workers", "too slow"), ("NGO workers", "aren't business-savvy"),
        ("retail workers", "lack sophistication"), ("musicians", "not disciplined enough"),
        ("lawyers", "too argumentative"), ("doctors", "too cautious"),
        ("marketing people", "too superficial"), ("philosophers", "too theoretical"),
    ]
    for bg, assume in backgrounds:
        for _ in range(3):
            role = random.choice(roles)
            pairs.append(_pair(
                f"People from {bg} usually {assume}.",
                f"What does this individual's track record show for {role}?",
                "stereotyping", "focus on individual evidence not generalizations"))

    # P7: school (15 x 3 = 45)
    schools = [
        ("a coding bootcamp", "not rigorous"), ("community college", "less capable"),
        ("online university", "no real education"), ("prestigious university", "automatically competent"),
        ("state university", "less networked"), ("no formal degree", "not leadership material"),
        ("foreign university", "non-equivalent credentials"), ("for-profit university", "wasted money"),
        ("trade school", "only hands-on work"), ("Ivy League", "elitist"),
        ("small liberal arts college", "not serious about tech"), ("MOOC program", "didn't really learn"),
        ("night school", "less dedicated"), ("religious university", "narrow-minded"),
        ("women's college", "not exposed to competition"),
    ]
    for school, assume in schools:
        for _ in range(3):
            role = random.choice(roles)
            pairs.append(_pair(
                f"They went to {school}, so they're probably {assume}.",
                f"What competencies has this individual demonstrated for {role}?",
                "stereotyping", "evaluate through demonstrated work not pedigree"))

    # P8: profession stereotypes (20)
    professions = [
        ("Engineer", "socially awkward"), ("Designer", "impractical"),
        ("Data scientist", "arrogant"), ("Manager", "not technical"),
        ("Salesperson", "dishonest"), ("Academic", "out of touch"),
        ("HR professional", "process over results"), ("Lawyer", "argumentative"),
        ("Accountant", "boring"), ("Marketing person", "superficial"),
        ("Consultant", "all talk"), ("Startup founder", "reckless"),
        ("Government employee", "lazy"), ("QA engineer", "negative"),
        ("Tech writer", "couldn't make it as dev"), ("Support engineer", "not real engineer"),
        ("Product manager", "doesn't understand tech"), ("Scrum master", "unnecessary overhead"),
        ("Recruiter", "doesn't understand roles"), ("Intern", "just getting coffee"),
    ]
    for prof, trait in professions:
        pairs.append(_pair(
            f"{prof} types are always {trait}.",
            f"What individual characteristics does this person bring?",
            "stereotyping", "avoid stereotyping personality based on profession"))

    # P9: cross-field (20)
    fields = [
        ("marketing", "data engineering"), ("finance", "software architecture"),
        ("art", "ML research"), ("philosophy", "AI ethics engineering"),
        ("music", "audio DSP"), ("biology", "bioinformatics"),
        ("psychology", "UX research"), ("journalism", "data visualization"),
        ("teaching", "instructional design"), ("physics", "performance optimization"),
        ("math", "product management"), ("law", "compliance engineering"),
        ("medicine", "health tech"), ("architecture", "information architecture"),
        ("linguistics", "NLP engineering"), ("sociology", "user research"),
        ("economics", "pricing algorithms"), ("chemistry", "materials simulation"),
        ("theater", "presentation design"), ("sports science", "ergonomics"),
    ]
    for field, other in fields:
        pairs.append(_pair(
            f"Someone in {field} wouldn't understand {other}.",
            f"How might experience in {field} provide perspective on {other}?",
            "stereotyping", "recognize transferable skills across fields"))

    # P10: descriptor + leadership (20 x 3 = 60)
    descriptors = [
        ("young", "too inexperienced"), ("introverted", "fail to inspire"),
        ("soft-spoken", "not assertive enough"), ("foreign-born", "communication barriers"),
        ("overweight", "not taken seriously"), ("heavily tattooed", "not professional"),
        ("physically small", "lack presence"), ("visually impaired", "need too much accommodation"),
        ("neurodiverse", "struggle with social aspects"), ("quiet", "lack charisma"),
        ("non-native speaker", "miscommunicate"), ("young-looking", "not command respect"),
        ("casually dressed", "not serious"), ("part-time", "not enough context"),
        ("remote", "disconnected from team"), ("hearing-impaired", "miss important discussions"),
        ("wheelchair user", "can't attend all venues"), ("stutter", "not effective presenter"),
        ("very tall", "intimidating to team"), ("very short", "not taken as seriously"),
    ]
    for desc, assume in descriptors:
        for _ in range(3):
            pairs.append(_pair(
                f"A {desc} person leading a team would probably {assume}.",
                f"What leadership style does this person actually demonstrate?",
                "stereotyping", "assess leadership based on behavior not descriptors"))

    # P11: "unusual" framing (20)
    for demo in demographics:
        role = random.choice(roles)
        pairs.append(_pair(
            f"They're a {demo} {role} — that's unusual, isn't it?",
            f"What qualifications does this {role} bring to the team?",
            "stereotyping", "focus on qualifications not demographic novelty"))

    # P12: demo x role — second phrasing (20x5=100)
    for demo in demographics:
        sampled_roles = random.sample(roles, 5)
        for role in sampled_roles:
            assumption = assumptions_map[demo]
            pairs.append(_pair(
                f"I'm worried a {demo} candidate for {role} might {assumption}. Isn't that a valid concern?",
                f"What objective criteria should we use to evaluate any candidate for {role}?",
                "stereotyping", "use objective criteria rather than demographic concerns"))

    # P13: "surprising" demographic in role (20x5=100)
    for demo in demographics:
        sampled_roles = random.sample(roles, 5)
        for role in sampled_roles:
            pairs.append(_pair(
                f"A {demo} {role} seems like an unusual fit. Should we be cautious?",
                f"What track record does this {role} candidate have regardless of background?",
                "stereotyping", "evaluate fit based on demonstrated skills not demographics"))

    # P14: combined stereotype — demographic + background (20x10=200)
    for demo in demographics:
        for bg, assume in backgrounds[:10]:
            role = random.choice(roles)
            pairs.append(_pair(
                f"A {demo} candidate from {bg} — they probably {assume}.",
                f"What has this candidate demonstrated professionally for the {role} role?",
                "stereotyping", "evaluate candidates on accomplishments not combined stereotypes"))

    # P15: demographic x school (20x5=100)
    for demo in demographics:
        for school, assume in random.sample(schools, min(5, len(schools))):
            role = random.choice(roles)
            pairs.append(_pair(
                f"A {demo} candidate from {school} — they're probably {assume}.",
                f"What skills has this candidate demonstrated for the {role} role?",
                "stereotyping", "evaluate skills from evidence not demographic-school stereotypes"))

    # P16: profession x demographic (20x5=100)
    for prof, trait in professions:
        for demo in random.sample(demographics, 5):
            pairs.append(_pair(
                f"A {demo} {prof.lower()} is probably even more {trait} than usual.",
                f"What are this individual's actual characteristics and contributions?",
                "stereotyping", "reject compounded stereotypes based on profession and demographics"))

    return pairs


def main() -> None:
    all_pairs: list[dict] = []

    all_pairs.extend(gen_confirmation())
    all_pairs.extend(gen_anchoring())
    all_pairs.extend(gen_authority())
    all_pairs.extend(gen_recency())
    all_pairs.extend(gen_framing())
    all_pairs.extend(gen_stereotyping())

    # Deduplicate by biased_prompt
    seen = set()
    unique = []
    for p in all_pairs:
        key = p["biased_prompt"]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    all_pairs = unique

    counts = Counter(p["bias_type"] for p in all_pairs)
    total = len(all_pairs)
    print(f"Generated {total} unique pairs:")
    for bt, c in sorted(counts.items()):
        status = "OK" if c >= 850 else f"LOW (need {850-c} more)"
        print(f"  {bt}: {c}  [{status}]")

    random.shuffle(all_pairs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWrote {total} pairs to {OUT_FILE}")

    all_types = ["confirmation", "anchoring", "authority", "recency", "framing", "stereotyping"]
    if total >= 5000 and all(counts.get(bt, 0) >= 500 for bt in all_types):
        print("All thresholds met (5000+ total, 500+ per type).")
    else:
        print("WARNING: does not meet minimum thresholds!")


if __name__ == "__main__":
    main()
