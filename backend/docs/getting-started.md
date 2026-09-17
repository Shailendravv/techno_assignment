# Getting Started

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt

cp .env.example .env            # then add your GROQ_API_KEY
```

A free Groq key takes about a minute to get at
[console.groq.com/keys](https://console.groq.com/keys){:target="_blank"}. No
card, no paid tier.

## Ask it something

```bash
# Ask a question
python -m agent "checkout-api is running hot on CPU - what should I check first?"

# Watch it decline, and see why
python -m agent "What is our refund policy for orders over $500?" --trace
```

`--trace` prints what each stage of the pipeline decided — which documents
survived the metadata filter, whether the gate passed, and what the model
cited. It is the fastest way to see *why* an answer came out the way it did.

## Run the evaluation harness

This is the scored deliverable — it calls `answer_question()` directly, so
there is no server involved and the thing being measured is the thing that
gets deployed.

```bash
python -m eval.harness --arm lexical --out harness_output.json

# Compare against the no-retrieval control
python -m eval.harness --compare lexical,baseline --delay 8
```

`--delay` paces requests to stay inside Groq's free-tier rate limit (roughly
8,000 tokens/minute). The pipeline arm costs about 1,500 tokens per question;
the baseline arm — which stuffs all twelve documents into one prompt — costs
about 6,000.

## Run the HTTP API

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How do I safely roll back checkout-api?"}'
```

## What runs without a key

**Most of the system needs no API key.** Retrieval, the metadata filter, the
gate, and the entire test suite run offline:

```bash
pytest -q                       # 261 tests, no network, no key
python -m eval.harness --sweep  # gate calibration, retrieval only
```

Only the grounding step (the actual LLM call that writes the answer) needs
`GROQ_API_KEY`. If it's missing, the CLI and API print what to do instead of a
traceback, and every path that can decline without a model still works.

## Build and view these docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000){:target="_blank"} in
a browser. `mkdocs build` produces a static `site/` directory that can be
hosted anywhere.
