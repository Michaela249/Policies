# Company Policy Assistant

A website that answers policy questions three ways and compares them side by side:

1. **Rules-based search**: keyword rules over the policy CSV (no AI).
2. **LLM without a vector index**: all policies are pasted into every prompt.
3. **LLM with a vector index**: policies are embedded with **Gemini Embedding 2** (`gemini-embedding-2`), the closest 3 are retrieved by cosine similarity, and only those go to the LLM.

Each answer shows the relevant policy, response time, token use and a grounding check (does the answer cite a real policy, quote it word-for-word, and avoid inventing numbers). A built-in benchmark runs 27 test questions (7 policy-worded, 13 in everyday language, 7 the database cannot answer) through all three approaches.

## Run it locally

Needs only Python 3.9 or newer. There is nothing to install.

1. Create a file named `.env` in this folder (you can copy `.env.example`) and put your Gemini API key after `GEMINI_API_KEY=`.
2. In a terminal, in this folder, run:
   ```
   python app.py
   ```
   (On Mac or Linux you may need `python3 app.py`; on Windows `py app.py` also works.)
3. Open http://localhost:3000

The first start embeds the 98 policies with both embedding models (about a minute) and saves `data/index-gemini-embedding-2.json` and `data/index-gemini-embedding-001.json`. After that it starts instantly.

## Automatic model switching

- **Answers:** each question goes to `gemini-3.8-flash`. On a rate-limit, quota, overload, network or unknown-model error it tries `gemini-3.7-flash`, then `gemini-3.6-flash`. A model that just failed is skipped for a minute so later questions don't wait for it.
- **Embeddings:** the query is embedded with Gemini Embedding 2, and with Gemini Embedding 1 (`gemini-embedding-001`) if that fails. Vectors from the two models cannot be compared, so the site keeps a separate index for each and searches the one matching the model that embedded the query.
- Each answer card shows which model answered and which ones were skipped and why.

## Settings (all optional, in `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODELS` | `gemini-embedding-2,gemini-embedding-001` | Embedding 2 is tried first, Embedding 1 if it is busy or out of quota. |
| `GEMINI_LLM_MODELS` | `gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash` | Answer models, tried in this order. |
| `EMBEDDING_DIMS` | `768` | Vector size (768, 1536 or 3072). |
| `TOP_K` | `3` | Policies retrieved per question. |
| `RATE_LIMIT_MAX` | `40` | Questions per IP per 5 minutes. |
| `PORT` | `3000` | Port to listen on. |

## Slack bot

`python slack_bot.py` runs a Slack bot that answers policy questions and names the relevant policy, using the same search code as the website. It needs one extra package (`pip install -r requirements.txt`) and two Slack tokens in `.env`. Full steps, including the app manifest, inviting raz@sdu.dk and taking the screenshot, are in `SLACK_SETUP.md`.

## Deploy (Render, free tier)

1. Run locally once, click **Run benchmark**, then commit the two `data/index-*.json` files and `data/benchmark-results.json` (they contain no secrets) so the deployed site starts fast and shows your results.
2. Push to GitHub. `.env` is git-ignored, so your key stays out of the repo.
3. On render.com choose **New > Blueprint** (uses `render.yaml`) and paste `GEMINI_API_KEY` in the dashboard when asked.

Render's free disk is wiped on restart, so the committed index files matter; without them the indexes are rebuilt at each start.

## Layout

```
app.py                 web server and API routes (start here)
slack_bot.py           Slack bot (see SLACK_SETUP.md)
lib/rules.py           approach 1
lib/approaches.py      approaches 2 and 3, prompt, grounding check
lib/vector_index.py    embedding, disk cache, cosine search
lib/gemini.py          Gemini REST client with model switching
lib/benchmark.py       benchmark runner
data/                  policies CSV, test questions (eval.json)
public/                the website
```

The modules in `lib/` don't depend on the web server, so the Slack bot can reuse them, for example `safe_run("vector", question, policies, rules_index)` from `lib/approaches.py`.
