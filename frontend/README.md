# Runbook Agent — web UI

The React client for the agent in [`backend/`](../backend). One page, one input
box: ask a question about the twelve operational runbooks and read the answer
**next to the evidence for it** — which documents it cited, how confident it is,
and the per-stage trace of how it got there.

The interesting case is the one where the agent declines. A `no_match` comes
back as a normal, successful answer, and this UI renders it as one — with the
trace still expandable, because for a refusal the trace is the only thing that
says *why*. Showing that as an error would be the wrong reading of a correct
result.

> Run both halves together from the repo root with `./start.sh`. It installs
> `node_modules` if they are missing or stale and serves this app on `:5173`
> against the API on `:8000`. Everything below is the manual path.

---

## Quick start

```bash
npm install
npm run dev             # http://localhost:5173
```

The app needs the backend running on `:8000` to answer anything:

```bash
cd ../backend && uvicorn app.main:app --reload
```

The backend must allow this origin — `CORS_ORIGINS` defaults to
`http://localhost:5173,http://127.0.0.1:5173`, so the default pairing works
unchanged. See [`backend/.env.example`](../backend/.env.example).

### Scripts

| Command | What it does |
| :--- | :--- |
| `npm run dev` | Vite dev server with HMR |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the built bundle locally |
| `npm run lint` | ESLint |
| `npm run test:e2e` | **18 Playwright tests, `POST /ask` stubbed.** No backend, no API key. This is what CI runs |
| `npm run test:e2e:live` | 3 `@live` tests through the real pipeline. Needs the API up, and spends Groq quota |
| `npm run test:e2e:ui` | The Playwright UI runner |

### Environment

```env
VITE_API_URL=http://localhost:8000
```

The only variable. `src/services/api.js` falls back to `http://localhost:8000`
if it is unset, so a local checkout works with no `.env` at all. Vite only
exposes variables prefixed `VITE_`.

---

## Stack

| Layer | Choice |
| :--- | :--- |
| UI | [React 19](https://react.dev/) |
| Build | [Vite 8](https://vitejs.dev/) |
| Styling | [Tailwind CSS 4](https://tailwindcss.com/) via `@tailwindcss/vite` — no PostCSS config file |
| Server state | [TanStack Query v5](https://tanstack.com/query/latest) (`useMutation`) |
| Routing | [React Router 7](https://reactrouter.com/) — one route, in place for a second |
| HTTP | [Axios](https://axios-http.com/), one configured instance |
| Markdown | [react-markdown](https://github.com/remarkjs/react-markdown) + [remark-gfm](https://github.com/remarkjs/remark-gfm) |
| Icons | [React Icons](https://react-icons.github.io/react-icons/) (IonIcons) |
| E2E | [Playwright](https://playwright.dev/), Chromium |

---

## Layout

```bash
src/
 ├── main.jsx                  entry; QueryClientProvider + StrictMode
 ├── App.jsx                   Router -> MainLayout -> "/" -> Home
 ├── index.css                 Tailwind entry and global styles
 ├── layouts/MainLayout.jsx    global wrapper
 ├── pages/Home.jsx            the single route
 ├── components/
 │    ├── ChatInterface.jsx    all the state: messages, the mutation, rendering
 │    ├── Header.jsx           sticky, backdrop-blurred title bar
 │    └── InputBar.jsx         fixed composer; presentational, fully controlled
 ├── services/api.js           axios instance + askQuestion()
 └── assets/
e2e/
 ├── chat.spec.js              18 hermetic tests, API stubbed
 ├── fixtures.js               stub responses copied from real /ask output
 └── live.spec.js              3 @live tests against a running backend
```

State lives in `ChatInterface`. `InputBar` and `Header` take props and own
nothing — the message list has a single source of truth, and the composer is
controlled from the same place the mutation is.

Two things in there are deliberate and easy to undo by accident:

- **Message ids come from `crypto.randomUUID()`**, not `Date.now()`. Two
  messages landing in the same millisecond collide, and React silently drops one
  of the duplicate keys from the list.
- **Every `setMessages` is a functional update.** Reading `messages` from the
  render closure loses a message when two sends land before React re-renders.
  There is an e2e test pinning this.

---

## Talking to the API

`POST /ask`, JSON in, JSON out — `src/services/api.js`.

| Field | Type | Notes |
| :--- | :--- | :--- |
| `question` | String | 1–2000 characters |
| `explain` | Boolean | This UI always sends `true` |
| `model_role` | String | `"generator"` (default) or `"reasoner"`; not currently sent |

```json
{
  "answer": "Check the following, in order: ...",
  "cited_doc_ids": ["RB-001"],
  "confidence": "high",
  "elapsed_ms": 812,
  "trace": ["analyze: service=checkout-api ...", "gate: pass ..."],
  "llm_calls": 3
}
```

`confidence` is one of `high | medium | low | no_match`, rendered as a coloured
pill beside the cited document ids. `trace` goes into a collapsed **"Why this
answer"** disclosure along with the model-call count — which is `0` for a
question the gate stopped, and worth seeing.

This mirrors `backend/app/schemas.py::AskResponse`. If that contract changes,
`e2e/fixtures.js` goes stale without failing anything — which is exactly what
`live.spec.js` exists to catch.

---

## Tests

The suite splits the same way the backend's pytest suite does, and for the same
reason: almost everything worth asserting about this UI is a property of the
frontend alone, and stubbing the API makes those assertions fast, deterministic,
and runnable with no secrets.

```bash
npm run test:e2e        # 18 tests, hermetic. Playwright starts Vite itself
npm run test:e2e:live   # 3 tests, real backend on :8000, real Groq quota
```

What the hermetic suite pins: that the trace is requested at all; that a refusal
renders as an answer rather than an error and cites nothing while still
explaining itself; that markdown renders instead of showing raw `**` and `1.`
characters; that the answer region is announced to assistive technology
(`role="log"`, `aria-live="polite"`) and the icon-only send button is labelled;
that an empty question cannot be sent; that Enter sends and the input clears;
and that a 503 surfaces instead of hanging, leaving the page usable afterwards.

Playwright starts the dev server itself and reuses one that is already running,
so a dev session is not killed and restarted on every run. Point it elsewhere
with `E2E_BASE_URL`.

---

## Deployment

Builds to a static `dist/` — its own Vercel project, or any static host. Set
`VITE_API_URL` to the deployed API's origin at **build** time (Vite inlines it;
changing it later means rebuilding), and add that host to the backend's
`CORS_ORIGINS`.

See the [root README](../README.md) for the backend, the retrieval design, and
the evaluation results.
