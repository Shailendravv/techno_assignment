import { expect } from '@playwright/test';

/**
 * Response fixtures, copied from real `POST /ask` responses.
 *
 * These mirror `backend/app/schemas.py::AskResponse`. If that contract changes,
 * these stubs go stale silently - which is exactly why `live.spec.js` exists and
 * asserts the same shape against the real server.
 */

export const ANSWERED = {
  answer:
    'Check the following, in order:\n\n' +
    '1. Confirm the CPU is genuinely saturated with `kubectl top pod`.\n' +
    '2. Profile the hot path using **py-spy**.\n' +
    '3. Check for a recent deploy that changed request handling.\n',
  cited_doc_ids: ['RB-001'],
  confidence: 'high',
  trace: [
    'analyze: service=checkout-api failure_mode=cpu intent=diagnose',
    'retrieve: hybrid, 8 lexical + 8 dense, 4 kept',
    'gate: pass - coverage 100%, lexical 1.26 per term',
    'grade: 4 candidates, 4 relevant',
    'ground: cited RB-001, invented none',
  ],
  llm_calls: 3,
};

export const REFUSED = {
  answer:
    "I don't have a runbook covering that. The documents I can see are about " +
    'checkout-api, payments-api and search-api.',
  cited_doc_ids: [],
  confidence: 'no_match',
  trace: [
    'analyze: service=None failure_mode=None intent=diagnose',
    'retrieve: hybrid, 8 lexical + 8 dense, 4 kept',
    'gate: fail - coverage 20% below floor 60%',
  ],
  llm_calls: 0,
};

/**
 * Intercept `POST /ask` and reply with `body`.
 *
 * Returns a `calls` array that collects each request payload, so a test can
 * assert on what the UI *sent* and not only on what it rendered. That is the
 * only way to catch the class of bug where the UI quietly stops asking for
 * something the backend still supports.
 */
export async function stubAsk(page, body, { status = 200 } = {}) {
  const calls = [];

  await page.route('**/ask', async (route) => {
    calls.push(JSON.parse(route.request().postData() || '{}'));
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });

  return calls;
}

/** Type a question and send it. */
export async function ask(page, question) {
  const box = page.getByPlaceholder('Ask about a runbook...');
  await box.fill(question);
  await page.getByRole('button', { name: 'Send question' }).click();
}

/** The live-answer region, which is also the accessibility contract. */
export function log(page) {
  return page.getByRole('log');
}

/** Wait until an assistant answer has landed in the log. */
export async function waitForAnswer(page) {
  await expect(log(page).getByText(/confidence:/)).toBeVisible({ timeout: 60_000 });
}
