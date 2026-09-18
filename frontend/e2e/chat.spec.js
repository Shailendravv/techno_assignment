import { test, expect } from '@playwright/test';
import { ANSWERED, REFUSED, ask, log, stubAsk, waitForAnswer } from './fixtures.js';

/**
 * The hermetic end-to-end suite: a real browser, the real React app, a stubbed
 * API. No backend, no credentials, no quota.
 *
 * Several of these guard a bug that was actually shipped, and each says so. A
 * regression test whose provenance is written down is a much harder thing to
 * delete by accident than one that merely passes.
 */

test.describe('the page itself', () => {
  test('loads under its own name', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Runbook Agent/i);
    await expect(page.getByRole('heading', { name: 'Runbook Agent' })).toBeVisible();
  });

  test('does not claim to be powered by Copilot', async ({ page }) => {
    // The header shipped with someone else's product name in it.
    await page.goto('/');
    await expect(page.locator('body')).not.toContainText(/copilot/i);
  });

  test('announces answers to assistive technology', async ({ page }) => {
    // The message list had no live region, so a screen reader was never told an
    // answer had arrived - the reply simply appeared, silently.
    await page.goto('/');
    const region = log(page);
    await expect(region).toHaveAttribute('aria-live', 'polite');
  });

  test('labels the icon-only send button', async ({ page }) => {
    // The button's only content is an SVG, so without a label it is announced
    // as "button" and nothing else.
    await page.goto('/');
    await expect(page.getByRole('button', { name: 'Send question' })).toBeVisible();
  });

  test('will not send an empty question', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('button', { name: 'Send question' })).toBeDisabled();
  });
});

test.describe('asking a question', () => {
  test('requests the trace', async ({ page }) => {
    // The regression that motivated this whole file. `ChatInterface` called
    // `askQuestion(question)` with no options, so `explain` defaulted to false
    // and the trace was never requested - while the backend schema described
    // the flag as "on in the UI". Nothing rendered differently, so nothing
    // caught it. This asserts on the request body rather than the DOM, because
    // the request body is where the bug lived.
    const calls = await stubAsk(page, ANSWERED);
    await page.goto('/');
    await ask(page, 'checkout-api is running hot on CPU - what should I check first?');
    await waitForAnswer(page);

    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      question: 'checkout-api is running hot on CPU - what should I check first?',
      explain: true,
    });
  });

  test('shows the question, then the answer, then the evidence', async ({ page }) => {
    await stubAsk(page, ANSWERED);
    await page.goto('/');
    await ask(page, 'checkout-api is running hot on CPU - what should I check first?');
    await waitForAnswer(page);

    await expect(log(page)).toContainText('checkout-api is running hot on CPU');
    await expect(log(page)).toContainText('py-spy');
    await expect(log(page).getByText('confidence: high')).toBeVisible();
    await expect(log(page).getByText('RB-001', { exact: true })).toBeVisible();
  });

  test('renders markdown rather than its source characters', async ({ page }) => {
    // The answers come back as markdown; before `react-markdown` was wired in,
    // the numbered steps rendered as literal "1." and code spans as backticks.
    await stubAsk(page, ANSWERED);
    await page.goto('/');
    await ask(page, 'cpu');
    await waitForAnswer(page);

    await expect(log(page).getByRole('listitem').first()).toBeVisible();
    await expect(log(page).locator('code').first()).toHaveText('kubectl top pod');
    await expect(log(page).locator('strong').first()).toHaveText('py-spy');
    await expect(log(page)).not.toContainText('**py-spy**');
  });

  test('clears the input after sending', async ({ page }) => {
    await stubAsk(page, ANSWERED);
    await page.goto('/');
    await ask(page, 'cpu');
    await expect(page.getByPlaceholder('Ask about a runbook...')).toHaveValue('');
  });

  test('sends on Enter', async ({ page }) => {
    const calls = await stubAsk(page, ANSWERED);
    await page.goto('/');
    await page.getByPlaceholder('Ask about a runbook...').fill('cpu');
    await page.getByPlaceholder('Ask about a runbook...').press('Enter');
    await waitForAnswer(page);
    expect(calls).toHaveLength(1);
  });

  test('keeps both questions when two are sent in quick succession', async ({ page }) => {
    // `handleSend` read `messages` from the render closure instead of using a
    // functional update, so a second send before React re-rendered dropped the
    // first message. `Date.now()` ids could also collide into a duplicate React
    // key, which silently drops a row.
    await stubAsk(page, ANSWERED);
    await page.goto('/');

    await ask(page, 'first question about cpu');
    await ask(page, 'second question about memory');

    await expect(log(page)).toContainText('first question about cpu');
    await expect(log(page)).toContainText('second question about memory');
  });
});

test.describe('the trace disclosure', () => {
  test('is collapsed by default and expands to the stage log', async ({ page }) => {
    await stubAsk(page, ANSWERED);
    await page.goto('/');
    await ask(page, 'cpu');
    await waitForAnswer(page);

    const summary = page.getByText(/Why this answer/);
    await expect(summary).toBeVisible();
    await expect(page.getByText('gate: pass - coverage 100%, lexical 1.26 per term')).toBeHidden();

    await summary.click();
    await expect(page.getByText('gate: pass - coverage 100%, lexical 1.26 per term')).toBeVisible();
    await expect(page.locator('details ol li')).toHaveCount(ANSWERED.trace.length);
  });

  test('counts the model calls, and pluralises honestly', async ({ page }) => {
    await stubAsk(page, { ...ANSWERED, llm_calls: 1 });
    await page.goto('/');
    await ask(page, 'cpu');
    await waitForAnswer(page);
    await expect(page.getByText(/1 model call\b/)).toBeVisible();
  });
});

test.describe('a refusal', () => {
  test('is an answer, not an error', async ({ page }) => {
    // `confidence: no_match` is a successful 200 response and the single most
    // important thing this system does. Rendering it as a failure would misread
    // the contract that `services/api.js` documents.
    await stubAsk(page, REFUSED);
    await page.goto('/');
    await ask(page, 'how do I file my taxes?');
    await waitForAnswer(page);

    await expect(log(page).getByText('confidence: no_match')).toBeVisible();
    await expect(log(page)).toContainText("I don't have a runbook covering that");
    await expect(log(page)).not.toContainText('Error:');
  });

  test('cites nothing', async ({ page }) => {
    await stubAsk(page, REFUSED);
    await page.goto('/');
    await ask(page, 'how do I file my taxes?');
    await waitForAnswer(page);
    await expect(log(page).getByText(/^RB-\d+$/)).toHaveCount(0);
  });

  test('still explains itself', async ({ page }) => {
    // For a declined question the trace is the only thing that says *why*.
    await stubAsk(page, REFUSED);
    await page.goto('/');
    await ask(page, 'how do I file my taxes?');
    await waitForAnswer(page);

    await page.getByText(/Why this answer/).click();
    await expect(page.getByText(/gate: fail/)).toBeVisible();
  });
});

test.describe('when the backend is unwell', () => {
  test('surfaces a 503 instead of hanging', async ({ page }) => {
    await stubAsk(
      page,
      { detail: 'The document store is unavailable. incident=abc123' },
      { status: 503 },
    );
    await page.goto('/');
    await ask(page, 'cpu');

    await expect(log(page).getByText(/Error:/)).toBeVisible({ timeout: 30_000 });
    await expect(log(page)).toContainText('The document store is unavailable');
  });

  test('stays usable after a failure', async ({ page }) => {
    await page.route('**/ask', (route) => route.abort('failed'));
    await page.goto('/');
    await ask(page, 'cpu');

    await expect(log(page).getByText(/Error:/)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByPlaceholder('Ask about a runbook...')).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Send question' })).toBeDisabled();
  });
});
