import { test, expect } from '@playwright/test';
import { ask, log, waitForAnswer } from './fixtures.js';

/**
 * The real thing: browser -> API -> retrieval -> model -> rendered answer.
 *
 * Tagged `@live` and excluded from `npm run test:e2e`, because these spend Groq
 * quota on a free tier. Run them deliberately:
 *
 *     cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000
 *     cd frontend && npm run test:e2e:live
 *
 * Kept deliberately small - two questions, one answered and one refused. The
 * per-question scoring job belongs to `backend/eval/harness.py`, which measures
 * twenty questions and produces a four-outcome report. Duplicating that here
 * would cost the same quota and produce no score.
 *
 * What these two add that the harness cannot: they prove the contract between
 * the stubs in `fixtures.js` and the server is still real. Every hermetic test
 * in `chat.spec.js` is only as truthful as those fixtures.
 */

test.describe('@live against a running backend', () => {
  // A real pipeline run is several model calls behind a paced free tier.
  test.setTimeout(120_000);

  test('answers a question the runbooks cover, with a citation @live', async ({ page }) => {
    await page.goto('/');
    await ask(page, 'checkout-api is running hot on CPU - what should I check first?');
    await waitForAnswer(page);

    // Not asserting the prose. The answer is model output and will vary; the
    // citation and the confidence are the contract.
    await expect(log(page).getByText(/confidence: (high|medium)/)).toBeVisible();
    await expect(log(page).getByText('RB-001', { exact: true })).toBeVisible();
  });

  test('declines a question the runbooks do not cover @live', async ({ page }) => {
    await page.goto('/');
    await ask(page, 'What is the refund policy for enterprise customers in Germany?');
    await waitForAnswer(page);

    await expect(log(page).getByText('confidence: no_match')).toBeVisible();
    await expect(log(page)).not.toContainText('Error:');
  });

  test('reports a real trace, not an empty one @live', async ({ page }) => {
    // Regression guard for the `explain` flag, asserted end to end: if the UI
    // stopped requesting it, or the backend stopped honouring it, the
    // disclosure would not render at all.
    await page.goto('/');
    await ask(page, 'checkout-api is running hot on CPU - what should I check first?');
    await waitForAnswer(page);

    const summary = page.getByText(/Why this answer/);
    await expect(summary).toBeVisible();
    await summary.click();
    expect(await page.locator('details ol li').count()).toBeGreaterThan(3);
  });
});
