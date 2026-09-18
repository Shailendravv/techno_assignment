import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests for the Runbook Agent UI.
 *
 * The suite is split the same way the backend's pytest suite is, and for the
 * same reason. `backend/pytest.ini` declares a `network` marker so that CI can
 * run `-m "not network"` and prove the suite needs no credentials; here the
 * equivalent is the `@live` tag.
 *
 *   npm run test:e2e        every test that stubs the API - free, hermetic,
 *                           needs no backend and no API key. This is what CI
 *                           runs.
 *   npm run test:e2e:live   the `@live` tests, which drive the real pipeline
 *                           through a running backend. These spend Groq quota,
 *                           so they are opt-in and never run automatically.
 *
 * The split matters more than it looks. Almost everything worth asserting about
 * this UI - that the trace is requested at all, that a refusal renders as an
 * answer rather than an error, that markdown is not shown raw - is a property
 * of the frontend alone, and stubbing the API is what makes those assertions
 * fast, deterministic, and runnable on a machine with no secrets.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],

  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],

  // Start Vite unless one is already up. `reuseExistingServer` keeps a dev
  // session from being killed and restarted on every run.
  webServer: {
    command: 'npm run dev',
    url: process.env.E2E_BASE_URL || 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
