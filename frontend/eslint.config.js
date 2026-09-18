import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist', 'playwright-report', 'test-results']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
  {
    // The Playwright config and the specs run in Node, not in the browser, so
    // they legitimately reach for `process`. Scoped rather than global, so the
    // application code keeps failing lint if it ever does the same.
    files: ['playwright.config.js', 'e2e/**/*.js'],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
])
