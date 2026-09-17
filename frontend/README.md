# AI Chat Assistant Interface 🤖

[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF.svg)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-4.0-38B2AC.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, high-performance frontend architecture for an AI-driven chat application. This project features a modular design, sophisticated file upload capabilities (specifically for Excel processing), and a polished, accessibility-focused UI built with React 19 and Tailwind CSS 4.

---

## 🚀 Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Core** | [React 19](https://react.dev/) | UI Library & Component Architecture |
| **Build Tool** | [Vite 8](https://vitejs.dev/) | Ultra-fast development server & bundling |
| **Styling** | [Tailwind CSS 4](https://tailwindcss.com/) | Utility-first CSS framework with native Vite support |
| **State Management** | [TanStack Query v5](https://tanstack.com/query/latest) | Server-state management & API caching |
| **Routing** | [React Router 7](https://reactrouter.com/) | Declarative client-side navigation |
| **HTTP Client** | [Axios](https://axios-http.com/) | Promise-based API requests |
| **Icons** | [React Icons](https://react-icons.github.io/react-icons/) | Unified icon system (Material, IonIcons) |

---

## ✨ Features

- **Advanced Chat Interface**: Smooth message streaming simulation and state-aware input controls.
- **Excel File Integration**: Robust support for attaching and uploading `.xlsx` files alongside text queries.
- **Intelligent Feedback**: Real-time upload status, file badges, and clearable attachments.
- **Sticky Architecture**: Backdrop-blur headers and gradient-masked input bars for a premium "glassmorphism" feel.
- **Mock System**: Integrated mock API layer for development without backend dependencies.
- **Responsive Design**: Mobile-first approach ensuring usability across all device sizes.
- **Theme Optimization**: Custom `#f7f4f0` cream palette designed for reduced eye strain during long sessions.

---

## 🏗️ Project Architecture Overview

This project follows a **Modular Feature-Based Architecture** designed for scalability and maintainability.

### 1. Component Strategy
- **Atomic-ish Design**: Components are split into logical units. Large components like `ChatInterface` are refactored into focused sub-components like `Header` and `InputBar`.
- **Props-Driven**: State is lifted to parent containers (`ChatInterface`) and passed down via props to ensure a single source of truth.

### 2. State Flow
- **Server State**: Managed exclusively by **TanStack Query**. This handles caching, loading states, and error handling for all API interactions.
- **Local State**: Managed via React's `useState` for UI-specific logic (e.g., input values, file selection).

### 3. API Communication Flow
We use a **Service Layer Pattern**:
1. **Components** trigger mutations via TanStack Query.
2. **Mutations** call functions in `src/services/api.js`.
3. **Services** use a pre-configured Axios instance to communicate with the backend.
4. **Multipart Support**: Optimized for sending `FormData` containing both text and binary files.

### 4. Folder Organization Philosophy
The structure is designed to separate concerns:
- `components/`: Pure UI and reusable logic.
- `services/`: All outbound networking logic.
- `pages/`: Layout-heavy route components.
- `layouts/`: Global wrappers (navigation, sidebars).

---

## 📁 Folder Structure

```bash
src/
 ├── assets/             # Static assets (images, global SVGs)
 ├── components/         # Modular UI components
 │    ├── ChatInterface  # Main chat logic & message rendering
 │    ├── Header         # Sticky top navigation
 │    └── InputBar       # Floating input field & file controls
 ├── hooks/              # Custom reusable React hooks
 ├── layouts/            # Page layout wrappers (e.g., MainLayout)
 ├── pages/              # Route-level components (e.g., Home)
 ├── services/           # API service layer (Axios instances)
 ├── utils/              # Helper functions & formatting logic
 ├── App.jsx             # Root application component
 ├── index.css           # Global styles & Tailwind directives
 └── main.jsx            # Entry point & Provider configuration
```

---

## 🛠️ Getting Started

### Prerequisites
- Node.js (v18.0.0 or higher)
- npm or yarn

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the frontend directory:
   ```bash
   cd mk_proj_2/frontend
   ```
3. Install dependencies:
   ```bash
   npm install
   ```

### Available Scripts

| Command | Description |
| :--- | :--- |
| `npm run dev` | Runs the app in development mode with HMR |
| `npm run build` | Builds the app for production in the `dist` folder |
| `npm run lint` | Runs ESLint to check for code quality issues |
| `npm run preview` | Previews the production build locally |

---

## 🌐 Environment Variables

Create a `.env` file in the root directory:

```env
# API Configuration
VITE_API_URL=http://localhost:8000
```

*Note: All environment variables must be prefixed with `VITE_` to be accessible in the codebase.*

---

## 📖 API Reference

Talks to the FastAPI backend in `backend/` (run with `uvicorn app.main:app --reload`
from `backend/`, default port `8000`). The backend must allow this app's origin via
its `CORS_ORIGINS` setting — see `backend/.env.example`.

### Ask a question
**Endpoint**: `POST /ask` (base URL from `VITE_API_URL`)
**Content-Type**: `application/json`

| Field | Type | Description |
| :--- | :--- | :--- |
| `question` | String | The user's question, 1-2000 chars |
| `model_role` | String | `"generator"` (default) or `"reasoner"` |
| `explain` | Boolean | Include the per-stage trace (default `false`) |

**Response Format**:
```json
{
  "answer": "...",
  "cited_doc_ids": ["RB-001"],
  "confidence": "high",
  "elapsed_ms": 812,
  "trace": [],
  "llm_calls": 1
}
```

`confidence` is one of `high | medium | low | no_match`. `no_match` (with an
empty `cited_doc_ids`) is a normal, successful response — the agent declining
to answer rather than guessing — and should be rendered as such, not as an
error.

---

## 🤝 Contributing

1. **Feature Branches**: Use `feature/` or `fix/` prefixes.
2. **Linting**: Ensure `npm run lint` passes before committing.
3. **Architecture**: Maintain the service layer pattern for all new API integrations.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Maintained by**: [Senior Frontend Team]  
**Last Updated**: May 2026
