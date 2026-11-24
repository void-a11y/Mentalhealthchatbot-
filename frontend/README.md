# Frontend

React + TypeScript + Vite chat UI for the RAG backend.

```bash
npm install
cp .env.example .env   # VITE_API_URL, default http://127.0.0.1:8000
npm run dev            # http://localhost:5173
npm test && npm run build
```

Calls `POST /chat` with `{session_id, query}`. Model output is parsed into data (bold, lists, `[n]` markers) and never injected as HTML.
