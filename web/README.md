# Web UI (React + Vite)

## Run

Terminal 1 (backend):

```bash
cd ..
pip install -e ".[api]"
PYTHONPATH=src python scripts/run_api.py
```

Terminal 2 (frontend):

```bash
cd web
npm install
npm run dev
```

## Notes

- UI posts to `/api/chat` through Vite proxy (`127.0.0.1:8000`).
- Current frontend includes a graceful demo fallback if backend is not connected.
