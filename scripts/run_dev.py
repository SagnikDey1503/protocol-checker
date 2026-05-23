"""
Run the FastAPI development server with auto-reload.
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI development server on http://localhost:8000")
    print("Press Ctrl+C to stop.")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
