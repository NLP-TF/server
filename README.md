# FastAPI Server

A simple FastAPI server with basic CRUD operations.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

## API Documentation

- Interactive API docs (Swagger UI): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc

## Endpoints

- `GET /` - Welcome message
- `GET /items/` - List all items
- `POST /items/` - Create a new item
- `GET /items/{item_id}` - Get a specific item by ID
