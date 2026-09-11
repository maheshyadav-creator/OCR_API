# OCR Application

FastAPI OCR service using:

- FastAPI
- Uvicorn
- PaddleOCR
- PostgreSQL
- SQLAlchemy
- Docker
- Docker Compose

## Architecture

```text
Client
   |
   v
FastAPI
   |
   v
OCR Route
   |
   v
OCR Service
   |
   v
PaddleOCR
   |
   v
OCR Result
   |
   +----------------+
   |                |
   v                v
PostgreSQL       API Response
