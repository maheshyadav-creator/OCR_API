# OCR Application

FastAPI OCR service using:

- FastAPI
- Uvicorn
- PaddleOCR
- PostgreSQL
- psycopg2
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
PostgreSQL
   |
   v
API Response
