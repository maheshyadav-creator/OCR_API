# OCR Application

A production-oriented OCR processing service built with **FastAPI, PaddleOCR, PostgreSQL, Redis Streams, and Docker**.

The application is designed to accept images and PDF documents, create asynchronous OCR jobs, process them using multiple OCR workers, store results in PostgreSQL, and provide job status and result APIs.

---

## Features

- FastAPI REST API
- PaddleOCR for text extraction
- Supports images and PDF files
- Multi-file upload
- Asynchronous job processing
- Redis Streams for job queuing
- 3 dedicated OCR worker processes
- PostgreSQL for persistent job metadata and OCR results
- Direct PostgreSQL access using `psycopg2`
- No SQLAlchemy or ORM
- Job status tracking
- Retry handling for temporary failures
- Permanent failure handling
- Redis pending-message recovery
- Periodic worker recovery
- Cron-based recovery
- Idempotent OCR result storage
- Docker and Docker Compose
- Persistent storage volumes
- Health-check endpoints
- File type validation
- File size validation

---

## Technology Stack

| Technology | Purpose |
|---|---|
| FastAPI | REST API |
| Uvicorn | ASGI server |
| PaddleOCR | OCR processing |
| PostgreSQL | Persistent data storage |
| psycopg2 | Direct PostgreSQL connection |
| Redis Streams | Asynchronous job queue |
| Docker | Containerization |
| Docker Compose | Multi-container orchestration |
| Python 3.11 | Application runtime |

---

## Architecture

The application separates API request handling from the expensive OCR processing.

```text
                         Client
                           |
                           | HTTP
                           v
                    +--------------+
                    |   FastAPI    |
                    |     API      |
                    +--------------+
                           |
              +------------+------------+
              |                         |
              v                         v
        +-----------+             +-----------+
        | PostgreSQL|             |   Redis   |
        |  Job Data |             |   Streams |
        +-----------+             +-----------+
                                      |
                                      |
                         +------------+------------+
                         |            |            |
                         v            v            v
                   +----------+ +----------+ +----------+
                   | Worker 1 | | Worker 2 | | Worker 3 |
                   +----------+ +----------+ +----------+
                         |            |            |
                         +------------+------------+
                                      |
                                      v
                                +-----------+
                                | PaddleOCR |
                                +-----------+
                                      |
                                      v
                                +-----------+
                                | PostgreSQL|
                                |  Results  |
                                +-----------+