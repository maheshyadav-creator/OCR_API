CREATE TABLE IF NOT EXISTS ocr_results (
    id SERIAL PRIMARY KEY,

    filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(100) NOT NULL,

    extracted_text TEXT NOT NULL,

    confidence DOUBLE PRECISION NOT NULL
        CHECK (confidence >= 0.0 AND confidence <= 1.0),

    result_json JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX IF NOT EXISTS idx_ocr_results_created_at
ON ocr_results (created_at DESC);