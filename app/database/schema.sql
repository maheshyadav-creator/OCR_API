-- ============================================================
-- OCR JOBS
-- Stores the lifecycle/state of every submitted document.
-- ============================================================

CREATE TABLE IF NOT EXISTS ocr_jobs (
    id UUID PRIMARY KEY,

    filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(100) NOT NULL,

    file_path TEXT NOT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'queued'
        CHECK (
            status IN (
                'queued',
                'processing',
                'retrying',
                'completed',
                'failed'
            )
        ),

    attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_count >= 0),

    worker_id VARCHAR(255),

    error_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- Index for finding jobs by status quickly.
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status
ON ocr_jobs (status);


-- Index for newest jobs.
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_created_at
ON ocr_jobs (created_at DESC);


-- Index for finding jobs handled by a particular worker.
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_worker_id
ON ocr_jobs (worker_id);


-- ============================================================
-- OCR RESULTS
-- Stores the actual OCR output.
-- ============================================================

CREATE TABLE IF NOT EXISTS ocr_results (
    id SERIAL PRIMARY KEY,

    job_id UUID,

    filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(100) NOT NULL,

    extracted_text TEXT NOT NULL,

    confidence DOUBLE PRECISION NOT NULL
        CHECK (confidence >= 0.0 AND confidence <= 1.0),

    result_json JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- Add job_id to an existing ocr_results table if necessary.
ALTER TABLE ocr_results
ADD COLUMN IF NOT EXISTS job_id UUID;


-- Only one OCR result should belong to one job.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ocr_results_job_id
ON ocr_results (job_id);


-- ============================================================
-- RELATIONSHIP BETWEEN JOB AND RESULT
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_ocr_results_job'
    ) THEN

        ALTER TABLE ocr_results
        ADD CONSTRAINT fk_ocr_results_job
        FOREIGN KEY (job_id)
        REFERENCES ocr_jobs(id)
        ON DELETE CASCADE;

    END IF;
END $$;


-- Index for newest OCR results.
CREATE INDEX IF NOT EXISTS idx_ocr_results_created_at
ON ocr_results (created_at DESC);