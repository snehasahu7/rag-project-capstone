-- ─────────────────────────────────────────────────────────────────────────────
-- Intelligent Document Explorer — Database Initialisation (SYNCED WITH APP)
-- ─────────────────────────────────────────────────────────────────────────────

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ── documents ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    document_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name         TEXT NOT NULL,
    blob_path         TEXT NOT NULL,
    storage_container TEXT NOT NULL,
    uploaded_at       TIMESTAMPTZ DEFAULT NOW(),
    file_type         TEXT DEFAULT 'unsupported',
    status            TEXT NOT NULL CHECK (
                          status IN ('pending', 'processing', 'completed', 'failed')
                      ) DEFAULT 'pending',
    page_count        INT DEFAULT 0,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── pages ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pages (
    id              SERIAL PRIMARY KEY,
    document_id     UUID NOT NULL,
    page_number     INT,
    page_blob_path  TEXT,
    local_path      TEXT,

    CONSTRAINT fk_pages_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
);

-- ── ocr_results ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ocr_results (
    id          SERIAL PRIMARY KEY,
    document_id UUID NOT NULL,
    page_number INT,
    content     TEXT,
    tags        TEXT[],
    bbox        JSONB,
    line_index  INT,
    created_at  TIMESTAMP DEFAULT NOW(),

    CONSTRAINT fk_ocr_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
);

-- ── embeddings (MATCHES YOUR PYTHON) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,

    document_id UUID NOT NULL,
    file_name TEXT,
    file_type TEXT,

    page_number INT,
    chunk_id INT,

    content TEXT,

    embedding VECTOR(384),  -- IMPORTANT: matches all-MiniLM-L6-v2

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_embeddings_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
);

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);

CREATE INDEX IF NOT EXISTS idx_pages_document_id
    ON pages (document_id);

CREATE INDEX IF NOT EXISTS idx_ocr_document_id
    ON ocr_results (document_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_document_id
    ON embeddings (document_id);

-- Vector index
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
ON embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);