from app.db.db import get_connection


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    # cur.execute("SET TIME ZONE 'Asia/Kolkata';")

    # ========================
    # Enable UUID extension
    # ========================
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    """)

    # ========================
    # ENABLE PGVECTOR
    # ========================
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    """)

    # ========================
    # DOCUMENTS TABLE
    # ========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

        file_name TEXT NOT NULL,

        blob_path TEXT NOT NULL,
        storage_container TEXT NOT NULL,

        uploaded_at TIMESTAMPTZ DEFAULT NOW(),
        file_type TEXT DEFAULT 'unsupported',

        status TEXT NOT NULL CHECK (
            status IN ('pending', 'processing', 'completed', 'failed')
        ) DEFAULT 'pending',

        page_count INT DEFAULT 0,

        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)

    # ========================
    # PAGES TABLE
    # ========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id SERIAL PRIMARY KEY,
        document_id UUID NOT NULL,
        page_number INT,
        page_blob_path TEXT,
        local_path TEXT,

        CONSTRAINT fk_pages_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
    );
    """)

    # ========================
    # OCR RESULTS TABLE
    # ========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ocr_results (
        id SERIAL PRIMARY KEY,
        document_id UUID NOT NULL,
        page_number INT,
        content TEXT,
        tags TEXT[],
        bbox JSONB,
        line_index INT,
        created_at TIMESTAMP DEFAULT NOW(),

        CONSTRAINT fk_ocr_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
    );
    """)

    # ========================
    # EMBEDDINGS TABLE
    # ========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,

        document_id UUID NOT NULL,
        file_name TEXT,
        file_type TEXT,

        page_number INT,
        chunk_id INT,

        content TEXT,

        embedding VECTOR(384),

        created_at TIMESTAMPTZ DEFAULT NOW(),

        CONSTRAINT fk_embeddings_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
    );
    """)

    # ========================
    # VECTOR INDEX
    # ========================
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings
    USING ivfflat (embedding vector_cosine_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("Tables created successfully")


if __name__ == "__main__":
    init_db()