from app.db.db import Database


def create_document(file_name, blob_path, container, file_type):
    conn = Database.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:

                cur.execute(
                    "SELECT document_id, status FROM documents WHERE file_name=%s",
                    (file_name,)
                )
                result = cur.fetchone()

                if result:
                    doc_id, status = result

                    if status == "completed":
                        return doc_id, True

                    cur.execute(
                        """
                        UPDATE documents
                        SET blob_path=%s,
                            storage_container=%s,
                            file_type=%s,
                            updated_at=NOW(),
                            status=%s
                        WHERE document_id=%s
                        """,
                        (blob_path, container, file_type, "pending", doc_id)
                    )

                else:
                    cur.execute(
                        """
                        INSERT INTO documents (
                            file_name,
                            blob_path,
                            storage_container,
                            file_type,
                            status
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING document_id
                        """,
                        (file_name, blob_path, container, file_type, "pending")
                    )

                    doc_id = cur.fetchone()[0]

                return doc_id, False

    finally:
        Database.return_connection(conn)


def update_status(doc_id, status):
    conn = Database.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status=%s WHERE document_id=%s",
                    (status, doc_id)
                )
    finally:
        Database.return_connection(conn)


def insert_page(document_id, page_number, page_blob_path, local_path):
    conn = Database.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pages (
                        document_id,
                        page_number,
                        page_blob_path,
                        local_path
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id, page_number) DO NOTHING
                    """,
                    (document_id, page_number, page_blob_path, local_path)
                )
    finally:
        Database.return_connection(conn)


def insert_ocr(document_id, page_number, content, tags):
    conn = Database.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ocr_results (
                        document_id,
                        page_number,
                        content,
                        tags
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id, page_number) DO NOTHING
                    """,
                    (document_id, page_number, content, tags)
                )
    finally:
        Database.return_connection(conn)


def update_page_count(doc_id, page_count):
    conn = Database.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET page_count=%s WHERE document_id=%s",
                    (page_count, doc_id)
                )
    finally:
        Database.return_connection(conn)


def insert_embedding(
    id,
    document_id,
    file_name,
    file_type,
    page_number,
    chunk_id,
    content,
    embedding
):
    conn = Database.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO embeddings (
                        id, document_id, file_name, file_type,
                        page_number, chunk_id, content, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, page_number, chunk_id) DO NOTHING
                    """,
                    (
                        id,
                        document_id,
                        file_name,
                        file_type,
                        page_number,
                        chunk_id,
                        content,
                        embedding
                    )
                )
    finally:
        Database.return_connection(conn)