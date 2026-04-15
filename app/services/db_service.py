from app.db.db import get_connection


def create_document(file_name, blob_path, container, file_type):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT document_id FROM documents WHERE file_name=%s",
        (file_name,)
    )
    result = cur.fetchone()

    if result:
        doc_id = result[0]


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

        cur.execute(
            "DELETE FROM ocr_results WHERE document_id=%s",
            (doc_id,)
        )
        cur.execute(
            "DELETE FROM pages WHERE document_id=%s",
            (doc_id,)
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

    conn.commit()
    cur.close()
    conn.close()

    return doc_id


def update_status(doc_id, status):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE documents SET status=%s WHERE document_id=%s",
        (status, doc_id)
    )

    conn.commit()
    cur.close()
    conn.close()


def insert_page(document_id, page_number, page_blob_path, local_path):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO pages (
            document_id,
            page_number,
            page_blob_path,
            local_path
        )
        VALUES (%s, %s, %s, %s)
        """,
        (document_id, page_number, page_blob_path, local_path)
    )

    conn.commit()
    cur.close()
    conn.close()


def insert_ocr(document_id, page_number, content, tags):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO ocr_results (
            document_id,
            page_number,
            content,
            tags
        )
        VALUES (%s, %s, %s, %s)
        """,
        (document_id, page_number, content, tags)
    )

    conn.commit()
    cur.close()
    conn.close()

def update_page_count(doc_id, page_count):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE documents SET page_count=%s WHERE document_id=%s",
        (page_count, doc_id)
    )

    conn.commit()
    cur.close()
    conn.close()

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
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO embeddings (
            id, document_id, file_name, file_type,
            page_number, chunk_id, content, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
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

    conn.commit()
    cur.close()
    conn.close()