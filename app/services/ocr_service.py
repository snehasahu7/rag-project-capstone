import os
from pypdf import PdfReader, PdfWriter
from langchain_core.documents import Document
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.storage_service import AzureStorageService
from app.services.db_service import insert_page, insert_ocr
from app.services.embedding_service import generate_embedding
from app.services.db_service import insert_embedding
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
storage = AzureStorageService()


def split_pdf(file_path: str):
    reader = PdfReader(file_path)
    page_files = []

    base_doc_name = os.path.splitext(os.path.basename(file_path))[0]

    base_doc_name = (
        base_doc_name
        .replace(" ", "_")
        .replace("'", "")
        .replace('"', "")
        .replace("-", "_")
    )

    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        output_path = f"/tmp/{base_doc_name}_page_{i+1}.pdf"

        with open(output_path, "wb") as f:
            writer.write(f)

        page_files.append((output_path, i + 1))

    return page_files


def clean_text(text: str):
    text = text.replace("<figure>", "").replace("</figure>", "")
    return " ".join(text.split())


def run_ocr(file_path: str, document_id: str, doc_id: int, file_type: str):
    page_files = split_pdf(file_path)
    all_docs = []

    client = DocumentIntelligenceClient(
        endpoint=settings.AZURE_DOC_INTEL_ENDPOINT,
        credential=DefaultAzureCredential()
    )

    BATCH_SIZE = 3

    def process_page(page_tuple):
        page_path, page_num = page_tuple

        logger.info(f"OCR on page {page_num}")

        blob_path = storage.upload_page_pdf(
            local_path=page_path,
            document_name=document_id,
            page_num=page_num,
            file_type=file_type
        )

        logger.info("Opening file...")

        with open(page_path, "rb") as f:
            logger.info(f"Sending OCR request for page {page_num}...")
            poller = client.begin_analyze_document(
                "prebuilt-read",
                body=f
            )

        logger.info(f"Waiting for OCR result for page {page_num}...")

        result = poller.result()

        logger.info(f"OCR completed for page {page_num}")

        insert_page(
            document_id=doc_id,
            page_number=page_num,
            page_blob_path=blob_path,
            local_path=page_path
        )

        docs = []
        full_text = ""

        try:
            if not result or not result.pages:
                logger.warning(f"No OCR pages returned for page {page_num}")
            else:
                for page in result.pages:
                    if not page:
                        continue

                    lines = getattr(page, "lines", None)
                    if not lines:
                        continue

                    for line in lines:
                        if line and getattr(line, "content", None):
                            full_text += line.content + " "

        except Exception as e:
            logger.error(f"Text extraction failed for page {page_num}: {e}")
            full_text = ""

        text = clean_text(full_text) if full_text else ""

        if not text:
            logger.warning(f"Empty OCR text for page {page_num}")
        tags = ["general"]

        insert_ocr(
            document_id=doc_id,
            page_number=page_num,
            content=text,
            tags=tags
        )

        
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "source": document_id,
                    "page": page_num,
                    "tags": tags
                }
            )
        )

        return docs

    for i in range(0, len(page_files), BATCH_SIZE):
        batch = page_files[i:i + BATCH_SIZE]

        logger.info(f"Processing batch: pages {[p[1] for p in batch]}")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_page, p) for p in batch]

            for future in as_completed(futures):
                try:
                    result_docs = future.result()
                    all_docs.extend(result_docs)
                except Exception as e:
                    logger.error(f"Error processing page: {e}")

    def _chunk_text(text, size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    logger.info("Starting embedding generation...")

    for doc in all_docs:
        text = doc.page_content
        page_num = doc.metadata["page"]

        if not text or not text.strip():
            continue

        chunks = _chunk_text(text)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            try:
                embedding = generate_embedding(chunk)

                if embedding is None:
                    continue

                insert_embedding(
                    id=f"{doc_id}_p{page_num}_c{i}",
                    document_id=doc_id,
                    file_name=document_id,
                    file_type=file_type,
                    page_number=page_num,
                    chunk_id=i,
                    content=chunk,
                    embedding=embedding
                )

            except Exception as e:
                logger.error(f"Embedding failed for page {page_num}, chunk {i}: {e}")

    return all_docs