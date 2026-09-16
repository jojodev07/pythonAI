"""OCR a PDF with Google Cloud Vision, embed the text, and index it in Chroma.

Setup:
    pip install -r requirements.txt
    gcloud auth application-default login

The Google client uses DOCUMENT_TEXT_DETECTION for every PDF page. Native PDF
text extraction is intentionally not used in this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chromadb
import pymupdf
from dotenv import load_dotenv
from google.cloud import vision
from openai import OpenAI


DEFAULT_COLLECTION = "arabic_books"
EMBEDDING_MODEL = "text-embedding-3-small"
WORDS_PER_CHUNK = 500
EMBEDDING_BATCH_SIZE = 32
OCR_SCALE = 2


def ocr_pdf_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
	"""Yield Google OCR text for each non-empty page in a PDF."""
	ocr_client = vision.ImageAnnotatorClient()
	language_context = vision.ImageContext(language_hints=["ar", "en"])

	with pymupdf.open(pdf_path) as document:
		for page_number, page in enumerate(document, start=1):
			pixmap = page.get_pixmap(
				matrix=pymupdf.Matrix(OCR_SCALE, OCR_SCALE),
				alpha=False,
			)
			image = vision.Image(content=pixmap.tobytes("png"))
			response = ocr_client.document_text_detection(
				image=image,
				image_context=language_context,
			)

			if response.error.message:
				raise RuntimeError(
					f"Google OCR failed on page {page_number}: "
					f"{response.error.message}"
				)

			text = response.full_text_annotation.text.strip()
			if text:
				yield page_number, text


def chunk_pages(
	pdf_path: Path,
	words_per_chunk: int = WORDS_PER_CHUNK,
) -> Iterator[tuple[str, list[int]]]:
	"""Combine OCR words into bounded chunks while retaining source pages."""
	words: list[str] = []
	pages: list[int] = []

	for page_number, page_text in ocr_pdf_pages(pdf_path):
		page_words = page_text.split()
		position = 0
		while position < len(page_words):
			remaining = words_per_chunk - len(words)
			words.extend(page_words[position : position + remaining])
			if not pages or pages[-1] != page_number:
				pages.append(page_number)
			position += remaining

			if len(words) == words_per_chunk:
				yield " ".join(words), pages.copy()
				words.clear()
				pages.clear()

	if words:
		yield " ".join(words), pages.copy()


def make_chunk_id(pdf_path: Path, pages: list[int], text: str) -> str:
	identity = f"{pdf_path.resolve()}:{pages}:{text}".encode("utf-8")
	return hashlib.sha256(identity).hexdigest()


def add_book_context(pdf_path: Path, text: str) -> str:
	return f"Book: {pdf_path.stem}\n{text}"


def build_collection(collection_name: str) -> Any:
	missing = [
		name
		for name in ("CHROMADB_KEY", "CHROMADB_TENANT")
		if not os.getenv(name)
	]
	if missing:
		raise RuntimeError(
			"Missing environment variables: " + ", ".join(missing)
		)

	client = chromadb.CloudClient(
		api_key=os.environ["CHROMADB_KEY"],
		tenant=os.environ["CHROMADB_TENANT"],
		database=os.getenv("CHROMADB_DATABASE", "prod"),
	)
	return client.get_or_create_collection(collection_name)


def index_pdf(
	pdf_path: Path,
	collection: Any,
	openai_client: OpenAI,
	batch_size: int = EMBEDDING_BATCH_SIZE,
) -> int:
	"""OCR, embed, and store one PDF without loading it all into memory."""
	pending: list[tuple[str, list[int], str]] = []
	indexed_count = 0

	def flush() -> None:
		nonlocal indexed_count
		if not pending:
			return

		texts = [item[0] for item in pending]
		response = openai_client.embeddings.create(
			model=EMBEDDING_MODEL,
			input=texts,
		)
		collection.upsert(
			ids=[item[2] for item in pending],
			embeddings=[item.embedding for item in response.data],
			documents=texts,
			metadatas=[
				{
					"file_name": pdf_path.name,
					"file_path": str(pdf_path.resolve()),
					"page_start": pages[0],
					"page_end": pages[-1],
					"pages": ",".join(str(page) for page in pages),
					"source": f"{pdf_path.name}, pages {pages[0]}-{pages[-1]}",
				}
				for _, pages, _ in pending
			],
		)
		indexed_count += len(pending)
		pending.clear()

	for text, pages in chunk_pages(pdf_path):
		indexed_text = add_book_context(pdf_path, text)
		pending.append(
			(indexed_text, pages, make_chunk_id(pdf_path, pages, text))
		)
		if len(pending) >= batch_size:
			flush()
	flush()
	return indexed_count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Google Document OCR -> OpenAI embeddings -> Chroma Cloud"
	)
	parser.add_argument("pdf", type=Path, help="PDF file to OCR and index")
	parser.add_argument("--collection", default=DEFAULT_COLLECTION)
	parser.add_argument("--env-file", type=Path, default=Path("minimal.env"))
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	load_dotenv(args.env_file)
	if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
		raise FileNotFoundError(f"PDF file not found: {args.pdf}")
	if not os.getenv("OPENAI_API_KEY"):
		raise RuntimeError("Missing environment variable: OPENAI_API_KEY")

	collection = build_collection(args.collection)
	openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
	count = index_pdf(args.pdf, collection, openai_client)
	print(f"Indexed {count} chunks from {args.pdf} into '{args.collection}'")


if __name__ == "__main__":
	main()
