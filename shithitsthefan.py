"""Stream Arabic/English PDF content into ChromaDB.

Install these packages outside this script when setting up the environment:
	pip install pymupdf openai chromadb python-dotenv google-cloud-vision
	# Optional, improves Arabic bidirectional text ordering:
	pip install python-bidi
	The script does not install packages. It is intentionally page-streaming so a
large PDF does not need to be loaded into RAM before it can be indexed.

Google OCR uses Application Default Credentials and the GOOGLE_CLOUD_PROJECT
environment variable is optional for direct image annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_COLLECTION = "arabic_books"
WORDS_PER_CHUNK = 500
EMBEDDING_BATCH_SIZE = 32

# Quick setup: paste your credentials between the quotes when running in Colab.
# Do not share this file or commit it after adding real keys.
DIRECT_OPENAI_API_KEY = ""
DIRECT_CHROMADB_KEY = ""
DIRECT_CHROMADB_TENANT = ""
DIRECT_CHROMADB_DATABASE = "prod"


def configure_credentials() -> None:
	"""Use direct values only when the corresponding environment value is absent."""
	credentials = {
		"OPENAI_API_KEY": DIRECT_OPENAI_API_KEY,
		"CHROMADB_KEY": DIRECT_CHROMADB_KEY,
		"CHROMADB_TENANT": DIRECT_CHROMADB_TENANT,
		"CHROMADB_DATABASE": DIRECT_CHROMADB_DATABASE,
	}
	for name, value in credentials.items():
		if value:
			os.environ.setdefault(name, value)


def repair_arabic_text(text: str) -> str:
	"""Repair PDFs that return Arabic words in visual right-to-left order."""
	try:
		from bidi.algorithm import get_display  # type: ignore[import-not-found]
		return get_display(text)
	except ImportError:
		pass

	def reverse_arabic_run(match: re.Match[str]) -> str:
		return match.group(0)[::-1]

	return re.sub(
		r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\uFB50-\uFDFF\uFE70-\uFEFF]+",
		reverse_arabic_run,
		text,
	)


def text_needs_ocr(text: str) -> bool:
	"""Detect common broken-font output that text cleanup cannot reconstruct."""
	if not text.strip():
		return True
	arabic_count = len(re.findall(r"[\u0600-\u06ff]", text))
	if "�" in text or arabic_count == 0:
		return "�" in text
	# Broken Arabic PDF maps often leak Latin S, ü, or control-like symbols.
	corruption_markers = len(re.findall(r"[SüÃ�]", text))
	return corruption_markers >= 3 and corruption_markers / max(arabic_count, 1) > 0.01


def google_ocr_page_text(page: Any) -> str:
	"""OCR one rendered page with Google Cloud Vision's document OCR."""
	try:
		from google.cloud import vision
	except ImportError as error:
		raise RuntimeError(
			"Google OCR requires google-cloud-vision; install it with pip."
		) from error

	pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
	client = vision.ImageAnnotatorClient()
	image = vision.Image(content=pixmap.tobytes("png"))
	context = vision.ImageContext(language_hints=["ar", "en"])
	response = client.document_text_detection(image=image, image_context=context)
	if response.error.message:
		raise RuntimeError(f"Google OCR failed: {response.error.message}")
	return response.full_text_annotation.text if response.full_text_annotation else ""


def clean_whitespace(text: str) -> str:
	"""Clean layout whitespace without changing character order."""
	text = text.replace("\u00a0", " ").replace("\r", "\n")
	text = re.sub(r"[ \t]+", " ", text)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()


def normalise_text(text: str) -> str:
	return repair_arabic_text(clean_whitespace(text))


def table_to_text(table: Any) -> str:
	"""Represent a detected table as labelled, searchable rows."""
	rows = table.extract() or []
	rendered_rows = []
	for row in rows:
		cells = [clean_whitespace(str(cell or "")) for cell in row]
		if any(cells):
			rendered_rows.append("[Table row] " + " | ".join(cells))
	return "\n".join(rendered_rows)


def rectangles_intersect(first: Any, second: Any) -> bool:
	"""Return whether two PyMuPDF rectangles overlap by a meaningful amount."""
	return fitz.Rect(first).intersects(fitz.Rect(second))


def extract_page_text(
	page: Any,
	use_ocr: bool = False,
	google_ocr: bool = False,
) -> str:
	"""Extract normal text and tables, preserving useful reading order.

	PyMuPDF's table finder is available in current releases. On older releases
	the block-text fallback still extracts Arabic and English PDF text.
	"""
	tables = []
	try:
		tables = list(page.find_tables().tables)
	except (AttributeError, TypeError, ValueError):
		tables = []

	table_rectangles = [table.bbox for table in tables]
	blocks = page.get_text("blocks", sort=True)
	parts = []
	for block in blocks:
		block_text = block[4]
		block_rect = block[:4]
		if block_text and not any(
			rectangles_intersect(block_rect, table_rect) for table_rect in table_rectangles
		):
			parts.append(block_text)

	for table_number, table in enumerate(tables, start=1):
		table_text = table_to_text(table)
		if table_text:
			parts.append(f"[Table {table_number}]\n{table_text}")

	page_text = normalise_text("\n".join(parts))
	if google_ocr:
		return clean_whitespace(google_ocr_page_text(page))
	return page_text


def iter_page_text(
	pdf_path: Path,
	use_ocr: bool = False,
	google_ocr: bool = False,
) -> Iterator[tuple[int, str]]:
	"""Yield one page at a time; the document is never copied into a list."""
	with fitz.open(pdf_path) as document:
		for page_number, page in enumerate(document, start=1):
			page_text = extract_page_text(
				page,
				use_ocr=use_ocr,
				google_ocr=google_ocr,
			)
			if page_text:
				yield page_number, page_text


def iter_chunks(
	pdf_path: Path,
	words_per_chunk: int = WORDS_PER_CHUNK,
	use_ocr: bool = False,
	google_ocr: bool = False,
) -> Iterator[tuple[str, list[int]]]:
	"""Yield chunks that continue onto later pages until the word limit is met."""
	words: list[str] = []
	pages: list[int] = []

	for page_number, page_text in iter_page_text(
		pdf_path,
		use_ocr=use_ocr,
		google_ocr=google_ocr,
	):
		page_words = page_text.split()
		page_position = 0
		while page_position < len(page_words):
			remaining = words_per_chunk - len(words)
			words.extend(page_words[page_position : page_position + remaining])
			if not pages or pages[-1] != page_number:
				pages.append(page_number)
			page_position += remaining

			if len(words) == words_per_chunk:
				yield " ".join(words), pages.copy()
				words.clear()
				pages.clear()

	if words:
		yield " ".join(words), pages.copy()


def iter_pdf_files(input_path: Path) -> Iterable[Path]:
	if input_path.is_file() and input_path.suffix.lower() == ".pdf":
		yield input_path
		return
	if input_path.is_dir():
		yield from sorted(path for path in input_path.rglob("*.pdf") if path.is_file())


def build_chroma_client() -> Any:
	"""Create the same Chroma Cloud connection used by the API."""
	missing = [
		name
		for name in ("CHROMADB_KEY", "CHROMADB_TENANT")
		if not os.getenv(name)
	]
	if missing:
		raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
	return chromadb.CloudClient(
		api_key=os.environ["CHROMADB_KEY"],
		tenant=os.environ["CHROMADB_TENANT"],
		database=os.getenv("CHROMADB_DATABASE", "prod"),
	)


def make_id(pdf_path: Path, pages: list[int], chunk: str) -> str:
	identity = f"{pdf_path.resolve()}:{pages}:{chunk}".encode("utf-8")
	return hashlib.sha256(identity).hexdigest()


def add_book_context(pdf_path: Path, chunk: str) -> str:
	return f"Book: {pdf_path.stem}\n{chunk}"


def index_pdf(
	pdf_path: Path,
	collection: Any,
	openai_client: OpenAI,
	embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
	use_ocr: bool = False,
	google_ocr: bool = False,
) -> int:
	"""Index one PDF with a bounded chunk and embedding batch."""
	pending: list[tuple[str, list[int], str]] = []
	indexed_count = 0

	def flush() -> None:
		nonlocal indexed_count
		if not pending:
			return
		chunks = [item[0] for item in pending]
		response = openai_client.embeddings.create(
			model=EMBEDDING_MODEL,
			input=chunks,
		)
		collection.upsert(
			ids=[item[2] for item in pending],
			embeddings=[item.embedding for item in response.data],
			documents=chunks,
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

	for chunk, pages in iter_chunks(
		pdf_path,
		use_ocr=use_ocr,
		google_ocr=google_ocr,
	):
		indexed_chunk = add_book_context(pdf_path, chunk)
		pending.append(
			(indexed_chunk, pages, make_id(pdf_path, pages, chunk))
		)
		if len(pending) >= embedding_batch_size:
			flush()
	flush()
	return indexed_count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Index Arabic/English PDFs in ChromaDB.")
	parser.add_argument("input", type=Path, help="A PDF file or a directory containing PDFs")
	parser.add_argument("--collection", default=DEFAULT_COLLECTION)
	parser.add_argument("--env-file", type=Path, default=Path("minimal.env"))
	parser.add_argument(
		"--ocr",
		action="store_true",
		help=argparse.SUPPRESS,
	)
	parser.add_argument(
		"--google-ocr",
		action="store_true",
		help="Use Google Cloud Vision OCR for every page",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	load_dotenv(args.env_file)
	configure_credentials()
	if not os.getenv("OPENAI_API_KEY"):
		raise RuntimeError("Missing environment variable: OPENAI_API_KEY")

	pdf_files = list(iter_pdf_files(args.input))
	if not pdf_files:
		raise FileNotFoundError(f"No PDF files found at {args.input}")

	collection = build_chroma_client().get_or_create_collection(args.collection)
	openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
	total_chunks = 0
	for pdf_path in pdf_files:
		count = index_pdf(
			pdf_path,
			collection,
			openai_client,
			use_ocr=False,
			google_ocr=args.google_ocr,
		)
		total_chunks += count
		print(f"Indexed {count} chunks from {pdf_path}")
	print(f"Indexed {total_chunks} chunks into '{args.collection}'")


if __name__ == "__main__":
	main()
