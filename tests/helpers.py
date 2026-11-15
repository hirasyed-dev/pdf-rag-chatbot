"""Helpers for building small test PDFs."""
import pymupdf


def make_pdf(path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox((50, 50, 550, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


PAGE_ONE = "quartz jazz exquisite quick quiz jinx " * 10
PAGE_TWO = "mom bob pop hip bubble pump mimic " * 10
