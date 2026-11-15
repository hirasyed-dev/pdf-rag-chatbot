from fastapi.testclient import TestClient

from app.main import app
from tests.helpers import PAGE_ONE, PAGE_TWO, make_pdf

client = TestClient(app)


def upload(path, name="report.pdf"):
    with open(path, "rb") as f:
        return client.post("/documents", files={"file": (name, f, "application/pdf")})


def test_health():
    assert client.get("/health").json() == {"status": "healthy"}


def test_rejects_non_pdf(isolated):
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_rejects_scanned_pdf(isolated):
    import pymupdf

    path = isolated / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    response = upload(path, "blank.pdf")
    assert response.status_code == 400
    assert "scanned" in response.json()["detail"].lower()


def test_upload_search_delete_flow(isolated):
    path = isolated / "report.pdf"
    make_pdf(path, [PAGE_ONE, PAGE_TWO])

    first = upload(path)
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "indexed"
    assert body["page_count"] == 2
    doc_id = body["doc_id"]

    # same file again is not indexed twice
    assert upload(path).json()["status"] == "already_indexed"

    listing = client.get("/documents").json()["documents"]
    assert [d["doc_id"] for d in listing] == [doc_id]

    result = client.get("/search", params={"q": "quartz jazz quiz", "top_k": 2}).json()
    assert result["results"][0]["page_number"] == 1
    assert result["results"][0]["filename"] == "report.pdf"
    assert "took_ms" in result

    only_page_two = client.get("/search", params={"q": "quartz", "page": 2}).json()
    assert all(r["page_number"] == 2 for r in only_page_two["results"])

    deleted = client.delete(f"/documents/{doc_id}")
    assert deleted.status_code == 200
    assert client.get("/documents").json()["documents"] == []
    assert client.get("/search", params={"q": "quartz"}).json()["results"] == []
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_index_folder_skips_duplicates_and_bad_files(isolated):
    library = isolated / "library"
    library.mkdir()
    make_pdf(library / "one.pdf", [PAGE_ONE])
    make_pdf(library / "two.pdf", [PAGE_TWO])
    (library / "notes.txt").write_text("ignored")

    first = client.post("/index-folder").json()
    assert [r["status"] for r in first["results"]] == ["indexed", "indexed"]

    second = client.post("/index-folder").json()
    assert [r["status"] for r in second["results"]] == ["already_indexed", "already_indexed"]


def test_search_needs_a_query():
    assert client.get("/search").status_code == 422
