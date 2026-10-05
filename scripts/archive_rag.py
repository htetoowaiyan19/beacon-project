"""Create and byte-verify a preservation archive before retiring retrieval files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["backend", "frontend", "variants", "outputs/chromadb", "outputs/uploads",
           "yolov8n.pt", "tests/test_pipeline.py", "tests/test_api.py", "tests/tests_support.py",
           "tests/test_variants_streaming.py", "tests/test_frontend_variant.cjs", "tests/test_stream_direct.py",
           "scripts/run_server.py", "run_server.bat", "requirements.txt", "requirements-inference.txt",
           "README.md", "API.md", "PROJECT_REVIEW.md", "TRAINING_GUIDE.md", "TRAINING_REPORT.md"]

def main():
    archive = ROOT / "archives/rag_5th_year_2026-10-04.zip"
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    files = {}
    for target in TARGETS:
        path = ROOT / target
        if not path.exists():
            continue
        for file in ([path] if path.is_file() else path.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                files[file.relative_to(ROOT).as_posix()] = file
    manifest = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            z.writestr(name, data)
        z.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, indent=2))
        z.writestr("RESTORE.md", """# Fifth-year retrieval project snapshot

Extract into a NEW project folder; do not overwrite the current model-only project.
Original paths are retained. This includes source code, both variants, old frontend,
tests, dependencies, uploaded documents, Chroma database/index and local YOLO weights.
Base Qwen weights, trained LoRA adapters, training datasets and virtual environment
are retained in the current project and intentionally not duplicated here. Copy those
if needed. Create a fresh environment and install variants/rag/requirements.txt;
for scanned PDFs also install variants/rag/requirements-ocr.txt. Run
python scripts/run_server.py --variant rag. Consult the archived README.md/API.md.
TrOCR microsoft/trocr-base-printed and Chroma's default embedding model may need
downloading on first use (external caches are not included). The archived OCR pipeline
has known Burmese recognition limitations; review PROJECT_REVIEW.md before reuse.
ARCHIVE_MANIFEST.json contains SHA-256 and sizes for every preserved original file.
""")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, "Corrupt ZIP"
        for entry in manifest:
            archived = z.read(entry["path"])
            assert hashlib.sha256(archived).hexdigest() == entry["sha256"]
            assert files[entry["path"]].read_bytes() == archived, "Source changed during archive"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Verified {len(manifest)} files; {archive.stat().st_size:,} bytes: {archive}")

if __name__ == "__main__":
    main()
