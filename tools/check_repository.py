
"""Static, standard-library-only release checks. Does not import experiment modules."""
from pathlib import Path
import ast
import hashlib
import json
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    errors = []
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts and "__pycache__" not in p.parts]
    banned = {".pt", ".pth", ".pkl", ".npy", ".npz", ".edf", ".bdf", ".mat", ".zip", ".rar", ".log", ".pyc"}
    secrets = [r"gh[pousr]_[A-Za-z0-9]{25,}", r"github_pat_[A-Za-z0-9_]{30,}", r"AKIA[A-Z0-9]{16}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    for p in files:
        rel = p.relative_to(ROOT)
        if p.is_symlink() or p.suffix.lower() in banned or p.name == ".env" or p.name.startswith(".env."):
            errors.append(f"Disallowed release file: {rel}")
        if p.stat().st_size > 5_000_000:
            errors.append(f"Oversized release file: {rel}")
        text = p.read_text(encoding="utf-8")
        if any(re.search(pattern, text) for pattern in secrets):
            errors.append(f"Potential credential in {rel}; value suppressed")
        if re.search(r"/(?:home|Users)/[A-Za-z0-9_.-]+/", text):
            errors.append(f"Host-specific private path: {rel}")
        if p.suffix == ".py":
            try:
                ast.parse(text, filename=str(rel), feature_version=(3, 11))
            except SyntaxError as exc:
                errors.append(f"Python syntax: {rel}:{exc.lineno}")
        elif p.suffix == ".json":
            json.loads(text)
    manifest = json.loads((ROOT/"docs/source_manifest.json").read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/"src").glob("*.py")}
    if actual != {row["file"] for row in manifest}:
        errors.append("Source inventory differs from manifest")
    for row in manifest:
        if digest(ROOT/row["file"]) != row["published_sha256"]:
            errors.append(f"Source checksum mismatch: {row['file']}")
    for row in json.loads((ROOT/"reports/evidence_manifest.json").read_text()):
        if digest(ROOT/row["path"]) != row["sha256"]:
            errors.append(f"Evidence checksum mismatch: {row['path']}")
    for row in json.loads((ROOT/"docs/protocol_manifest.json").read_text()):
        if digest(ROOT/row["path"]) != row["published_sha256"]:
            errors.append(f"Protocol checksum mismatch: {row['path']}")
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("href", "src") and value and not urlsplit(value).scheme and not value.startswith("#"):
                    path = self.base / unquote(urlsplit(value).path)
                    if not path.exists():
                        errors.append(f"Missing HTML link: {self.file}: {value}")
    for p in files:
        if p.suffix == ".html":
            parser = Links()
            parser.base, parser.file = p.parent, p.relative_to(ROOT)
            parser.feed(p.read_text())
    readme=(ROOT/"README.md").read_text()
    for link in re.findall(r"\]\(([^)]+)\)", readme):
        if not urlsplit(link).scheme and not (ROOT/link).exists():
            errors.append(f"Missing README link: {link}")
    status=json.loads((ROOT/"docs/release_status.json").read_text())
    if status["research_status"] != "PAUSED_BY_USER":
        errors.append("Research pause status was not preserved")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(dict(status="PASS",source_files=len(manifest),release_files=len(files),
        release_bytes=sum(p.stat().st_size for p in files),syntax="Python3.11",
        source_and_evidence_hashes=True,local_document_links=True,
        credential_pattern_scan=True,no_experiment_import_or_execution=True),indent=2))
if __name__ == "__main__":
    main()
