"""Checkpoint downloads are atomic and pinned to the released file digests."""

import hashlib
import json
from pathlib import Path

import pytest

import scripts.download_checkpoints as downloader


def test_checkpoint_download_checks_digest_before_replacing_destination(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"checkpoint fixture")
    destination = tmp_path / "checkpoints/model.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert downloader.download(source.as_uri(), destination, digest) == destination
    assert destination.read_bytes() == source.read_bytes()


def test_checkpoint_download_rejects_corruption_and_preserves_existing_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupt content")
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.download(source.as_uri(), destination, "0" * 64)
    assert destination.read_bytes() == b"existing checkpoint"
    assert not destination.with_suffix(".pt.part").exists()


def test_release_download_flag_uses_manifest_urls_and_digests(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(downloader, "download", lambda url, path, sha256: calls.append((url, path.name, sha256)) or path)
    monkeypatch.setattr("sys.argv", ["download_checkpoints.py", "--gexco", "--output-dir", str(tmp_path)])
    downloader.main()
    manifest = json.loads((Path(__file__).parents[1] / "gexco/weights/release.json").read_text())
    assert calls == [(value["url"], name, value["sha256"]) for name, value in manifest["assets"].items()]
    assert {name for _, name, _ in calls} == {"gexco.pt", "world_model.pt"}
