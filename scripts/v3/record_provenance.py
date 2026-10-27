"""Bind completed original-source verification to the existing license registry.

Offline ingestion only: no downloads, grouping changes, license choice or approval.
The existing audited map fingerprints must match. Re-run packaging to check the
actual source/render bytes before staging; this helper checks the recorded evidence.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from material_classifier.v3.data import (  # noqa: E402
    _fingerprint,
    _no_private_paths,
    json_bytes,
    read_csv,
    relative_path,
    sha256_file,
)


def _verified_map(record: dict) -> dict:
    """Check the comparison values, not just a report's pass flag."""
    relative_path(record["path"])
    matches = record.get("official_matches", [])
    if not matches and "official_file" in record:
        official = record["official_file"]
        matches = [dict(
            url=official["url"], official_md5=official["md5"],
            official_size_bytes=official["size"],
        )]
    valid = []
    for match in matches:
        size_ok = match.get("official_size_bytes") == record["size_bytes"]
        if record["verification_status"] == "official_md5_match":
            hash_ok = match.get("official_md5") == record["local_md5"]
        elif record["verification_status"] == "official_archive_sha256_match":
            hash_ok = match.get("official_member_sha256_computed") == record["local_sha256"]
        else:
            hash_ok = False
        if size_ok and hash_ok:
            valid.append(match)
    if not valid:
        raise ValueError(f"no matching official bytes: {record['path']}")
    # One matching official file/member is sufficient; API formats often repeat it.
    return dict(
        path=record["path"], sha256=record["local_sha256"],
        md5=record["local_md5"], size_bytes=record["size_bytes"],
        verification_status=record["verification_status"], official_match=valid[0],
        manifest_referenced=record.get("manifest_referenced", True),
    )


def record_provenance(report_path: Path, audit_dir: Path, material_manifest: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("source verification is incomplete")
    # Historical CSV checkout line endings may differ; retain the report's byte pin.
    raw_manifest = material_manifest.read_bytes()
    manifest_hashes = {
        hashlib.sha256(raw_manifest).hexdigest(),
        hashlib.sha256(raw_manifest.replace(b"\r\n", b"\n")).hexdigest(),
    }
    if report["manifest"]["sha256"] not in manifest_hashes:
        raise ValueError("original material manifest changed")
    registry_path = audit_dir / "license_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = copy.deepcopy(registry)
    maps = read_csv(audit_dir / "source_maps.csv")
    materials = {m["material_id"]: m for m in report["materials"]}
    ids = [e["material_id"] for e in registry["sources"]]
    if len(materials) != len(report["materials"]) or len(ids) != len(set(ids)):
        raise ValueError("duplicate materials")
    if set(ids) != set(materials) or set(ids) != {r["material_id"] for r in maps}:
        raise ValueError("source registry and verification scope differ")
    originals = {r["material_id"]: r for r in read_csv(material_manifest)}
    if set(originals) != set(ids):
        raise ValueError("original material manifest scope differs")
    evidence_files = {}
    for entry in registry["sources"]:
        mid = entry["material_id"]
        source = materials[mid]
        if not source.get("byte_verification_complete") or source["license"] != "CC0-1.0":
            raise ValueError(f"incomplete CC0 source evidence: {mid}")
        if source["source_material_name"] != entry["source_material_name"]:
            raise ValueError(f"source identity changed: {mid}")
        fingerprint = _fingerprint(maps, mid)
        if fingerprint != entry["source_fingerprint"]:
            raise ValueError(f"source fingerprint changed: {mid}")
        records = list(source["maps"])
        expected_paths = {
            f"materials/{originals[mid]['class']}/{mid}/{originals[mid][role]}"
            for role in ("base_color_file", "roughness_file", "normal_file", "metallic_file")
            if originals[mid].get(role)
        }
        if {m["path"] for m in records} != expected_paths:
            raise ValueError(f"referenced map scope differs: {mid}")
        records.extend(r for r in report.get("unreferenced_asset_verification", [])
                       if r["material_id"] == mid)
        verified = sorted((_verified_map(r) for r in records), key=lambda r: r["path"])
        actual = sorted((r["path"], r["sha256"], int(r["size_bytes"]))
                        for r in maps if r["material_id"] == mid)
        claimed = sorted((r["path"], r["sha256"], r["size_bytes"]) for r in verified)
        if actual != claimed:
            raise ValueError(f"verification does not cover exact audited maps: {mid}")
        provider_key = {"Poly Haven": "polyhaven", "ambientCG": "ambientcg"}[source["provider"]]
        license_evidence = report["license_and_provider_evidence"][provider_key]
        if license_evidence["url"] != source["license_evidence_url"]:
            raise ValueError(f"license evidence differs: {mid}")
        if source["provider"] == "Poly Haven":
            attribution = "; ".join(f"{name} ({role})" for name, role in source["author"].items())
            attribution += ". Publisher: Poly Haven."
        else:
            attribution = (
                "Publisher: ambientCG. Individual author not exposed by the official records; "
                "Lennart Demes is the provider operator, not asserted as this asset's author."
            )
        evidence = dict(
            schema_version=1, material_id=mid, source_fingerprint=fingerprint,
            verified_by="agent", human_review_claimed=False,
            original_material_manifest_sha256=report["manifest"]["sha256"],
            verification_report_sha256=sha256_file(report_path),
            verification_report="reports/results/v3/provenance_verification.json",
            provider=source["provider"], provider_asset_id=source["provider_asset_id"],
            source_url=source["source_url"], author=source["author"],
            author_status=source["author_status"], attribution=attribution,
            license_spdx=source["license"], license_url=source["license_url"],
            license_evidence=license_evidence, metadata_evidence=source["metadata_evidence"],
            verification_tier=source["verification_tier"], maps=verified,
            verification_tier_definition=report["verification_tiers"][source["verification_tier"]],
            evidence_storage="Compact verification record; full provider response bodies omitted.",
        )
        if provider_key == "polyhaven":
            evidence["official_file_metadata_evidence"] = source["official_file_metadata_evidence"]
        else:
            evidence["archive_verification_evidence"] = {
                k: v for k, v in source["archive_verification_evidence"].items()
                if k not in ("members", "license_members")
            }
            evidence["operator_evidence"] = report["license_and_provider_evidence"][
                "ambientcg_operator"
            ]
        for record in records:
            if record.get("manifest_referenced") is False:
                evidence["extra_map_metadata_evidence"] = record["metadata_evidence"]
        payload = json_bytes(evidence)
        _no_private_paths(payload.decode())
        evidence_path = f"license_evidence/{mid}.json"
        evidence_files[evidence_path] = payload
        entry.update(
            verification_status="verified", verified_by="agent", human_review_claimed=False,
            license_spdx=source["license"], license_url=source["license_url"],
            redistribution_allowed=True, source_url=source["source_url"],
            provider=source["provider"], provider_asset_id=source["provider_asset_id"],
            attribution=attribution, evidence_url=source["license_evidence_url"],
            evidence_path=evidence_path, evidence_sha256=hashlib.sha256(payload).hexdigest(),
            retrieved_at=source["retrieval_date"], verification_tier=source["verification_tier"],
        )
    # Ingesting evidence never carries approval forward or chooses the owner's license.
    registry.update(dataset_license="", release_approved=False)
    _no_private_paths(registry)
    for relative, payload in evidence_files.items():
        path = audit_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    registry_path.write_bytes(json_bytes(registry))
    return dict(verified_sources=len(ids), verified_maps=len(maps),
                dataset_license="", release_approved=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=REPO / "reports/results/v3/provenance_verification.json")
    parser.add_argument("--audit-dir", type=Path, default=REPO / "metadata/v3")
    parser.add_argument("--material-manifest", type=Path,
                        default=REPO / "metadata/material_manifest.csv")
    args = parser.parse_args()
    result = record_provenance(args.report, args.audit_dir, args.material_manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
