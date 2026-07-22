#!/usr/bin/env python3
"""Sync the CoordinationOS transcript share (S3, read-only) into the local
canonical mirror the pipeline reads from: data/vault/ (index.csv + transcripts/).

Mirrors exactly (deletes locally-removed keys), and reports what changed so the
orchestrator knows which transcripts to (re)process and which stale outputs to
drop.

Prints a JSON summary to stdout: {"added": [...ids], "changed": [...ids],
"removed": [...ids]} where ids are index.csv transcript ids.
"""
import csv
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.join(HERE, "data", "vault")


def load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def id_by_path(index_csv_text):
    """map transcripts/<path> -> transcript id, from an index.csv string."""
    out = {}
    for r in csv.DictReader(io.StringIO(index_csv_text)):
        out[r["path"]] = r["id"]
    return out


def main():
    load_env()
    import boto3
    bucket = os.environ["S3_TRANSCRIPT_BUCKET"]
    prefix = os.environ.get("S3_TRANSCRIPT_PREFIX", "latest/")
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    os.makedirs(VAULT, exist_ok=True)

    # old id->path from the current local index (if any)
    old_index_path = os.path.join(VAULT, "index.csv")
    old_map = {}
    if os.path.exists(old_index_path):
        old_map = id_by_path(open(old_index_path).read())

    # list remote objects
    remote = {}
    pg = s3.get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            remote[o["Key"]] = o["ETag"].strip('"')

    def local_path(key):
        return os.path.join(VAULT, key[len(prefix):])

    def local_etag(path):
        if not os.path.exists(path):
            return None
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    downloaded, changed_keys = 0, []
    for key, etag in remote.items():
        lp = local_path(key)
        if "-" in etag or local_etag(lp) != etag:  # multipart etag or mismatch → fetch
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            s3.download_file(bucket, key, lp)
            downloaded += 1
            changed_keys.append(key)

    # delete local files not in remote (mirror)
    removed_keys = []
    for root, _, files in os.walk(os.path.join(VAULT, "transcripts")):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, VAULT)   # "transcripts/..."
            if (prefix + rel) not in remote:
                os.remove(fp)
                removed_keys.append(prefix + rel)

    # compute added/changed/removed transcript ids via the new index
    new_map = id_by_path(open(old_index_path).read()) if os.path.exists(old_index_path) else {}
    old_ids = set(old_map.values())
    new_ids = set(new_map.values())
    # changed transcript files (excluding index.csv)
    changed_ids = set()
    for key in changed_keys:
        rel = key[len(prefix):]
        if rel.startswith("transcripts/"):
            path = rel  # "transcripts/..."
            tid = new_map.get(path)
            if tid:
                changed_ids.add(tid)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    changed = sorted((changed_ids & old_ids) - set(added))

    summary = {"downloaded_objects": downloaded, "deleted_objects": len(removed_keys),
               "added": added, "changed": changed, "removed": removed}
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
