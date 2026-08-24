# Paper evidence releases

This directory is the bridge between the open-source implementation and the
paper artifact. It stores small, reviewable release records—not model responses,
generated papers, or large archives in Git.

Each JSON record pins the bundle filename, bundle and paper SHA-256 digests,
claim-audit counts, empirical scope, and exclusions. The archive itself should
be attached to a versioned release together with its `.sha256` sidecar. After
uploading, set `download_url` in the release record and verify the downloaded
copy while signed out.

Verify the current v8 release:

```bash
python scripts/verify_evidence_bundle.py \
  EviBind_ICLR_2027_evidence_bundle_20260821_v8.zip \
  --sidecar EviBind_ICLR_2027_evidence_bundle_20260821_v8.zip.sha256 \
  --release-metadata evidence/paper-v8.json
```

The verifier does not extract the archive. It checks path safety, a single
archive root, complete internal-manifest coverage, every member digest, the
canonical paper digest, and consistency between `ARTIFACT.json`, the paper
audit, and this repository's release record.
