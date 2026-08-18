# EviBind 0.3.0 Release Audit

Status: passed locally on 2026-07-30.

This record describes the source state in the release checkpoint commit. It is
a local mirror of the maintained CI gates, not evidence that hosted CI,
provider integrations, or the preregistered powered experiment have run.

## Source Gates

- Python 3.13.11: 522 tests passed.
- Maintained product, evaluation, release, and test paths: Ruff passed.
- Grammar/conformance suite: 16 checks, zero failures.
- Strict example-schema lint: zero errors and zero warnings.
- Paper audit: 9 checks passed across 10 source-bound claims and 12 citations.
- EviBench v1 corpus digest:
  `8ab09c08e74d9ea639b83678ae1abdf4bf652cc61a7a788cdda653ec8598be59`.
- EviBench compiler audit: 10 true positives, one false positive, one false
  negative, and zero untrusted critical admissions.

## Paper Gate

- `latexmk` completed without LaTeX warnings, overfull boxes, undefined
  references, or undefined citations.
- The dependency-light preview is six letter-size pages.
- The powered extension remains
  `protocol_frozen_execution_pending`: four preregistered model slots, seven
  one-call conditions, at least 2,500 cases, and at least 50 tool families.

## Package Gate

- Built `evibind-0.3.0.tar.gz` and
  `evibind-0.3.0-py3-none-any.whl` from the source tree.
- Archive audit found no generated experiment paths, Python cache files,
  generated paper outputs, or machine-local absolute paths.
- The sdist contains the canonical paper sources, claim ledger, and frozen
  EviBench corpus; generated LaTeX files are explicitly excluded.
- A fresh isolated wheel install passed public API import, both CLI help
  surfaces, 16 conformance checks, strict schema lint, and packaged-corpus
  digest verification.

## Container Gate

- The image built from `python:3.12-slim`.
- Runtime UID is 10001, confirming the configured non-root user.
- The installed container package passed all 16 conformance checks.

## Explicitly Not Completed

- Hosted-provider live validation requiring credentials.
- Independent double human annotation and blinded adjudication.
- The powered, independently authored, multi-model EviBench extension.
- Multi-call planning, streaming, distributed nonce consumption, and
  long-horizon task-utility validation.

These remain external work and must not be inferred from this release audit.
