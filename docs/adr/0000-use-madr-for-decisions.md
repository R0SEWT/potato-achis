# ADR-0000: Use MADR for Architecture Decision Records

## Status

Accepted

## Date

2026-04-09

## Context

We need a lightweight way to document architecture decisions versioned alongside the code. Without recorded decisions, future contributors (and future selves) lose the *why* behind design choices.

## Decision

Use [MADR](https://adr.github.io/madr/) (Markdown Architectural Decision Records) stored in `docs/adr/`. Each record follows the format: Status, Date, Context, Decision, Consequences.

Numbering: `NNNN-slug.md` (zero-padded, sequential).

## Consequences

- All significant architecture decisions get a short markdown file.
- ADRs are created as decisions are made, not retroactively in batch — except for a few foundational decisions documented now.
- Superseded decisions keep their file with status changed to `Superseded by ADR-XXXX`.
