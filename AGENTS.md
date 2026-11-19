# Project instructions

## Project context

- Read `docs/spec.md` for confirmed scope and implementation defaults.
- Read `docs/track1-submission.md` for official Track 1 scoring and submission requirements. Keep organizer requirements distinct from internal experiment goals.
- Read `docs/development.md` for branch, integration, and review workflow.
- Read `DESIGN.md` before making UI or interaction changes.
- Prioritize a complete hackathon demo. Keep changes focused, preserve others' edits, and report validation gaps accurately.

## Git conventions

- Use concise English Conventional Commits: `<type>(<scope>): <summary>`, with an optional scope. Example: `docs: define branch review workflow`.
- The first commit must be exactly `Initial commit`; this is the exception to the Conventional Commit format.
- Do not add AI coauthor trailers or AI attribution signatures to commits.
- Keep two long-lived branches: `main` for reviewed, verified changes and `dev` for integration and teammate review.
- Create development branches from the latest `dev`, named `type/description-yyyymmdd`, for example `feat/turn-replay-20261119`. Target `dev` when integrating them.
- Promote `dev` to `main` only when the entire proposed diff has completed teammate review and relevant validation. If unreviewed changes remain in `dev`, postpone promotion.
- Bootstrap `main` with a reviewed `Initial commit`, then create `dev` at that commit. After bootstrap, update `main` only through reviewed merge commits; never force-push it.
- After a promotion, synchronize `main` back into `dev` before starting subsequent work.
- Merge with `--no-ff`. Do not squash merges.
- Keep unrelated changes out of the same commit and never revert another contributor's work without a task-specific reason.

## Shared documentation

- Keep the root README focused on the public project introduction, its problem, intended behavior, and verified implementation status. Put team schedules, environment planning, review workflow, and submission coordination under `docs/`, indexed by `docs/README.md`.
- `docs/` contains only material intended to be shared with teammates: specifications, architecture decisions, interfaces, runbooks, validation summaries, and team handoffs.
- When a team handoff is needed, use `docs/handoffs/<topic>.md`. Include only the context, deliverables, checks, unresolved issues, and next steps the team needs.
- Keep lasting design decisions and operating procedures in the existing shared documents and link to them from team handoffs rather than duplicating them.
- Do not put credentials, private source records, or raw private logs into shared documentation.
- Curated submission traces from actual runs belong in `docs/submission/logs/` as JSONL. Review the exported records before adding them to the repository.

## Git tracking boundaries

- Keep `AGENTS.md` and team documentation under `docs/` eligible for normal Git tracking.
- Exclude local secrets, downloaded datasets, model weights, and generated experiment output according to `.gitignore`.
