# Contributing to bili-core

Thank you for your interest in contributing. bili-core is an open-source research framework, and contributions of all kinds are welcome: bug reports, fixes, new features, documentation improvements, and adversarial test payloads for AEGIS.

This document covers how to file issues, the contribution workflow, and a few rules that apply to anything that lands in the public repository.

## Reporting bugs and requesting features

- Use [GitHub Issues](https://github.com/msu-denver/bili-core/issues) for bug reports, feature requests, and questions.
- **Do not report security vulnerabilities in public issues.** Use the process described in [SECURITY.md](SECURITY.md) instead.
- When reporting a bug, include the version of bili-core (release tag, branch, or commit SHA), the relevant component (IRIS, AETHER, AEGIS, auth, Flask, Streamlit), and a minimal reproduction.

## Development workflow

bili-core uses a GitFlow-shaped branching model.

- The default branch is `develop`. Feature work targets `develop`.
- `main` only receives merges from `develop` as part of a tagged release.
- Feature branches use the prefix `feat/`, `fix/`, `chore/`, or `docs/` followed by a short descriptive slug (e.g. `feat/aegis-new-attack-suite`, `fix/checkpoint-race`).

### Setting up the development environment

The container-based development environment is the supported path. See the top-level [README.md](README.md) for the full setup walkthrough.

```bash
cd scripts/development
./start-container.sh
./attach-container.sh
```

Inside the container, `streamlit` starts the UI on port 8501 and `flask` starts the API on port 5001.

### Code quality

Before opening a pull request:

- Run `./run_python_formatters.sh` to apply Black, Autoflake, and Isort.
- Run `pylint bili/ --fail-under=9` to confirm the lint score is at least 9.0.
- Add tests for new functionality. Existing test conventions for each component live under `bili/<component>/tests/`.
- Use type hints throughout new code.

The pre-commit hooks enforce most of the above automatically.

### Opening a pull request

- Target `develop`, not `main`.
- Reference the issue your PR addresses, if applicable.
- Write a PR body that summarizes the change in 2–3 sentences and lists any user-visible behavior changes.
- The Claude code review workflow will run automatically on every push and post a review comment. Address the substantive findings before requesting human review.
- Conversations on PR comments must be marked as resolved before the PR is merged.

## Automated review

Three Claude-driven review workflows run on PRs in this repository:

- **Claude Code Review** runs on every PR push and posts a single review comment surfacing correctness, style, and small-scale design issues.
- **Claude Security Review** runs on PRs targeting `main` or `develop` and performs a deeper pass focused on security-relevant changes (input handling, authentication paths, secret handling, supply-chain regressions).
- **@claude bot** is available in PR and issue comments for follow-up questions and targeted fixes. The bot only responds to comments from users with write access to the repository.

The reviews are best treated as a strong first pass, not a substitute for human review of substantive changes. Address the substantive findings; a final summary line from the reviewer noting that remaining items are "low-severity wording quibbles, none block merge" is a reasonable place to stop iterating.

## No personal data in public code

bili-core is a public repository. Test fixtures, example configurations, comments, and documentation must not include real personal names, real institutional abbreviations beyond MSU Denver itself, or real email addresses.

Use these substitutions:

- Personal names: Alice Garcia, Bob Jones, Carol Lee, Maria A. Smith. Generic enough not to identify anyone.
- Institutional abbreviations in examples: ZX, YW, FOO, BAR.
- Email addresses: `@example.com` or `@example.edu` (RFC-reserved domains).

The discipline is to apply the substitution at the moment of writing the test, not after the fact. Force-pushing a feature branch to scrub a commit that already contains real names is possible but is best avoided.

## License

bili-core is licensed under the MIT License (see [LICENSE](LICENSE)). By contributing, you agree that your contributions are licensed under the same terms.
