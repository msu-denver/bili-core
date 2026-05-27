# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

bili-core uses GitHub's Private Vulnerability Reporting (PVR) for confidential disclosure.

### How to report

- Go to the [Security tab](https://github.com/msu-denver/bili-core/security/advisories/new) of this repository and open a new private advisory.
- Alternatively, email `dpittma8@msudenver.edu` with the subject line `bili-core security report`.

### What to include

- A description of the issue and its potential impact (which component, what an attacker could achieve).
- Steps to reproduce, ideally with a minimal proof-of-concept.
- The version of bili-core where you observed the issue (release tag, branch, or commit SHA).
- Any suggested mitigation if you have one.

### What to expect

- Acknowledgement within 5 business days.
- A coordinated disclosure timeline once the maintainers have triaged the report. We aim to ship a fix within 30 days for high-severity issues; lower-severity issues may be batched into the next release.
- Credit in the release notes for the fix, if you would like to be acknowledged.

## Supported Versions

bili-core is research software; security fixes ship on the current major version line. Versions older than the current major are not supported.

| Version | Supported |
|---------|-----------|
| 5.x     | ✓         |
| < 5.0   | ✗         |

## Scope

This policy covers the bili-core framework code in this repository:

- IRIS, AETHER, and AEGIS components and their supporting infrastructure (`bili/`)
- Authentication and authorization paths (`bili/auth/`, `bili/flask_api/`)
- The Streamlit and Flask interfaces
- CI/CD workflows and build / packaging scripts
- Container build configuration

The following are explicitly out of scope:

- Vulnerabilities in upstream dependencies that are tracked by Dependabot and have not yet been triaged. Please report those to the upstream project.
- Adversarial behavior of LLM models invoked by bili-core. AEGIS exists specifically to study LLM adversarial robustness; reports of model misbehavior are research findings rather than vulnerabilities in this framework.
- Issues that require physical or root access to the host running bili-core.
- Misconfiguration of downstream deployments using bili-core as a library. Please report those to the deployment owner.

## Threat Model

bili-core is intended to run in research or evaluation environments. Production deployments must implement their own controls for credential management, network egress, and tenant isolation. The framework provides primitives; the deployment is responsible for hardening.
