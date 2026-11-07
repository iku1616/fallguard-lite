# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this repository
when it is available. Do not include camera frames, faces, access tokens, or
other personal data in a public issue.

## Security properties

- The application does not intentionally persist received frames.
- Browser frames are accepted only over a same-origin WebSocket by default.
- JPEG size and decoded pixel count are limited before inference.
- The provided container runs as an unprivileged user and Compose drops Linux
  capabilities, uses a read-only root filesystem, and binds only to localhost.
- Model files can be pinned with `MODEL_SHA256`; the supplied image creates and
  verifies a checksum sidecar for its bundled model.

This is a demonstration project, not a certified safety or medical system.
