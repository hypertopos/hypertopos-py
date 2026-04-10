# Security

## Current version: 0.2.x

hypertopos runs locally. No network services, no auth layer, no multi-tenancy.

## What to watch for

- **Pickle files** — chain cache (`.cache/chains_*.pkl`) uses pickle. Don't load cache files from untrusted sources.
- **Sphere paths** — `HyperSphere.open(path)` reads from the local filesystem. Don't point it at user-controlled paths without validation.
- **MCP server** — communicates over stdio. Not designed for network exposure.

## Reporting

If you find a security issue: [GitHub private vulnerability reporting](https://github.com/hypertopos/hypertopos-py/security/advisories/new) or email [contact@hypertopos.com](mailto:contact@hypertopos.com).
