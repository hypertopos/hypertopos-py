# Security

## Current version: 0.9.0

hypertopos runs locally. No network services, no auth layer, no multi-tenancy.

## What to watch for

- **Sphere paths** — `HyperSphere.open(path)` reads from the local filesystem. Don't point it at user-controlled paths without validation.
- **MCP server** — communicates over stdio. Not designed for network exposure.
- **SQL injection protection** — `lance_sql_agg` validates user-controlled SQL inputs. Filter values are escaped (doubles single quotes, rejects backslash and ASCII control chars). Column identifiers are validated against `^[A-Za-z_][A-Za-z0-9_]*$`.

## Supported Versions

| Version | Supported |
|---|---|
| Latest 0.9.x | Yes |
| < 0.9 | No |

## Reporting

If you find a security issue: [GitHub private vulnerability reporting](https://github.com/hypertopos/hypertopos-py/security/advisories/new) or email [contact@hypertopos.com](mailto:contact@hypertopos.com).
