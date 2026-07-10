"""Short URL-safe unique IDs — Python port of src/lib/id.ts (nanoid 12 chars)."""
import secrets

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def new_id(size: int = 12) -> str:
    """Generate a short, URL-safe unique ID (default 12 chars, 62^12 space)."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(size))


# Drop-in alias matching the TS `id()` usage.
id = new_id
