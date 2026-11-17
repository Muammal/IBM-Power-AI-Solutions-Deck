"""auth.py: bearer-token authentication with roles for the operator APIs.

Tokens are never stored in clear: tokens.yaml holds SHA-256 hashes.
    python -m vision.auth new-token ahmed reviewer      # prints a token once + the line to add to tokens.yaml
In production put the API behind your reverse proxy with TLS + SSO as well (defence in depth).
"""
import hashlib
import hmac
import pathlib
import secrets
import sys

import yaml
from fastapi import Header, HTTPException

ROLE_ORDER = {"viewer": 1, "operator": 2, "reviewer": 3, "admin": 4}


class TokenAuth:
    def __init__(self, tokens_file):
        self.path = pathlib.Path(tokens_file)
        self.users = {}
        if self.path.exists():
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self.users = {u["sha256"]: (u["name"], u["role"]) for u in data.get("users", [])}

    def identify(self, token):
        digest = hashlib.sha256(token.encode()).hexdigest()
        for h, user in self.users.items():
            if hmac.compare_digest(h, digest):
                return user
        return None

    def require(self, min_role):
        """FastAPI dependency: Depends(auth.require("reviewer")) -> (name, role)."""
        def dep(authorization: str = Header(default="")):
            token = authorization.removeprefix("Bearer ").strip()
            user = self.identify(token) if token else None
            if not user:
                raise HTTPException(401, "missing or invalid token")
            if ROLE_ORDER.get(user[1], 0) < ROLE_ORDER[min_role]:
                raise HTTPException(403, f"role {min_role} required")
            return user
        return dep


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "new-token":
    tok = secrets.token_urlsafe(32)
    print(f"token (give to the user, shown once): {tok}")
    print(f"tokens.yaml entry:\n  - name: {sys.argv[2]}\n    role: {sys.argv[3]}\n    sha256: {hashlib.sha256(tok.encode()).hexdigest()}")
