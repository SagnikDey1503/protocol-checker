"""JWT token helpers shared by HTTP and WebSocket auth."""

from __future__ import annotations

from uuid import UUID

from jose import JWTError, jwt

from app.config import get_settings


class TokenValidationError(Exception):
    """Raised when a JWT cannot be decoded or is missing required claims."""


def create_access_token(subject: str) -> str:
    """Generate a signed JWT for the given subject (user id)."""
    from datetime import datetime, timedelta, timezone

    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_user_id(token: str) -> UUID:
    """Decode a JWT and return the user id from the subject claim."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        subject = payload.get("sub")
        if not subject:
            raise TokenValidationError("Token missing subject claim")
        return UUID(str(subject))
    except (JWTError, ValueError) as exc:
        raise TokenValidationError("Invalid or expired token") from exc
