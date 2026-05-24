import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.core.auth import create_access_token, decode_user_id, TokenValidationError
from app.dependencies import get_current_user
from app.main import app
from app.models.database import User


def test_create_and_decode_access_token():
    user_id = uuid.uuid4()
    token = create_access_token(str(user_id))
    assert decode_user_id(token) == user_id


def test_decode_invalid_token_raises():
    with pytest.raises(TokenValidationError):
        decode_user_id("not-a-valid-jwt")


@pytest.fixture
def client():
    return TestClient(app)


def test_protected_route_requires_auth(client):
    response = client.get("/api/v1/protocols")
    assert response.status_code == 401


def test_auth_me_with_valid_token(client):
    user_id = uuid.uuid4()
    dummy_user = User(
        id=user_id,
        email="test@example.com",
        full_name="Test User",
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    token = create_access_token(str(user_id))

    app.dependency_overrides[get_current_user] = lambda: dummy_user
    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["email"] == "test@example.com"
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_cors_preflight_includes_allow_origin(client):
    response = client.options(
        "/api/v1/protocols",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") is not None
