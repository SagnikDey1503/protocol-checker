import pytest
import uuid
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.dependencies import get_current_user
from app.models.database import User

@pytest.fixture
def client():
    return TestClient(app)

def test_auth_me_cache_headers(client):
    """Test that GET /api/v1/auth/me response contains no-cache headers."""
    # Create a dummy user with proper validation values
    dummy_user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
        is_active=True,
        created_at=datetime.now(timezone.utc)
    )
    
    # Override get_current_user dependency to return dummy user
    app.dependency_overrides[get_current_user] = lambda: dummy_user
    
    try:
        response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer dummy_token"})
        assert response.status_code == 200
        
        # Verify cache control headers
        headers = response.headers
        assert headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, max-age=0"
        assert headers.get("Pragma") == "no-cache"
        assert headers.get("Expires") == "0"
    finally:
        # Clear dependency overrides
        app.dependency_overrides.pop(get_current_user, None)
