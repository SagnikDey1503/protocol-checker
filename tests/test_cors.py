import os
from unittest.mock import patch

from app.config import Settings, get_settings


def test_resolved_cors_origins_adds_render_frontend():
    with patch.dict(
        os.environ,
        {
            "RENDER_EXTERNAL_URL": "https://protocol-backend-glbk.onrender.com",
            "PINECONE_API_KEY": "test-key",
        },
        clear=False,
    ):
        settings = Settings(
            cors_origins="http://localhost:5173",
            pinecone_api_key="test-key",
        )
        origins, allow_credentials = settings.resolved_cors_origins()

        assert "https://protocol-frontend-glbk.onrender.com" in origins
        assert allow_credentials is True


def test_resolved_cors_origins_wildcard_disables_credentials():
    settings = Settings(cors_origins="*", pinecone_api_key="test-key")
    origins, allow_credentials = settings.resolved_cors_origins()

    assert origins == ["*"]
    assert allow_credentials is False
