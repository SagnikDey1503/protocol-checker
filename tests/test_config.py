import os
import pytest
from unittest.mock import patch
from app.config import get_settings, Settings
from app.core.llm import get_llm, get_fast_llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def test_settings_load():
    """Verify that settings can load and contain the new fast provider settings."""
    settings = get_settings()
    assert hasattr(settings, "fast_llm_provider")
    assert hasattr(settings, "groq_requests_per_minute")
    assert settings.fast_llm_provider in ["groq", "gemini"]
    assert settings.groq_requests_per_minute > 0

def test_factory_instantiation():
    """Test that get_llm and get_fast_llm return the expected clients under mocked env."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "gemini",
        "FAST_LLM_PROVIDER": "groq",
        "GOOGLE_API_KEY": "AIzaSy_test_google_key",
        "GROQ_API_KEY": "gsk_test_groq_key"
    }):
        # Clear cache to reload settings
        get_settings.cache_clear()
        get_llm.cache_clear()
        get_fast_llm.cache_clear()

        llm = get_llm()
        assert isinstance(llm, ChatGoogleGenerativeAI) or hasattr(llm, "model")
        
        fast_llm = get_fast_llm()
        assert isinstance(fast_llm, ChatGroq) or hasattr(fast_llm, "model_name")

def test_database_url_fix():
    """Verify that postgres:// and postgresql:// URL schemes are correctly rewritten to postgresql+asyncpg://"""
    settings_pg = Settings(database_url="postgres://user:pass@host:5432/db", pinecone_api_key="test")
    assert settings_pg.database_url == "postgresql+asyncpg://user:pass@host:5432/db"

    settings_pgsql = Settings(database_url="postgresql://user:pass@host:5432/db", pinecone_api_key="test")
    assert settings_pgsql.database_url == "postgresql+asyncpg://user:pass@host:5432/db"

    settings_already_async = Settings(database_url="postgresql+asyncpg://user:pass@host:5432/db", pinecone_api_key="test")
    assert settings_already_async.database_url == "postgresql+asyncpg://user:pass@host:5432/db"
