"""
Authentication endpoints.
Handles user registration, login, and profile fetching.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Response
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.auth import create_access_token
from app.dependencies import get_db, get_current_user
from app.models.database import User
from app.models.schemas import UserCreate, UserLogin, UserResponse, TokenResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash the plain text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify standard plain password against stored hash."""
    return pwd_context.verify(plain_password, hashed_password)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Auth"],
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Register a new user account.
    """
    # Check if user already exists
    existing = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    # Create new user record
    new_user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        is_active=True,
    )

    db.add(new_user)
    await db.flush()
    await db.refresh(new_user)

    logger.info("New user registered successfully: %s", new_user.id)
    return new_user


@router.post("/login", response_model=TokenResponse, tags=["Auth"])
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Authenticate a user and return a JWT access token.
    """
    # Fetch user
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is inactive.",
        )

    # Generate token
    token = create_access_token(str(user.id))
    logger.info("User logged in successfully: %s", user.id)
    return TokenResponse(access_token=token, user=user)


@router.get("/me", response_model=UserResponse, tags=["Auth"])
async def get_me(
    response: Response,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the authenticated user's profile details.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return current_user
