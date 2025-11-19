import logging
from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError
from cryptography.fernet import Fernet, InvalidToken
from datetime import datetime

from backend.core.config import settings

# Configuration validation
if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")
if not settings.ENCRYPTION_KEY:
    raise ValueError(
        "ENCRYPTION_KEY environment variable is not set. "
        "Generate one with: python -c 'from cryptography.fernet import Fernet; "
        "print(Fernet.generate_key().decode())'"
    )

# Crypto setup
try:
    fernet = Fernet(settings.ENCRYPTION_KEY.encode())
except ValueError:
    # Most common cause: key is not a valid 32-byte URL-safe base64 string.
    raise ValueError(
        "Invalid ENCRYPTION_KEY. It must be a 32-byte URL-safe base64-encoded key."
    )

# SQLAlchemy bootstrap
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Data model for credentials
class UserCredential(Base):
    """
    Stores one encrypted refresh token per user.

    Columns
    --------
    user_id : str
        Unique subject identifier from the external provider.
    encrypted_refresh_token : str
        Encrypted Google refresh token (Fernet AES-128).
    updated_at : datetime
        Timestamp of the last update for audit tracking.
    """
    __tablename__ = "user_credentials"

    user_id = Column(String, primary_key=True, index=True)
    encrypted_refresh_token = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())


class User(Base):
    """
    Minimal user record persisted in the SQL store.

    Notes
    -----
    - `google_id` is the external stable identifier and is unique.
    - `email` is also unique; collisions should be prevented at ingestion time.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    name = Column(String)

    # Add relationship to chat messages
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")

# Data model for chat history
class ChatMessage(Base):
    """
    Stores one chat message for a user's conversation history.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now())

    # Establish relationship to the User model
    user = relationship("User", back_populates="chat_messages")


# Table creation helper
def create_credentials_table():
    """
    Idempotently create the credentials table.

    This is safe to call at startup; SQLAlchemy will only create the table
    if it does not already exist.
    """
    Base.metadata.create_all(bind=engine)
    logging.info("User credentials table created or already exists.")


# ---- Credentials service ----------------------------------------------------
class CredentialsService:
    """
    Service providing high-level access to encrypted refresh token storage.

    Responsibilities
    ----------------
    - Encrypt and store new refresh tokens.
    - Retrieve and decrypt stored tokens on demand.
    - Enumerate existing user records.
    """

    def save_refresh_token(self, user_id: str, refresh_token: str):
        """
        Encrypt and persist a refresh token for a user.

        Notes
        -----
        - If the user already has a stored token, it is overwritten.
        - Encryption uses Fernet symmetric crypto (AES + HMAC).
        """
        db = SessionLocal()
        try:
            encrypted_token = fernet.encrypt(refresh_token.encode()).decode()
            credential = (
                db.query(UserCredential)
                .filter(UserCredential.user_id == user_id)
                .first()
            )

            if credential:
                credential.encrypted_refresh_token = encrypted_token
            else:
                credential = UserCredential(
                    user_id=user_id,
                    encrypted_refresh_token=encrypted_token
                )
                db.add(credential)

            db.commit()
            logging.info(f"Successfully saved encrypted refresh token for user {user_id}.")
        except SQLAlchemyError as e:
            db.rollback()
            logging.error(f"Database error while saving token for user {user_id}: {e}")
            raise
        finally:
            db.close()

    def get_refresh_token(self, user_id: str) -> str | None:
        """
        Retrieve and decrypt the refresh token for the specified user.

        Returns
        -------
        str | None
            Decrypted refresh token if available, otherwise None.

        Notes
        -----
        - Returns None if no record is found, if decryption fails, or if
          a database error occurs.
        """
        db = SessionLocal()
        try:
            credential = (
                db.query(UserCredential)
                .filter(UserCredential.user_id == user_id)
                .first()
            )
            if not credential:
                return None
            decrypted_token = fernet.decrypt(
                credential.encrypted_refresh_token.encode()
            ).decode()
            return decrypted_token
        except InvalidToken:
            logging.error(
                f"Failed to decrypt token for user {user_id}. "
                "The encryption key may have changed."
            )
            return None
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving token for user {user_id}: {e}")
            return None
        finally:
            db.close()

    def get_all_user_ids(self) -> list[str]:
        """
        Return a list of all user IDs that currently have stored credentials.

        Returns
        -------
        list[str]
            All user identifiers found in the credentials table.
        """
        db = SessionLocal()
        try:
            user_ids = db.query(UserCredential.user_id).all()
            return [uid for (uid,) in user_ids]
        except SQLAlchemyError as e:
            logging.error(f"Database error while fetching all user IDs: {e}")
            return []
        finally:
            db.close()


# Singleton instance used throughout the backend.
credentials_service = CredentialsService()
