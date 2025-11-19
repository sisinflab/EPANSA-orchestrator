import logging
from sqlalchemy.orm import Session
# Import the User model from its new location
from backend.services.secure_store import Base, SessionLocal, engine, User

def create_user_table():
    """
    Idempotently create the `users` table.

    This call is safe to execute multiple times; SQLAlchemy will only create
    missing tables and leave existing schemas intact.
    """
    # Ensure metadata (including models registered elsewhere) is reflected.
    Base.metadata.create_all(bind=engine)
    logging.info("Users table created or already exists.")


class UserService:
    """
    Service layer for user retrieval/creation.

    The public API is intentionally small to keep transactional semantics clear
    and avoid leaking ORM internals to callers.
    """

    def get_or_create_user_by_google_id(self, google_id: str, email: str, name: str) -> User:
        """
        Fetch an existing user by `google_id`; if not found, create a new one.
        ...
        """
        db: Session = SessionLocal()
        try:
            # Attempt to find an existing user by Google ID (login path).
            user = db.query(User).filter(User.google_id == google_id).first()

            if user:
                logging.info(f"User found with google_id={google_id}.")
                return user

            # Registration path: create and persist a new user record.
            logging.info(f"Creating new user with google_id={google_id}.")
            new_user = User(
                google_id=google_id,
                email=email,
                name=name,
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return new_user
        finally:
            # Always release session resources.
            db.close()

    def get_user_by_id(self, user_id: int) -> User | None:
        """

        Fetch a user by their primary key ID.
        """
        db: Session = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user
        finally:
            db.close()


user_service = UserService()