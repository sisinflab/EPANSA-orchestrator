import logging
from sqlalchemy.orm import Session
from sqlalchemy import exists
from backend.services.secure_store import SessionLocal, ChatMessage
from datetime import datetime

def get_temporal_information():
    date_raw = datetime.now()
    day_of_week = date_raw.strftime("%A")
    date = date_raw.strftime("%d %B %Y")
    time = date_raw.strftime("%H:%M")
    return day_of_week, date, time

def add_message(user_id: int, role: str, content: str):
    """
    Adds a new message to the user's chat history.

    Args:
        user_id (int): The internal integer ID of the user.
        role (str): The role of the message sender ('user' or 'assistant').
        content (str): The text content of the message.
    """
    db: Session = SessionLocal()
    try:
        is_first_message = not db.query(
            exists().where(ChatMessage.user_id == user_id)
        ).scalar()

        if is_first_message:
            day_of_week, date, time = get_temporal_information()
            temporal_sys_prompt = f"""TEMPORAL CONTEXT: CURRENT DAY OF WEEK: {day_of_week}, CURRENT DATE: {date}, CURRENT TIME: {time}.
            Use these temporal details if the question requires time-awareness, otherwise ignore them."""

            new_message = ChatMessage(
                user_id=user_id,
                role="system",
                # content=temporal_sys_prompt,

                # FOR LOCAL TESTING PURPOSES ONLY
                content="TEMPORAL CONTEXT: CURRENT DAY OF WEEK: Monday, CURRENT DATE: 1 September 2025, CURRENT TIME: 13:00. Use these temporal details if the question requires time-awareness, otherwise ignore them."
            )
            db.add(new_message)
            db.commit()

        new_message = ChatMessage(
            user_id=user_id,
            role=role,
            content=content
        )
        db.add(new_message)
        db.commit()
        logging.info(f"Saved chat message for user_id={user_id}, role={role}")
    except Exception as e:
        db.rollback()
        logging.error(f"Failed to save chat message for user_id={user_id}: {e}")
        raise
    finally:
        db.close()


def get_history(user_id: int, limit: int = 50):
    """
    Retrieves the recent chat history for a user.

    Args:
        user_id (int): The internal integer ID of the user.
        limit (int): The maximum number of recent messages to retrieve.
    """
    db: Session = SessionLocal()
    try:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.timestamp)
            .limit(limit)
            .all()
        )

        return [{"role": msg.role, "content": msg.content} for msg in messages]

    except Exception as e:
        logging.error(f"Failed to retrieve chat history for user_id={user_id}: {e}")
        return []
    finally:
        db.close()

def delete_history(user_id: int):
    from backend.services.conversation import agent_cache
    db: Session = SessionLocal()
    try:
        # Directly delete all chat messages for the user
        deleted_count = db.query(ChatMessage).filter(ChatMessage.user_id == user_id).delete()
        db.commit()
        agent_cache.delete_item(user_id)
        return {"ok": True, "message": f"Chat history reset. Deleted {deleted_count} messages."}
    except Exception as e:
        db.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        db.close()
