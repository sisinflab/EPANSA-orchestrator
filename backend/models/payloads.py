# =============================================================================
# Pydantic Payload Models
# -----------------------------------------------------------------------------
# Payloads exchanged between:
#   - client routes (FastAPI),
#   - Google sync services (Drive/Photos/Calendar/Contacts),
#   - the LLM extractor,
#   - and the graph store.

# =============================================================================

from __future__ import annotations
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field, constr

# Basic ISO-like constraints; intentionally permissive to avoid over-validation.
ISOTime = constr(strip_whitespace=True, min_length=5)        # "HH:MM[:SS]"
ISODateTime = constr(strip_whitespace=True, min_length=8)    # "YYYY-MM-DDTHH:MM[:SS[.fff]]"


class DeletePayload(BaseModel):
    """Generic delete envelope used across entity types."""
    id: str
    source_app: str
    metadata : Optional[dict] = None

class NoteMetadata(BaseModel):
    """Normalized note metadata.
    - Timestamps are ISO strings passed through without normalization here.
    - titles are optional and should remain concise"""
    creation_date: str = None
    modified_date: str = None
    creation_time: Optional[ISOTime] = None
    modified_time: Optional[ISOTime] = None
    title: Optional[str] = None

class NotePayloadLLM(BaseModel):
    """
    Note payload structure.

    """
    note: str
    metadata: Optional[NoteMetadata] = None
    content: str = None
    source_app: Optional[str] = None


class NotePayload(BaseModel):
    """
    Free-form note payload (e.g., Google Keep, Apple Notes).

    Notes
    -----
    - `text` is required; titles are optional and should remain concise.
    - Timestamps are ISO strings passed through without normalization here.
    """
    note: str
    title: Optional[str] = None
    text: str
    date_created: Optional[ISODateTime] = None
    date_modified: Optional[ISODateTime] = None
    source_app: Optional[str] = None
    kind: str = "note"
    
    
    def to_llm_payload(self) -> NotePayloadLLM:
        """Convert to LLM-focused payload structure."""
        metadata = NoteMetadata(
            creation_date=self.date_created.split("T")[0] if self.date_created else None,
            creation_time=self.date_created.split("T")[1] if self.date_created and "T" in self.date_created else None,
            modified_date=self.date_modified.split("T")[0] if self.date_modified else None,
            modified_time=self.date_modified.split("T")[1] if self.date_modified and "T" in self.date_modified else None,
            title=self.title
        )
        return NotePayloadLLM(
            note=self.note,
            metadata=metadata,
            content=self.text,
            source_app=self.source_app
        )


class ContactMetadata(BaseModel):
    """Contact metadata normalized for downstream processing."""
    name: str
    telephone_number: str


class ContactPayload(BaseModel):
    """
    Standardized contact structure.

    Storage/Files
    -------------
    - Validates objects coming from `contact_*.txt` files.
    - Uses alias `contact` to bind the `id` field to on-disk key names.
    """
    contact: str = Field(..., alias="contact")
    source_app: str
    metadata: ContactMetadata


class SingleEventMetadata(BaseModel):
    """Metadata for a single-occurrence calendar event."""
    label: str
    date: str
    start_time: ISOTime
    end_time: ISOTime


class RecurrentEventMetadata(BaseModel):
    """
    Metadata for a recurrent calendar event.

    Notes
    -----
    - `repeat_frequency` carries the recurrence rule (RRULE string or similar).
    - `on` is a free-form descriptor (e.g., day names) aligned with upstream text.
    """
    label: str
    start_time: ISOTime
    end_time: ISOTime
    repeat_frequency: str
    on: str = Field(..., alias="on")


class EventPayload(BaseModel):
    """
    Canonical event structure (single or recurrent).

    Storage/Files
    -------------
    - Validates objects from `event_*.txt` files.
    - `recurrence_info` should be either "single-occurrence" or "recurrent".
    """
    event: str = Field(..., alias="event")
    source_app: str
    recurrence_info: str
    metadata: Union[SingleEventMetadata, RecurrentEventMetadata]


class PhoneCallMetadata(BaseModel):
    """Normalized phone call details."""
    date: str
    start_time: ISOTime
    end_time: ISOTime
    duration: str
    call_direction: Literal["incoming", "outgoing", "missed"]
    with_contact: Optional[str] = None


class PhoneCallPayload(BaseModel):
    """
    Canonical phone call structure.

    Storage/Files
    -------------
    - Validates objects coming from `phoneCall_*.txt` files.
    """
    call: str = Field(..., alias="call")
    source_app: str
    metadata: PhoneCallMetadata




class SingleAlarmMetadata(BaseModel):
    """Alarm metadata for a one-time alarm."""
    label: str
    date: str            # expected input like "YYYY-MM-DD" (will be rendered as "DD-MMM-YYYY")
    time: ISOTime        # "HH:MM"


class RecurrentAlarmMetadata(BaseModel):
    """Alarm metadata for a recurring alarm."""
    label: str
    time: ISOTime
    repeat_frequency: str   # "daily" | "weekly" | "monthly" | "yearly" (free text tolerated)
    on: str                 # e.g., "MO,TU,WE" or "Monday, Tuesday", "15", "3 TU", "11-Sep", etc.


class AlarmPayload(BaseModel):
    """
    Canonical alarm structure (supports single and recurrent alarms).

    Notes
    -----
    - `recurrence_type` controls which metadata variant is active.
    - `to_natural_language_dict()` formats data into the textual structure
      expected by the downstream extractor (matches sample files exactly).
    """
    alarm: str = Field(..., alias="alarm")
    source_app: str
    recurrence_type: Literal["single-occurrence", "recurrent"] = Field(..., alias="recurrence_type")
    metadata: Union[SingleAlarmMetadata, RecurrentAlarmMetadata]

    @staticmethod
    def _month_abbr(n: int) -> str:
        return ["January","February","March","April","May","June","July","August","September","October","November","December"][n-1]

    @staticmethod
    def _ordinal(n_str: str) -> str:
        """Turn '15' into '15th', '1' -> '1st', etc. Falls back to input on errors."""
        try:
            n = int(n_str)
            if 10 <= n % 100 <= 20:
                suf = "th"
            else:
                suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            return f"{n}{suf}"
        except Exception:
            return n_str

    @staticmethod
    def _weekday_map() -> dict:
        return {"MO": "Monday","TU": "Tuesday","WE": "Wednesday","TH": "Thursday","FR": "Friday","SA": "Saturday","SU": "Sunday"}

    @classmethod
    def _normalize_on(cls, freq: str, on_val: str) -> str:
        """
        Normalize `on` to match samples:
          - weekly: 'MO,TU' or 'Monday,Tuesday' -> 'Monday, Tuesday'
          - monthly (day-of-month): '15' -> '15th'
          - monthly (nth weekday): '3 TU' -> '3rd Tuesday', '-1 FR' -> 'last Friday'
          - yearly: 'YYYY-MM-DD' or 'MM-DD' -> 'DD-MMM'
        Otherwise passes through unchanged.
        """
        if not on_val:
            return ""

        freq_l = (freq or "").strip().lower()

        # Weekly
        if freq_l == "weekly":
            val = on_val.replace(" ", "")
            if "," in val and all(x in cls._weekday_map() for x in val.split(",")):
                names = [cls._weekday_map()[x] for x in val.split(",") if x]
                return ", ".join(names)
            # Already names?
            parts = [p.strip().capitalize() for p in on_val.split(",") if p.strip()]
            # Normalize capitalization of weekday names
            weekday_names = set(cls._weekday_map().values())
            if all(p in weekday_names for p in parts):
                return ", ".join(parts)
            return on_val  # passthrough

        # Monthly
        if freq_l == "monthly":
            s = on_val.strip()
            # '15' -> '15th'
            if s.isdigit():
                return cls._ordinal(s)
            # '3 TU' or '-1 FR'
            tokens = s.split()
            if len(tokens) == 2:
                pos, wd = tokens
                wd_name = cls._weekday_map().get(wd.upper(), wd)
                if pos.lstrip("-").isdigit():
                    if pos == "-1":
                        return f"last {wd_name}"
                    # ordinal position
                    return f"{cls._ordinal(pos)} {wd_name}"
            return on_val

        # Yearly: allow YYYY-MM-DD or MM-DD -> DD-MMM
        if freq_l == "yearly":
            s = on_val.strip()
            try:
                if len(s) == 10 and s[4] == "-" and s[7] == "-":
                    y, m, d = map(int, s.split("-"))
                    return f"{int(d):02d}-{cls._month_abbr(int(m))}"
                if len(s) == 5 and s[2] == "-":
                    m, d = map(int, s.split("-"))
                    return f"{int(d):02d}-{cls._month_abbr(int(m))}"
            except Exception:
                pass
            return s

        return on_val

    def to_natural_language_dict(self) -> dict:
        """
        Render the payload as a natural-language-like dictionary that matches
        the alarm samples exactly (including 'reurrence_type' key).
        """
        # Determine the outward-facing "alarm" key:
        # - recurrent -> "recurrentAlarm_<id>"
        # - single    -> "alarm_<id>"
        raw_id = self.alarm
        is_recurrent = self.recurrence_type == "recurrent"

        if is_recurrent:
            alarm_key = raw_id if raw_id.startswith("recurrentAlarm_") else f"recurrentAlarm_{raw_id.lstrip('alarm_')}"
        else:
            alarm_key = raw_id if raw_id.startswith("alarm_") else f"alarm_{raw_id.lstrip('recurrentAlarm_')}"

        meta_out = {"label": self.metadata.label}

        if is_recurrent:
            # Normalize "on" a bit to approach the sample style.
            rf = getattr(self.metadata, "repeat_frequency", "") or ""
            on_raw = getattr(self.metadata, "on", "") or ""
            time_val = getattr(self.metadata, "time", "") or ""
            meta_out["time"] = time_val
            meta_out["repeat_frequency"] = rf
            meta_out["on"] = self._normalize_on(rf, on_raw)
        else:
            # Convert YYYY-MM-DD to DD-MMM-YYYY
            date_val = getattr(self.metadata, "date", "") or ""
            out_date = date_val
            try:
                if date_val and "-" in date_val and len(date_val.split("-")[0]) == 4:
                    y, m, d = map(int, date_val.split("-"))
                    out_date = f"{int(d):02d}-{self._month_abbr(int(m))}-{y}"
            except Exception:
                pass
            meta_out["date"] = out_date
            meta_out["time"] = getattr(self.metadata, "time", "") or ""


        return {
            "source_app": "alarm",
            "alarm": alarm_key,
            "reurrence_type": "recurrent" if is_recurrent else "single-occurrence",
            "metadata": meta_out,
        }



class ChatPayload(BaseModel):
    """Plain text chat command/request."""
    text: str
    
class ChatResponse(BaseModel):
    """Response from chat command processing."""
    response: Optional[str] = None
    action: Optional[str] = None
    action_data: Optional[dict] = None
    error: Optional[str] = None
    
# --- Photos -------------------------------------------------------------------

class PhotoMetadata(BaseModel):
    """Minimal photo metadata; binary path is provided at routing time if needed."""
    path: str
    creation_date: str
    creation_time: str
    location: Optional[str] = None


class PhotoPayload(BaseModel):
    """
    Canonical photo structure.

    Storage/Files
    -------------
    - Validates objects coming from `photo_*.txt` files.
    - Binary image path (if any) is attached by the router layer.
    """
    photo: str = Field(..., alias="photo")
    source_app: str
    metadata: PhotoMetadata

