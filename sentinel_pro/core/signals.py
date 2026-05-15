import logging
import os
import re
from typing import Dict, Optional

try:
    from textblob import TextBlob

    try:
        from textblob.exceptions import MissingCorpusError
    except Exception:  # pragma: no cover
        MissingCorpusError = Exception
    TEXTBLOB_AVAILABLE = True
except Exception:  # pragma: no cover
    TextBlob = None
    MissingCorpusError = Exception
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
SELF_HARM_PATTERN = re.compile(
    r"\b(suicide|kill myself|end my life|self[- ]harm|cut myself|overdose|die)\b"
)
JAILBREAK_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "you are now",
    "do anything now",
    "dan",
    "developer mode",
    "jailbreak",
    "system prompt",
    "bypass safety",
)
BIAS_PROTECTED_GROUPS = (
    "women",
    "men",
    "black",
    "white",
    "asian",
    "hispanic",
    "latino",
    "muslim",
    "jewish",
    "christian",
    "gay",
    "lesbian",
    "trans",
    "immigrants",
    "disabled",
)
BIAS_NEGATIVE_DESCRIPTORS = (
    "inferior",
    "stupid",
    "lazy",
    "criminal",
    "dirty",
    "violent",
    "evil",
    "worthless",
    "should be banned",
    "should be deported",
)
REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i am unable to",
    "as an ai",
    "compliance violation",
    "against my programming",
    "i apologize",
    "i will not",
    "i won't",
)

DEFAULT_TOXICITY_MODEL = "unitary/unbiased-toxic-roberta"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"


class SignalDetector:
    def __init__(
        self, toxicity_model: Optional[str] = None, enable_toxicity: bool = True
    ):
        self.toxicity_model = toxicity_model or os.getenv(
            "SENTINEL_TOXICITY_MODEL", DEFAULT_TOXICITY_MODEL
        )
        self.enable_toxicity = (
            enable_toxicity and os.getenv("SENTINEL_DISABLE_TOXICITY", "0") != "1"
        )
        self._toxicity_pipeline = None
        self._toxicity_load_error = None
        self._sentiment_warned = False
        self._logger = logging.getLogger(__name__)
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def _load_toxicity_pipeline(self) -> None:
        if not self.enable_toxicity:
            return
        if self._toxicity_pipeline is not None or self._toxicity_load_error is not None:
            return
        if not TRANSFORMERS_AVAILABLE:
            self._toxicity_load_error = RuntimeError("transformers is not installed")
            self._logger.warning("transformers is not installed; toxicity disabled.")
            return
        try:
            self._logger.info("Loading toxicity model: %s", self.toxicity_model)
            self._toxicity_pipeline = pipeline(
                "text-classification", model=self.toxicity_model, top_k=None
            )
        except Exception as exc:
            self._toxicity_load_error = exc
            self._logger.warning("Toxicity model unavailable: %s", exc)

    def detect_pii(self, text: str) -> Dict[str, object]:
        """Scans for email addresses and phone numbers using regex."""
        emails = EMAIL_PATTERN.findall(text or "")
        phones = PHONE_PATTERN.findall(text or "")
        pii_data = emails + phones
        pii_types = []
        if emails:
            pii_types.append("email")
        if phones:
            pii_types.append("phone")
        return {
            "has_pii": len(pii_data) > 0,
            "pii_data": pii_data,
            "pii_types": pii_types,
        }

    def detect_toxicity(self, text: str) -> float:
        """Returns a float score 0.0 to 1.0. Defaults to 0.0 if disabled."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        if not self.enable_toxicity:
            return 0.0
        self._load_toxicity_pipeline()
        if self._toxicity_pipeline is None:
            return 0.0
        try:
            results = self._toxicity_pipeline(text)
            for result in results[0]:
                if result.get("label") == "toxicity":
                    return float(result.get("score", 0.0))
        except Exception as exc:
            self._logger.warning("Toxicity scoring failed: %s", exc)
        return 0.0

    def detect_refusal(self, text: str) -> bool:
        """Checks if the model refused to answer (compliance signal)."""
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        return any(phrase in lower_text for phrase in REFUSAL_PHRASES)

    def find_refusal_phrase(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        lower_text = text.lower()
        for phrase in REFUSAL_PHRASES:
            if phrase in lower_text:
                return phrase
        return None

    def detect_self_harm(self, text: str) -> bool:
        """Heuristic detection of self-harm related content."""
        if not isinstance(text, str) or not text.strip():
            return False
        return bool(SELF_HARM_PATTERN.search(text.lower()))

    def find_self_harm_match(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        match = SELF_HARM_PATTERN.search(text.lower())
        return match.group(0) if match else None

    def detect_jailbreak(self, text: str) -> bool:
        """Heuristic detection of prompt injection / jailbreak attempts."""
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        return any(phrase in lower_text for phrase in JAILBREAK_PHRASES)

    def find_jailbreak_phrase(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        lower_text = text.lower()
        for phrase in JAILBREAK_PHRASES:
            if phrase in lower_text:
                return phrase
        return None

    def detect_bias(self, text: str) -> bool:
        """Heuristic detection of biased / hateful language."""
        if not isinstance(text, str) or not text.strip():
            return False
        lower_text = text.lower()
        group_hit = any(group in lower_text for group in BIAS_PROTECTED_GROUPS)
        negative_hit = any(term in lower_text for term in BIAS_NEGATIVE_DESCRIPTORS)
        return group_hit and negative_hit

    def find_bias_match(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None
        lower_text = text.lower()
        group = next((g for g in BIAS_PROTECTED_GROUPS if g in lower_text), None)
        negative = next((t for t in BIAS_NEGATIVE_DESCRIPTORS if t in lower_text), None)
        if group and negative:
            return f"{group} + {negative}"
        return None

    def redact_pii(self, text: str) -> Dict[str, object]:
        """Redact PII from text before persistence."""
        if not isinstance(text, str) or not text:
            return {"redacted_text": text, "redaction_count": 0}
        redacted, email_count = EMAIL_PATTERN.subn(REDACTED_EMAIL, text)
        redacted, phone_count = PHONE_PATTERN.subn(REDACTED_PHONE, redacted)
        return {
            "redacted_text": redacted,
            "redaction_count": email_count + phone_count,
        }

    def detect_sentiment(self, text: str) -> float:
        """Uses TextBlob for simple polarity check (-1 to 1)."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        if not TEXTBLOB_AVAILABLE:
            if not self._sentiment_warned:
                self._logger.warning("textblob is not installed; sentiment disabled.")
                self._sentiment_warned = True
            return 0.0
        try:
            return float(TextBlob(text).sentiment.polarity)
        except MissingCorpusError:
            self._logger.warning("TextBlob corpora missing; sentiment disabled.")
        except Exception as exc:
            self._logger.warning("Sentiment scoring failed: %s", exc)
        return 0.0

    def analyze_output(self, output_text: str) -> Dict[str, object]:
        """Run all signal detectors on the output text."""
        pii_result = self.detect_pii(output_text)
        return {
            "toxicity_score": self.detect_toxicity(output_text),
            "pii": pii_result,
            "is_refusal": self.detect_refusal(output_text),
            "self_harm": self.detect_self_harm(output_text),
            "jailbreak": self.detect_jailbreak(output_text),
            "bias": self.detect_bias(output_text),
            "sentiment_score": self.detect_sentiment(output_text),
        }
