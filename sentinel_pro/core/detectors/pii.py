from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sentinel_pro.core.detectors.base import DetectionResult, Detector

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"

try:
    from presidio_analyzer import (
        AnalyzerEngine,
        EntityRecognizer,
        RecognizerRegistry,
        RecognizerResult,
    )
    from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine, NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    PRESIDIO_AVAILABLE = True
except Exception:  # pragma: no cover
    AnalyzerEngine = None
    AnonymizerEngine = None
    EntityRecognizer = None
    NlpArtifacts = None
    NlpEngine = object
    NlpEngineProvider = None
    OperatorConfig = None
    RecognizerRegistry = None
    RecognizerResult = None
    PRESIDIO_AVAILABLE = False


DIRECT_PII_ENTITIES = {
    "ABA_ROUTING_NUMBER",
    "AU_ABN",
    "AU_ACN",
    "AU_MEDICARE",
    "AU_TFN",
    "CREDIT_CARD",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "FI_PERSONAL_IDENTITY_CODE",
    "IBAN_CODE",
    "IN_AADHAAR",
    "IN_GSTIN",
    "IN_PAN",
    "IN_PASSPORT",
    "IN_VEHICLE_REGISTRATION",
    "IN_VOTER",
    "IP_ADDRESS",
    "IT_DRIVER_LICENSE",
    "IT_FISCAL_CODE",
    "IT_IDENTITY_CARD",
    "IT_PASSPORT",
    "IT_VAT_CODE",
    "KR_BRN",
    "KR_DRIVER_LICENSE",
    "KR_FRN",
    "KR_PASSPORT",
    "KR_RRN",
    "MAC_ADDRESS",
    "MEDICAL_LICENSE",
    "NG_NIN",
    "NG_VEHICLE_REGISTRATION",
    "PHONE_NUMBER",
    "PL_PESEL",
    "SG_NRIC_FIN",
    "SG_UEN",
    "TH_TNIN",
    "UK_NHS",
    "UK_NINO",
    "UK_PASSPORT",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_MBI",
    "US_NPI",
    "US_PASSPORT",
    "US_SSN",
}

CONTEXTUAL_PII_ENTITIES = {
    "DATE_TIME",
    "LOCATION",
    "NRP",
    "ORGANIZATION",
    "PERSON",
    "UK_POSTCODE",
    "URL",
}

PII_TYPE_LABELS = {
    "ABA_ROUTING_NUMBER": "aba_routing_number",
    "AU_ABN": "au_abn",
    "AU_ACN": "au_acn",
    "AU_MEDICARE": "au_medicare",
    "AU_TFN": "au_tfn",
    "CREDIT_CARD": "credit_card",
    "CRYPTO": "crypto_wallet",
    "DATE_TIME": "date_time",
    "EMAIL_ADDRESS": "email",
    "FI_PERSONAL_IDENTITY_CODE": "fi_personal_identity_code",
    "IBAN_CODE": "iban",
    "IN_AADHAAR": "in_aadhaar",
    "IN_GSTIN": "in_gstin",
    "IN_PAN": "in_pan",
    "IN_PASSPORT": "in_passport",
    "IN_VEHICLE_REGISTRATION": "in_vehicle_registration",
    "IN_VOTER": "in_voter",
    "IP_ADDRESS": "ip_address",
    "IT_DRIVER_LICENSE": "it_driver_license",
    "IT_FISCAL_CODE": "it_fiscal_code",
    "IT_IDENTITY_CARD": "it_identity_card",
    "IT_PASSPORT": "it_passport",
    "IT_VAT_CODE": "it_vat_code",
    "KR_BRN": "kr_brn",
    "KR_DRIVER_LICENSE": "kr_driver_license",
    "KR_FRN": "kr_frn",
    "KR_PASSPORT": "kr_passport",
    "KR_RRN": "kr_rrn",
    "LOCATION": "location",
    "MAC_ADDRESS": "mac_address",
    "MEDICAL_LICENSE": "medical_license",
    "NG_NIN": "ng_nin",
    "NG_VEHICLE_REGISTRATION": "ng_vehicle_registration",
    "NRP": "nationality_religion_political",
    "ORGANIZATION": "organization",
    "PERSON": "person",
    "PHONE_NUMBER": "phone",
    "PL_PESEL": "pl_pesel",
    "SG_NRIC_FIN": "sg_nric_fin",
    "SG_UEN": "sg_uen",
    "TH_TNIN": "th_tnin",
    "UK_NHS": "uk_nhs",
    "UK_NINO": "uk_nino",
    "UK_PASSPORT": "uk_passport",
    "UK_POSTCODE": "uk_postcode",
    "URL": "url",
    "US_BANK_NUMBER": "us_bank_number",
    "US_DRIVER_LICENSE": "us_driver_license",
    "US_ITIN": "us_itin",
    "US_MBI": "us_mbi",
    "US_NPI": "us_npi",
    "US_PASSPORT": "us_passport",
    "US_SSN": "us_ssn",
}

REDACTION_TOKENS = {
    "EMAIL_ADDRESS": REDACTED_EMAIL,
    "PHONE_NUMBER": REDACTED_PHONE,
}


@dataclass(frozen=True)
class PiiEntity:
    entity_type: str
    pii_type: str
    text: str
    start: int
    end: int
    score: float
    source: str = "presidio"

    def to_dict(self) -> Dict[str, object]:
        return {
            "entity_type": self.entity_type,
            "pii_type": self.pii_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
        }


if PRESIDIO_AVAILABLE:

    class _NoOpNlpEngine(NlpEngine):  # type: ignore[misc, valid-type]
        def load(self) -> None:
            return None

        def is_loaded(self) -> bool:
            return True

        def process_text(self, text: str, language: str) -> Any:
            return NlpArtifacts([], [], [], [], self, language)

        def process_batch(
            self,
            texts: Iterable[str],
            language: str,
            batch_size: int = 1,
            n_process: int = 1,
            **kwargs: Any,
        ) -> Iterable[tuple[str, Any]]:
            for text in texts:
                yield text, self.process_text(text, language)

        def is_stopword(self, word: str, language: str) -> bool:
            return False

        def is_punct(self, word: str, language: str) -> bool:
            return False

        def get_supported_entities(self) -> List[str]:
            return []

        def get_supported_languages(self) -> List[str]:
            return ["en"]

else:
    _NoOpNlpEngine = None  # type: ignore[assignment]


def _entity_type_to_label(entity_type: str) -> str:
    return PII_TYPE_LABELS.get(entity_type, entity_type.lower())


def _redaction_token(entity_type: str) -> str:
    return REDACTION_TOKENS.get(entity_type, f"[REDACTED_{entity_type}]")


def _parse_entity_list(value: str) -> Optional[set[str]]:
    if not value.strip():
        return None
    return {item.strip().upper() for item in value.split(",") if item.strip()}


class PiiDetector(Detector):
    label = "pii"

    def __init__(
        self,
        *,
        language: str = "en",
        score_threshold: Optional[float] = None,
        entities: Optional[Sequence[str]] = None,
    ) -> None:
        self.language = language
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else float(os.getenv("SENTINEL_PII_SCORE_THRESHOLD", "0.35"))
        )
        include_contextual = os.getenv("SENTINEL_PII_INCLUDE_CONTEXTUAL", "0") == "1"
        configured_entities = _parse_entity_list(os.getenv("SENTINEL_PII_ENTITIES", ""))
        if entities is not None:
            self.entities = {entity.upper() for entity in entities}
        elif configured_entities is not None:
            self.entities = configured_entities
        else:
            self.entities = set(DIRECT_PII_ENTITIES)
            if include_contextual:
                self.entities.update(CONTEXTUAL_PII_ENTITIES)

        self._analyzer = None
        self._anonymizer = None
        self._presidio_load_error: Optional[Exception] = None
        self._logger = logging.getLogger(__name__)

    def _build_nlp_engine(self) -> Any:
        nlp_model = os.getenv("SENTINEL_PII_NLP_MODEL", "").strip()
        if not nlp_model:
            return _NoOpNlpEngine()
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": os.getenv("SENTINEL_PII_NLP_ENGINE", "spacy"),
                "models": [{"lang_code": self.language, "model_name": nlp_model}],
            }
        )
        return provider.create_engine()

    def _add_global_recognizers(self, registry: Any) -> None:
        try:
            import presidio_analyzer.predefined_recognizers.country_specific as countries
        except Exception as exc:  # pragma: no cover
            self._logger.info("Presidio country recognizers unavailable: %s", exc)
            return

        seen = {type(recognizer).__name__ for recognizer in registry.recognizers}
        for package in pkgutil.iter_modules(
            countries.__path__, countries.__name__ + "."
        ):
            module = importlib.import_module(package.name)
            if not hasattr(module, "__path__"):
                continue
            for submodule_info in pkgutil.iter_modules(
                module.__path__, module.__name__ + "."
            ):
                submodule = importlib.import_module(submodule_info.name)
                for class_name, recognizer_cls in inspect.getmembers(
                    submodule, inspect.isclass
                ):
                    if recognizer_cls.__module__ != submodule.__name__:
                        continue
                    if not issubclass(recognizer_cls, EntityRecognizer):
                        continue
                    if class_name in seen:
                        continue
                    try:
                        recognizer = recognizer_cls(supported_language=self.language)
                    except TypeError:
                        continue
                    registry.add_recognizer(recognizer)
                    seen.add(class_name)

    def _load_presidio(self) -> None:
        if self._analyzer is not None or self._presidio_load_error is not None:
            return
        if not PRESIDIO_AVAILABLE:
            self._presidio_load_error = RuntimeError("presidio is not installed")
            self._logger.warning(
                "Presidio is not installed; regex PII fallback enabled."
            )
            return

        try:
            registry = RecognizerRegistry(supported_languages=[self.language])
            registry.load_predefined_recognizers(languages=[self.language])
            if not os.getenv("SENTINEL_PII_NLP_MODEL", "").strip():
                registry.recognizers = [
                    recognizer
                    for recognizer in registry.recognizers
                    if type(recognizer).__name__ != "SpacyRecognizer"
                ]
            self._add_global_recognizers(registry)
            nlp_engine = self._build_nlp_engine()
            self._analyzer = AnalyzerEngine(
                registry=registry,
                nlp_engine=nlp_engine,
                supported_languages=[self.language],
                default_score_threshold=self.score_threshold,
            )
            self._anonymizer = AnonymizerEngine()
        except Exception as exc:
            self._presidio_load_error = exc
            self._logger.warning("Presidio PII detector unavailable: %s", exc)

    def _fallback_entities(self, text: str) -> List[PiiEntity]:
        entities: List[PiiEntity] = []
        for match in EMAIL_PATTERN.finditer(text):
            entities.append(
                PiiEntity(
                    entity_type="EMAIL_ADDRESS",
                    pii_type="email",
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                    source="regex_fallback",
                )
            )
        for match in PHONE_PATTERN.finditer(text):
            entities.append(
                PiiEntity(
                    entity_type="PHONE_NUMBER",
                    pii_type="phone",
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=0.75,
                    source="regex_fallback",
                )
            )
        return self._dedupe_overlaps(entities)

    def _dedupe_overlaps(self, entities: List[PiiEntity]) -> List[PiiEntity]:
        ordered = sorted(
            entities,
            key=lambda entity: (
                entity.start,
                -(entity.end - entity.start),
                -entity.score,
            ),
        )
        selected: List[PiiEntity] = []
        for entity in ordered:
            overlaps = [
                existing
                for existing in selected
                if entity.start < existing.end and entity.end > existing.start
            ]
            if not overlaps:
                selected.append(entity)
                continue
            strongest = max(
                overlaps,
                key=lambda existing: (existing.score, existing.end - existing.start),
            )
            current_key = (entity.score, entity.end - entity.start)
            strongest_key = (strongest.score, strongest.end - strongest.start)
            if current_key > strongest_key:
                selected = [item for item in selected if item not in overlaps]
                selected.append(entity)
        return sorted(selected, key=lambda entity: entity.start)

    def _presidio_entities(self, text: str) -> List[PiiEntity]:
        self._load_presidio()
        if self._analyzer is None:
            return self._fallback_entities(text)

        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=sorted(self.entities),
            score_threshold=self.score_threshold,
        )
        entities = [
            PiiEntity(
                entity_type=result.entity_type,
                pii_type=_entity_type_to_label(result.entity_type),
                text=text[result.start : result.end],
                start=result.start,
                end=result.end,
                score=float(result.score),
            )
            for result in results
            if result.entity_type in self.entities
        ]
        return self._dedupe_overlaps(entities)

    def detect_pii(self, text: str) -> Dict[str, object]:
        if not isinstance(text, str) or not text:
            return {"has_pii": False, "pii_data": [], "pii_types": [], "entities": []}

        entities = self._presidio_entities(text)
        pii_types = sorted({entity.pii_type for entity in entities})
        pii_data = [entity.text for entity in entities]
        return {
            "has_pii": bool(entities),
            "pii_data": pii_data,
            "pii_types": pii_types,
            "entities": [entity.to_dict() for entity in entities],
            "engine": "presidio" if self._analyzer is not None else "regex_fallback",
        }

    def redact_pii(self, text: str) -> Dict[str, object]:
        if not isinstance(text, str) or not text:
            return {"redacted_text": text, "redaction_count": 0}

        entities = self._presidio_entities(text)
        if not entities:
            return {"redacted_text": text, "redaction_count": 0}

        self._load_presidio()
        if self._anonymizer is not None and PRESIDIO_AVAILABLE:
            analyzer_results = [
                RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start,
                    end=entity.end,
                    score=entity.score,
                )
                for entity in entities
            ]
            operators = {
                entity.entity_type: OperatorConfig(
                    "replace",
                    {"new_value": _redaction_token(entity.entity_type)},
                )
                for entity in entities
            }
            anonymized = self._anonymizer.anonymize(
                text=text, analyzer_results=analyzer_results, operators=operators
            )
            return {
                "redacted_text": anonymized.text,
                "redaction_count": len(anonymized.items),
            }

        redacted = text
        redaction_count = 0
        for entity in reversed(entities):
            redacted = (
                redacted[: entity.start]
                + _redaction_token(entity.entity_type)
                + redacted[entity.end :]
            )
            redaction_count += 1
        return {"redacted_text": redacted, "redaction_count": redaction_count}

    def detect(self, text: str) -> DetectionResult:
        result = self.detect_pii(text)
        entities = result.get("entities", [])
        pii_types = result.get("pii_types", [])
        match_count = len(entities) if isinstance(entities, list) else 0
        detected = bool(result.get("has_pii"))
        if not detected:
            return DetectionResult(label=self.label, detected=False)

        high_sensitivity = {
            "credit_card",
            "crypto_wallet",
            "iban",
            "us_ssn",
            "uk_nhs",
            "uk_nino",
            "in_aadhaar",
            "pl_pesel",
            "kr_rrn",
            "au_tfn",
        }
        pii_type_set = {str(item) for item in pii_types}
        risk_score = 0.9 if pii_type_set & high_sensitivity else 0.75
        if match_count > 1 or len(pii_type_set) > 1:
            risk_score = max(risk_score, 0.85)
        pii_label = ", ".join(str(item) for item in pii_types) or "unknown"
        return DetectionResult(
            label=self.label,
            detected=True,
            risk_score=risk_score,
            severity="high",
            explanation=f"PII detected ({pii_label})",
            metadata={
                "pii_types": pii_types,
                "match_count": match_count,
                "engine": result.get("engine"),
                "entities": entities,
            },
        )
