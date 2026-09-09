"""Application defaults and a few environment settings; never persist the API key."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    extraction_model: str = "gpt-5.6-terra"
    verification_model: str = "gpt-5.6-terra"
    embedding_model: str = "text-embedding-3-small"
    verification_claim_limit: int = 20
    max_tool_calls: int = 20
    cohere_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        integer_fields = {
            "verification_claim_limit": "VERIFICATION_CLAIM_LIMIT",
            "max_tool_calls": "MAX_TOOL_CALLS",
        }
        defaults = cls()
        values = {}
        for name, env_name in integer_fields.items():
            try:
                value = int(os.getenv(env_name, str(getattr(defaults, name))))
            except ValueError:
                raise ValueError(f"{env_name} must be a positive integer.") from None
            if value <= 0:
                raise ValueError(f"{env_name} must be a positive integer.")
            values[name] = value
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            cohere_key=os.getenv("COHERE_KEY", "").strip(),
            extraction_model=os.getenv("EXTRACTION_MODEL", defaults.extraction_model).strip(),
            verification_model=os.getenv("VERIFICATION_MODEL", defaults.verification_model).strip(),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or defaults.embedding_model,
            **values,
        )

    def require_openai(self, stage="extraction"):
        if stage not in {"extraction", "verification", "embedding"}:
            raise ValueError("Stage must be extraction, verification, or embedding.")
        missing = []
        if not self.api_key:
            missing.append("OPENAI_API_KEY")
        if not getattr(self, f"{stage}_model"):
            missing.append("OPENAI_EMBEDDING_MODEL" if stage == "embedding" else f"{stage.upper()}_MODEL")
        if missing:
            raise ValueError(f"Set {', '.join(missing)} in .env before live {stage}.")

    def public_dict(self):
        return {key: value for key, value in asdict(self).items() if key not in {"api_key", "cohere_key"}}
