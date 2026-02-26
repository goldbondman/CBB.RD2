from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha1


@dataclass(frozen=True)
class RunContext:
    run_id: str
    as_of: str
    model_version: str
    feature_version: str
    prediction_timestamp: str

    @classmethod
    def build(
        cls,
        *,
        as_of: datetime,
        model_version: str,
        feature_version: str,
        prediction_timestamp: datetime | None = None,
    ) -> "RunContext":
        ts = prediction_timestamp or datetime.now(timezone.utc)
        payload = f"{as_of.isoformat()}|{model_version}|{feature_version}|{ts.isoformat()}"
        run_id = sha1(payload.encode("utf-8")).hexdigest()[:16]
        return cls(
            run_id=run_id,
            as_of=as_of.isoformat(),
            model_version=model_version,
            feature_version=feature_version,
            prediction_timestamp=ts.isoformat(),
        )

    def to_dict(self) -> dict:
        return asdict(self)
