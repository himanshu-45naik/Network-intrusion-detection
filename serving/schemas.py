"""Request and response models for the inference API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# `model_version` and `model_loaded` are domain terms here, not pydantic
# internals, so the protected `model_` namespace is switched off.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class FlowRequest(BaseModel):
    """A single network flow to score.

    Features are passed as a name/value map rather than a fixed field list:
    CICFlowMeter emits ~78 columns, the trained model keeps 70, and a map lets a
    client send its full row without tracking which columns were dropped.
    """

    features: Dict[str, Optional[float]] = Field(
        ..., description="Flow feature values keyed by CICIDS2017 column name."
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Decision threshold on P(attack). Defaults to the service setting.",
    )


class BatchRequest(BaseModel):
    """A batch of flows, scored in one vectorised call."""

    flows: List[Dict[str, Optional[float]]] = Field(..., min_length=1)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class Prediction(BaseModel):
    verdict: str = Field(..., description="`attack` or `benign`.")
    attack_type: str = Field(..., description="Predicted attack family, or BENIGN.")
    attack_probability: float = Field(..., description="1 - P(BENIGN).")
    class_probabilities: Dict[str, float]


class PredictResponse(Prediction):
    model_config = _ALLOW_MODEL_PREFIX

    model_version: str
    threshold: float
    latency_ms: float
    unknown_features: List[str] = Field(
        default_factory=list, description="Payload keys the model does not use."
    )
    missing_features: List[str] = Field(
        default_factory=list, description="Expected features absent from the payload."
    )


class BatchResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    predictions: List[Prediction]
    count: int
    attacks_detected: int
    model_version: str
    threshold: float
    latency_ms: float


class HealthResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float


class DriftResponse(BaseModel):
    status: str
    rows_observed: int
    rows_required: Optional[int] = None
    max_psi: Optional[float] = None
    alert: bool = False
    alert_threshold: float
    features: List[Dict[str, Any]]
