from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field

Metric = Optional[Literal["P", "Q", "S"]]

class CIMRequestModeDecision(BaseModel):
    intent: str = Field(description="Allowed values are provided in the prompt.")
    confidence: str = Field(description="Allowed values are provided in the prompt.")
    target_kind: str = Field(description="Allowed values are provided in the prompt.")
    request_mode: str = Field(description="Allowed values are provided in the prompt.")
    safe_to_execute: bool = Field(
        description="True if the workflow can be executed with the currently supported capabilities"
    )
    missing_context: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the decision")


class CIMRequestModeOnlyDecision(BaseModel):
    request_mode: str = Field(description="Allowed values are provided in the prompt.")
    reasoning: str = Field(description="Short explanation of the decision")


class CIMCustomPlanDecision(BaseModel):
    required_steps: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the planning decision")

class EquipmentTypeDecision(BaseModel):
    selected_type: Optional[str] = Field(
        default=None,
        description="Exact class name from the provided equipment type list, or null if no safe match exists.",
    )
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the selection")
    should_execute: bool = Field(
        description="True only if the selected type is a safe unambiguous choice."
    )
    alternatives: List[str] = Field(default_factory=list)


class EquipmentInstanceDecision(BaseModel):
    selected_equipment_id: Optional[str] = Field(
        default=None,
        description="Exact canonical_id from the provided candidate list, or null if no safe match exists.",
    )
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the selection")
    should_execute: bool = Field(
        description="True only if the selected equipment candidate is a safe unambiguous choice."
    )
    alternatives: List[str] = Field(default_factory=list)


class ParsedQueryNormalizationDecision(BaseModel):
    equipment_type_hint: Optional[str] = Field(default=None)
    equipment_name_hint: Optional[str] = Field(default=None)


class ComparisonResolutionDecision(BaseModel):
    comparison_type: Optional[str] = Field(default=None)
    should_execute: bool = Field(default=False)
    rationale: str = Field(default="")


class FinalAnswerDecision(BaseModel):
    answer: str = Field(default="")

class BaseAttributeIntentDecision(BaseModel):
    requested_attributes: List[str] = Field(default_factory=list)
    should_use_preselected_attributes: bool = Field(default=False)
    rationale: str = Field(default="")

class VoltageLimitSelectionDecision(BaseModel):
    low_candidate_id: Optional[str] = Field(default=None)
    high_candidate_id: Optional[str] = Field(default=None)
    should_execute: bool = Field(default=False)
    rationale: str = Field(default="")

class EquipmentSelection(BaseModel):
    equipment_type: str
    equipment_key: str
    equipment_name: Optional[str] = None
    equipment_id: Optional[str] = None


class QueryParse(BaseModel):
    equipment_detected: List[str] = Field(default_factory=list)
    state_detected: List[str] = Field(default_factory=list)
    metric: Metric = None
    equipment_selection: List[EquipmentSelection] = Field(default_factory=list)
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    time_label: Optional[str] = None


class CandidateChoice(BaseModel):
    equipment_key: Optional[str] = None
    need_clarification: bool = False
    clarification_question: Optional[str] = None

class BaseAttributeSelectionDecision(BaseModel):
    selected_attributes: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(default="")

class AttributeRetryDecision(BaseModel):
    should_retry_with_fallback: bool = Field(default=False)
    rationale: str = Field(default="")

class CandidateShortlistDecision(BaseModel):
    selected_candidate_keys: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(default="")

class BaseAttributeCandidateDecision(BaseModel):
    selected_candidates: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(default="")