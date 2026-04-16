from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LoadChangeInstruction(BaseModel):
    action: Literal["change_load"] = "change_load"
    load_name: str
    delta_p_mw: float
    result_requests: List[Literal["bus_voltage", "bus_p", "bus_q", "line_loading"]] = Field(
        default_factory=lambda: ["bus_voltage"]
    )


class DataQueryInstruction(BaseModel):
    query_type: Literal["element_data"] = "element_data"
    entity_type: str
    entity_name_raw: str
    attribute_request_text: str = ""
    requested_attribute_names: List[str] = Field(default_factory=list)
    data_source_preference: Literal["base", "result"] = "base"
    data_source_note: str = ""
    selected_attribute_handles: List[str] = Field(default_factory=list)


class RequestedAttributeNameDecision(BaseModel):
    requested_attribute_names: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    rationale: str
    should_execute: bool


class AttributeDescriptionShortlistDecision(BaseModel):
    shortlisted_attribute_names: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    rationale: str
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool


class AttributeDescriptionMatchDecision(BaseModel):
    selected_attribute_names: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    rationale: str
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool


# ------------------------------------------------------------------
# LLM OUTPUT MODELS FOR DATA QUERY
# ------------------------------------------------------------------
class DataQueryTypeDecision(BaseModel):
    selected_entity_type: Optional[str] = Field(default=None)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True only if the selected entity type is sufficiently safe.")


class InventoryObjectMatchDecision(BaseModel):
    selected_object_name: Optional[str] = Field(default=None)
    selected_object_names: List[str] = Field(default_factory=list)
    selection_mode: Optional[str] = Field(
        default=None,
        description="Use 'one' for a single exact object or 'all' if the request clearly targets all provided candidates."
    )
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    alternatives: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True only if the selected object is a safe unambiguous choice.")


class AttributeSelectionDecision(BaseModel):
    selected_attribute_handles: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True only if the selected attributes are a safe grounded match.")



class DataSourceDecision(BaseModel):
    selected_data_source: str = Field(description="One of: base, result, ambiguous")
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the decision")
    should_execute: bool = Field(description="True if the decision is grounded enough to use directly without fallback.")


class ResultPredefinedFieldDecision(BaseModel):
    selected_field_names: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the selection decision")
    should_execute: bool = Field(description="True only if the selected predefined result fields are a safe grounded match.")

class TopologyEntityNameCandidatesDecision(BaseModel):
    candidate_names: List[str] = Field(
        default_factory=list,
        description="Ordered list of plausible PowerFactory asset name candidates derived from the user request."
    )
    rationale: str = Field(
        default="",
        description="Short explanation of how the candidate names were derived."
    )

class TopologyEntityTypeDecision(BaseModel):
    entity_type: Optional[str] = Field(
        default=None,
        description="One supported topology entity type from the available types, or null if no safe choice is possible."
    )
    confidence: str = Field(
        default="low",
        description="One of: high, medium, low"
    )
    should_execute: bool = Field(
        default=False,
        description="True if the entity type classification is grounded enough."
    )
    rationale: str = Field(
        default="",
        description="Short explanation of the classification decision."
    )

class SwitchInstructionDecision(BaseModel):
    operation: Optional[str] = Field(
        default=None,
        description="One of: open, close, toggle, or null if no safe switch operation can be determined."
    )
    should_execute: bool = Field(
        default=False,
        description="True if the requested switch operation is grounded enough to execute."
    )
    confidence: str = Field(
        default="low",
        description="One of: high, medium, low."
    )
    rationale: str = Field(
        default="",
        description="Short explanation of the decision."
    )