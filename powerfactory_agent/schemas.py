from __future__ import annotations

from typing import List, Literal

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
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the extraction decision")
    should_execute: bool = Field(description="True if the extracted attribute names are grounded in the user request.")


class AttributeDescriptionShortlistDecision(BaseModel):
    shortlisted_attribute_names: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the shortlist decision")
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True if the shortlist is grounded enough to continue with final disambiguation.")


class AttributeDescriptionMatchDecision(BaseModel):
    selected_attribute_names: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the description-based match decision")
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool = Field(
        description="True only if the selected attributes are a safe grounded match against PowerFactory attribute descriptions."
    )