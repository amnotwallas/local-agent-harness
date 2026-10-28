from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    statement: str
    evidence_sources: list[str] = Field(
        description="Tool names whose successful results support this observation or hypothesis."
    )


class ToolFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_name: str
    summary: str


class IncidentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confirmed_observations: list[EvidenceStatement] = Field(
        description="Facts directly supported by successful tool results."
    )
    hypotheses: list[EvidenceStatement] = Field(
        description="Possible explanations, not confirmed findings."
    )
    missing_evidence: list[str] = Field(
        description="Evidence needed to answer the task but not available in the results."
    )
    tool_failures: list[ToolFailure] = Field(
        description="Tools that failed and what could not be retrieved."
    )
