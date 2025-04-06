from dataclasses import dataclass, field


@dataclass
class ResponseObject:
    """Response object to be returned by the models."""

    sentence_id: int
    output: str
    duration: float
    errors: list = field(default_factory=list)
