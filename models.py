from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Speech:
    content: str
    politician: str
    term: int
    position: str
    date: str
    faction: str
    embedding: Optional[List[float]] = field(default=None)
    cluster: Optional[int] = field(default=None)
    topic: Optional[int] = field(default=None)
    topic_desc: Optional[str] = field(default=None)
    cluster_desc: Optional[str] = field(default=None)
