from dataclasses import dataclass
import os
from typing import List


@dataclass
class Settings:
    """Centralized configuration for the financial analysis system."""

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    mcp_market_url: str = os.getenv("MCP_MARKET_URL", "wss://mcp.example.com/market")
    mcp_doc_url: str = os.getenv("MCP_DOC_URL", "https://mcp.example.com/doc")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./artifacts/vectorstore")
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    allowed_modalities: List[str] = None

    def __post_init__(self) -> None:
        if self.allowed_modalities is None:
            self.allowed_modalities = ["text", "pdf", "image", "audio"]


def load_settings() -> Settings:
    return Settings()
