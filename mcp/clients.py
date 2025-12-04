from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import httpx
import websockets


@dataclass
class MarketRequest:
    symbol: str
    range: str = "1mo"
    interval: str = "1h"


@dataclass
class DocumentRequest:
    uri: str
    media_type: str = "application/pdf"


class MCPMarketClient:
    """Minimal MCP market client using WebSocket streaming."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    async def stream_prices(self, request: MarketRequest) -> Iterable[Dict[str, Any]]:
        async with websockets.connect(self.endpoint) as ws:  # type: ignore[arg-type]
            await ws.send(request.__dict__)
            async for raw in ws:  # pragma: no cover - network I/O
                yield raw

    async def fetch_history(self, request: MarketRequest) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.endpoint.replace('wss://', 'https://')}/history",
                params=request.__dict__,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()


class MCPDocumentClient:
    """MCP document/media client that normalizes PDF/image/audio access."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    async def fetch(self, request: DocumentRequest) -> bytes:
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.endpoint, params=request.__dict__, timeout=30.0)
            resp.raise_for_status()
            return resp.content

    async def ocr_image(self, request: DocumentRequest) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.endpoint}/ocr", params=request.__dict__, timeout=30.0)
            resp.raise_for_status()
            return resp.text

    async def speech_to_text(self, request: DocumentRequest) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.endpoint}/asr", params=request.__dict__, timeout=30.0)
            resp.raise_for_status()
            return resp.text


def normalize_documents(responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for resp in responses:
        metadata = resp.get("metadata", {})
        content = resp.get("content", "")
        normalized.append({"page_content": content, "metadata": metadata})
    return normalized
