from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Optional

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
    channel: str = "document"


class MCPMessageBus:
    """Lightweight client for the Multi-Agent Communication Protocol (MCP).

    Agents publish/subscribe over channels (topics) to exchange tasks, payloads,
    and streaming updates. This wrapper provides a minimal request/response
    helper for demo purposes; production systems should add auth, retries, and
    tracing.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    async def publish(self, channel: str, message: Dict[str, Any]) -> None:
        async with websockets.connect(self.endpoint) as ws:  # type: ignore[arg-type]
            await ws.send(json.dumps({"channel": channel, "message": message}))

    async def subscribe(self, channel: str) -> AsyncIterable[Dict[str, Any]]:
        async with websockets.connect(self.endpoint) as ws:  # type: ignore[arg-type]
            await ws.send(json.dumps({"channel": channel, "subscribe": True}))
            async for raw in ws:  # pragma: no cover - network I/O
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    yield {"channel": channel, "message": raw}

    async def request(self, channel: str, message: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        async with websockets.connect(self.endpoint) as ws:  # type: ignore[arg-type]
            await ws.send(json.dumps({"channel": channel, "message": message, "reply": True}))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)  # pragma: no cover - network I/O
            except asyncio.TimeoutError:
                raise TimeoutError(f"MCP bus request to {channel} timed out")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"payload": raw}


class MCPMarketClient:
    """Market data helper built on top of the MCP message bus."""

    def __init__(self, bus: MCPMessageBus, channel: str = "market") -> None:
        self.bus = bus
        self.channel = channel

    async def stream_prices(self, request: MarketRequest) -> AsyncIterable[Dict[str, Any]]:
        # Rely on MCP streaming channel; market-agent publishes incremental ticks.
        async for event in self.bus.subscribe(self.channel):
            payload = event.get("message", {})
            if payload.get("symbol") == request.symbol:
                yield payload

    async def fetch_history(self, request: MarketRequest) -> Dict[str, Any]:
        return await self.bus.request(
            self.channel,
            {"action": "history", "params": request.__dict__},
        )


class MCPDocumentClient:
    """Document/media helper built on MCP message bus.

    The expectation is another agent (e.g., `document-service-agent`) listens on
    the same channel and returns base64-encoded content or OCR/ASR text.
    """

    def __init__(self, bus: MCPMessageBus, channel: str = "document") -> None:
        self.bus = bus
        self.channel = channel

    async def fetch(self, request: DocumentRequest) -> bytes:
        resp = await self.bus.request(
            request.channel or self.channel,
            {"action": "fetch", "uri": request.uri, "media_type": request.media_type},
        )
        content_b64: Optional[str] = resp.get("content_base64")
        if content_b64:
            return base64.b64decode(content_b64)
        if "content" in resp and isinstance(resp["content"], (bytes, bytearray)):
            return bytes(resp["content"])
        # Fallback: pull via HTTP if MCP agent returned signed URL
        if "signed_url" in resp:
            async with httpx.AsyncClient() as client:
                http_resp = await client.get(resp["signed_url"], timeout=30.0)
                http_resp.raise_for_status()
                return http_resp.content
        raise ValueError("MCP fetch response missing content")

    async def ocr_image(self, request: DocumentRequest) -> str:
        resp = await self.bus.request(
            request.channel or self.channel,
            {"action": "ocr", "uri": request.uri, "media_type": request.media_type},
        )
        if "text" in resp:
            return str(resp["text"])
        raise ValueError("MCP OCR response missing text")

    async def speech_to_text(self, request: DocumentRequest) -> str:
        resp = await self.bus.request(
            request.channel or self.channel,
            {"action": "asr", "uri": request.uri, "media_type": request.media_type},
        )
        if "text" in resp:
            return str(resp["text"])
        raise ValueError("MCP ASR response missing text")


def normalize_documents(responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for resp in responses:
        metadata = resp.get("metadata", {})
        content = resp.get("content", "")
        normalized.append({"page_content": content, "metadata": metadata})
    return normalized
