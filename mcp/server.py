"""Minimal MCP (Multi-Agent Communication Protocol) server for demo/testing.

This server exposes two channels over WebSocket:
- ``market``: fetch historical quotes via Yahoo Finance (yfinance)
- ``document``: fetch remote files (HTTP/S) and return as base64 for OCR/LLM ingestion

The server is intentionally simple for local development and can be replaced by
production-grade message buses (NATS/Kafka) with the same JSON envelope shape.
"""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import websockets
import yfinance as yf


@dataclass
class MCPConfig:
    host: str = "0.0.0.0"
    port: int = 8765


class MarketConnector:
    """Connector that wraps Yahoo Finance for demo usage."""

    async def history(self, symbol: str, range_: str, interval: str) -> Dict[str, Any]:
        # yfinance is sync; run in thread executor to avoid blocking event loop.
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: yf.download(symbol, period=range_, interval=interval).reset_index())
        if data.empty:
            return {"symbol": symbol, "data": [], "source": "yahoo"}
        records = data.to_dict(orient="records")
        return {"symbol": symbol, "data": records, "source": "yahoo"}


class DocumentConnector:
    """Fetch documents via HTTP(S) and return base64 content."""

    async def fetch(self, uri: str, media_type: Optional[str] = None) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(uri, timeout=30.0)
            resp.raise_for_status()
            content_b64 = base64.b64encode(resp.content).decode("utf-8")
            return {"uri": uri, "media_type": media_type or resp.headers.get("content-type", "application/octet-stream"), "content_base64": content_b64}


class MCPServer:
    """Tiny MCP server that understands market/document request envelopes."""

    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self.market = MarketConnector()
        self.documents = DocumentConnector()

    async def dispatch(self, message: Dict[str, Any]) -> Dict[str, Any]:
        channel = message.get("channel", "")
        payload = message.get("message", {})
        action = payload.get("action")

        if channel.startswith("market"):
            if action == "history":
                params = payload.get("params", {})
                return await self.market.history(
                    symbol=params.get("symbol", "AAPL"),
                    range_=params.get("range", "1mo"),
                    interval=params.get("interval", "1d"),
                )
            return {"error": f"Unsupported market action: {action}"}

        if channel.startswith("document") or channel.startswith("doc"):
            if action == "fetch":
                return await self.documents.fetch(payload.get("uri", ""), payload.get("media_type"))
            return {"error": f"Unsupported document action: {action}"}

        return {"error": f"Unknown channel: {channel}"}

    async def handler(self, websocket: websockets.WebSocketServerProtocol) -> None:  # type: ignore[type-arg]
        async for raw in websocket:
            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "invalid JSON"}))
                continue

            response = await self.dispatch(envelope)
            await websocket.send(json.dumps(response))

    async def run(self) -> None:
        async with websockets.serve(self.handler, self.config.host, self.config.port):
            print(f"MCP server running at ws://{self.config.host}:{self.config.port}")
            await asyncio.Future()  # run forever


def start_server() -> None:
    config = MCPConfig()
    server = MCPServer(config)
    asyncio.run(server.run())


if __name__ == "__main__":
    start_server()
