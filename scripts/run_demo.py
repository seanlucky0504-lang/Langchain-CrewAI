"""Example end-to-end run combining MCP ingestion, RAG, CrewAI and report generation."""
from __future__ import annotations

import asyncio
from typing import List

from langchain.chat_models import ChatOpenAI

from chains.ingestion_chain import IngestionChain
from chains.rag_chain import RAGPipeline
from chains.report_chain import ReportChain
from config.settings import load_settings
from data.preprocess import ParsedModal
from mcp.clients import DocumentRequest, MCPDocumentClient, MCPMarketClient
from orchestration.crew_setup import FinancialCrewFactory


def build_llm(settings) -> ChatOpenAI:
    return ChatOpenAI(model=settings.model, temperature=0.1, api_key=settings.deepseek_api_key)


def build_requests() -> List[DocumentRequest]:
    return [
        DocumentRequest(uri="s3://reports/q1.pdf", media_type="application/pdf"),
        DocumentRequest(uri="s3://reports/earnings.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        DocumentRequest(uri="https://cdn.example.com/chart.png", media_type="image/png"),
        DocumentRequest(uri="https://news.example.com/feed", media_type="text/plain"),
    ]


async def main() -> None:
    settings = load_settings()
    llm = build_llm(settings)

    doc_client = MCPDocumentClient(settings.mcp_doc_url)
    market_client = MCPMarketClient(settings.mcp_market_url)

    ingestion_chain = IngestionChain(doc_client, llm)
    rag = RAGPipeline(llm, persist_path=settings.vector_store_path)
    report_chain = ReportChain(llm)
    crew_factory = FinancialCrewFactory(llm, rag, report_chain)

    requests = build_requests()
    parsed: List[ParsedModal] = await ingestion_chain.fetch_and_parse(requests)
    summaries = await ingestion_chain.summarize_documents(parsed)
    retriever = rag.index(parsed)
    docs = rag.retrieve("请给出本周 AAPL 技术与风险结论", retriever)
    conclusion = rag.summarize_answers("请给出本周 AAPL 技术与风险结论", docs)

    report = crew_factory.run_report(conclusion, docs)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
