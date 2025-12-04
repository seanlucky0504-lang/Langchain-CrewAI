from __future__ import annotations

from typing import Any, Dict, Iterable, List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from config.settings import load_settings
from data.preprocess import ParsedModal, parse_image, parse_pdf, parse_table, parse_text, to_documents
from mcp.clients import DocumentRequest, MCPDocumentClient


class IngestionChain:
    """Ingest multi-modal data via MCP message bus and normalize into LangChain documents."""

    def __init__(self, doc_client: MCPDocumentClient, llm: ChatOpenAI) -> None:
        self.doc_client = doc_client
        self.llm = llm
        self._summarizer = self._build_summarizer()

    def _build_summarizer(self) -> LLMChain:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是金融数据预处理助手，清洗文本并输出摘要，用中文，50词内。"),
                ("human", "原文: {text}\n请输出摘要"),
            ]
        )
        return LLMChain(llm=self.llm, prompt=prompt, output_parser=StrOutputParser())

    async def fetch_and_parse(self, requests: Iterable[DocumentRequest]) -> List[ParsedModal]:
        parsed: List[ParsedModal] = []
        for req in requests:
            content = await self.doc_client.fetch(req)
            if req.media_type == "application/pdf":
                parsed.append(parse_pdf(content, req.uri))
            elif req.media_type in {"application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}:
                parsed.append(parse_table(content, req.uri))
            elif req.media_type.startswith("image/"):
                parsed.append(parse_image(content, req.uri))
            else:
                parsed.append(parse_text(content.decode("utf-8"), req.uri))
        return parsed

    async def summarize_documents(self, parsed: List[ParsedModal]) -> List[Dict[str, Any]]:
        docs = to_documents(parsed)
        summaries: List[Dict[str, Any]] = []
        for doc in docs:
            summary = await self._summarizer.arun(text=doc.page_content)
            summaries.append({"summary": summary, "metadata": doc.metadata})
        return summaries
