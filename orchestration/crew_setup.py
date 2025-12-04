from __future__ import annotations

from typing import Any, Dict, List

from crewai import Crew, Process, Task
from langchain.chat_models import ChatOpenAI

from agents.definitions import build_analysis_agents, build_ingestion_agents
from chains.report_chain import ReportChain
from chains.rag_chain import RAGPipeline


class FinancialCrewFactory:
    """Factory to build a hierarchical CrewAI workflow."""

    def __init__(self, llm: ChatOpenAI, rag: RAGPipeline, report_chain: ReportChain) -> None:
        self.llm = llm
        self.rag = rag
        self.report_chain = report_chain

    def build(self) -> Crew:
        ingestion_agents = build_ingestion_agents(self.llm)
        analysis_agents = build_analysis_agents(self.llm)
        all_agents = ingestion_agents + analysis_agents

        ingest_task = Task(
            description="并行抓取新闻/PDF/图片等多模态数据并清洗", agent=ingestion_agents[0], async_execution=True
        )
        pdf_task = Task(description="解析PDF/表格，提取关键财务指标", agent=ingestion_agents[1], async_execution=True)
        image_task = Task(description="OCR 图像并输出要点", agent=ingestion_agents[2], async_execution=True)

        technical_task = Task(
            description="基于检索证据与技术指标生成信号表", agent=analysis_agents[0], async_execution=True
        )
        fundamental_task = Task(
            description="结合财报与估值给出基本面结论", agent=analysis_agents[1], async_execution=True
        )
        risk_task = Task(description="监控风险与异常事件，给出警报", agent=analysis_agents[2], async_execution=True)

        report_task = Task(
            description="汇总所有结论，生成 Markdown 报告并附表格与可视化链接",
            agent=analysis_agents[3],
            async_execution=False,
            depends_on=[technical_task, fundamental_task, risk_task],
        )

        crew = Crew(
            agents=all_agents,
            tasks=[ingest_task, pdf_task, image_task, technical_task, fundamental_task, risk_task, report_task],
            process=Process.hierarchical,
            memory=True,
        )
        return crew

    def run_report(self, question: str, evidence: List[Any]) -> str:
        table_rows: List[Dict[str, Any]] = []
        for doc in evidence:
            table_rows.append(
                {
                    "symbol": doc.metadata.get("symbol", "N/A"),
                    "score": doc.metadata.get("score", 0),
                    "comment": doc.page_content[:80],
                }
            )
        charts = [doc.metadata.get("chart_uri", "") for doc in evidence if doc.metadata.get("chart_uri")]
        return self.report_chain.run(summary=question, evidence="\n".join([d.page_content for d in evidence]), table_rows=table_rows, charts=charts)
