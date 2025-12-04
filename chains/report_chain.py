from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


class ReportChain:
    """Generate human-friendly reports with tables and chart references."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
你是资深投研分析师，请基于输入的多智能体结果生成 Markdown 报告。
- 必须引用来源与时间戳
- 输出章节：摘要、证据表格、策略建议、风险、下一步行动
- 使用中文输出

摘要: {summary}
证据: {evidence}
表格CSV: {table_csv}
可视化: {charts}
"""
        )

    def build_table_csv(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "symbol,score,comment\n"
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def run(self, summary: str, evidence: str, table_rows: List[Dict[str, Any]], charts: List[str]) -> str:
        chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=StrOutputParser())
        return chain.run(
            summary=summary,
            evidence=evidence,
            table_csv=self.build_table_csv(table_rows),
            charts="; ".join(charts),
        )
