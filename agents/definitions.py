from __future__ import annotations

from typing import List

from crewai import Agent
from langchain.chat_models import ChatOpenAI


def build_ingestion_agents(llm: ChatOpenAI) -> List[Agent]:
    news_agent = Agent(
        role="新闻舆情采集",
        goal="抓取并摘要实时新闻，标注情绪与来源",
        backstory="熟悉财经媒体和社交渠道，擅长热点聚类",
        allow_delegation=True,
        llm=llm,
    )
    pdf_agent = Agent(
        role="财报PDF解析",
        goal="读取PDF/表格，提取关键财务指标",
        backstory="具备财报阅读经验，能识别表格和文字",
        allow_delegation=True,
        llm=llm,
    )
    image_agent = Agent(
        role="图表/OCR",
        goal="解析行情截图、图表图片，输出文字和要点",
        backstory="使用 OCR 和视觉模型读取图像内容",
        allow_delegation=False,
        llm=llm,
    )
    return [news_agent, pdf_agent, image_agent]


def build_analysis_agents(llm: ChatOpenAI) -> List[Agent]:
    technical_agent = Agent(
        role="技术面分析",
        goal="根据K线、成交量与技术指标生成信号",
        backstory="擅长趋势、形态与动量分析",
        allow_delegation=False,
        llm=llm,
    )
    fundamental_agent = Agent(
        role="基本面分析",
        goal="结合财务数据、估值与行业对标给出判断",
        backstory="具有券商研究背景，熟悉财务建模",
        allow_delegation=False,
        llm=llm,
    )
    risk_agent = Agent(
        role="风险监控",
        goal="监测波动率、新闻风险与暴露度，给出风险提示",
        backstory="风险量化专家，关注极端情景与合规",
        allow_delegation=False,
        llm=llm,
    )
    report_agent = Agent(
        role="报告生成",
        goal="整合所有结论，生成结构化报告并输出表格/可视化链接",
        backstory="投研首席，擅长写作与总结",
        allow_delegation=True,
        llm=llm,
    )
    return [technical_agent, fundamental_agent, risk_agent, report_agent]
