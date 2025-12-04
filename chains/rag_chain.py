from __future__ import annotations

from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import BaseRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from data.preprocess import to_documents


class RAGPipeline:
    """Agentic RAG pipeline with dynamic routing and compression."""

    def __init__(self, llm: ChatOpenAI, persist_path: str = "./artifacts/vectorstore") -> None:
        self.llm = llm
        self.persist_path = persist_path
        self.store = InMemoryStore()
        self.vectorstore = Chroma(collection_name="finance", persist_directory=persist_path)
        self.router = self._build_router()

    def _build_router(self) -> LLMChain:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "判断问题类型: technical / fundamental / sentiment / macro。只输出一个类型关键词。",
                ),
                ("human", "问题: {question}"),
            ]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def _make_compressed_retriever(self, base: BaseRetriever) -> BaseRetriever:
        return ContextualCompressionRetriever.from_llm(llm=self.llm, base_retriever=base)

    def index(self, parsed_documents: List[Any]) -> MultiVectorRetriever:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        docs = to_documents(parsed_documents)
        splits = splitter.split_documents(docs)
        multivector_retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key="doc_id",
        )
        multivector_retriever.add_documents(splits)
        return multivector_retriever

    def retrieve(self, question: str, retriever: MultiVectorRetriever) -> List[Any]:
        category = self.router.run(question=question).strip().lower()
        compressed = self._make_compressed_retriever(retriever)
        if category == "technical":
            return compressed.get_relevant_documents(question + " 使用近期K线与指标数据")
        if category == "fundamental":
            return compressed.get_relevant_documents(question + " 聚焦财报、估值、现金流")
        if category == "sentiment":
            return compressed.get_relevant_documents(question + " 聚焦新闻舆情和时间衰减")
        return compressed.get_relevant_documents(question)

    def summarize_answers(self, question: str, docs: List[Any]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是金融分析师，严格基于提供的证据回答。无法回答则输出 UNKNOWN。格式: 摘要/驱动因素/风险/引用",
                ),
                ("human", "问题: {question}\n证据: {evidence}"),
            ]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        evidence = "\n\n".join([f"[{d.metadata.get('source','?')}] {d.page_content[:400]}" for d in docs])
        return chain.run(question=question, evidence=evidence)
