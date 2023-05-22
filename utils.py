from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import Optional, List
from helper import get_abs_path
import os

index_folder = get_abs_path('faiss_index')


class ProxyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt


def init_knowledge_vector_store(path: str, embeddings):
    if os.path.exists(index_folder):
        try:
            print("start load faiss index")
            vector_store = FAISS.load_local("faiss_index", embeddings)
            print("load faiss index finished")
            return vector_store
        except Exception as err:
            print(f"load faiss index error: {err}")
    else:
        fold_path = path
        docs = []
        if not os.path.exists(fold_path):
            print(f"{fold_path} 路径不存在")
            return None
        elif os.path.isdir(fold_path):
            try:
                loader = DirectoryLoader(fold_path, glob='**/*.md')
                docs = loader.load()
                print(f"{fold_path} 已成功加载")
            except Exception as err:
                print(err)
                print(f"{fold_path} 未能成功加载")
        text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        # 切割加载的 document
        print("start split docs...")
        split_docs = text_splitter.split_documents(docs)
        print("split docs finished")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store


def init_chain_proxy(llm_proxy: LLM, vector_store, top_k=5):
    prompt_template = """你是一个专业的人工智能助手，以下是一些提供给你的已知内容，请你简洁和专业的来回答用户的问题，答案请使用中文。

已知内容:
{context}

参考以上内容请回答如下问题:
{question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    knowledge_chain = RetrievalQA.from_llm(
        llm=llm_proxy,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt,
        return_source_documents=True
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content", "source"], template="{page_content}(出处：{os.path.split(source)[-1]})"
    )
    return knowledge_chain
