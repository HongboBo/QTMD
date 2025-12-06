# rag_llm_dynamic.py - 支持动态上传文件的RAG系统
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from typing import Dict, List, Optional
import tempfile
import os


class DynamicMultiAgentRAG:
    """
    支持用户动态上传文件的多Agent RAG系统

    特性：
    1. 支持为每个agent单独上传知识库文件
    2. 支持多种文件格式（txt, md, pdf, docx）
    3. 可以使用默认知识库或用户上传的知识库
    4. 实时构建向量数据库
    """

    # 默认知识库文件路径
    DEFAULT_KB_FILES = {
        "ConservationAgent": "RAG/v2/ConservationAgent.txt",
        "FarmerAgent": "RAG/v2/FarmerAgent.txt",
        "CommunityAgent": "RAG/v2/CommunityAgent.txt"
    }

    def __init__(self):
        # 使用轻量级的embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        # 存储各agent的向量数据库
        self.agent_vectorstores: Dict[str, FAISS] = {}

        # 标记是否使用默认知识库
        self.using_default: Dict[str, bool] = {}

    def load_default_knowledge_base(self, agent: str):
        """加载默认知识库"""
        if agent not in self.DEFAULT_KB_FILES:
            raise ValueError(f"Unknown agent: {agent}")

        file_path = self.DEFAULT_KB_FILES[agent]

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"Warning: Default KB file not found for {agent}: {file_path}")
            # 创建空的向量库
            self.agent_vectorstores[agent] = FAISS.from_documents(
                [Document(page_content="Default knowledge base is empty.")],
                self.embeddings
            )
            self.using_default[agent] = True
            return

        # 加载并构建向量库
        loader = TextLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        vectordb = FAISS.from_documents(split_docs, self.embeddings)

        self.agent_vectorstores[agent] = vectordb
        self.using_default[agent] = True
        print(f"✅ Loaded default KB for {agent}: {len(split_docs)} chunks")

    def load_from_uploaded_file(
            self,
            agent: str,
            file_content: str,
            file_name: str = "uploaded.txt"
    ):
        """
        从上传的文件内容构建向量库

        Args:
            agent: Agent名称
            file_content: 文件内容（字符串）
            file_name: 文件名（用于日志）
        """
        if not file_content.strip():
            print(f"Warning: Empty file content for {agent}")
            return False

        # 创建文档对象
        doc = Document(
            page_content=file_content,
            metadata={"source": file_name, "agent": agent}
        )

        # 分割文本
        split_docs = self.text_splitter.split_documents([doc])

        if not split_docs:
            print(f"Warning: No chunks created for {agent}")
            return False

        # 构建向量库
        vectordb = FAISS.from_documents(split_docs, self.embeddings)

        self.agent_vectorstores[agent] = vectordb
        self.using_default[agent] = False

        print(f"✅ Loaded custom KB for {agent}: {len(split_docs)} chunks from {file_name}")
        return True

    def load_from_multiple_files(
            self,
            agent: str,
            file_contents: List[tuple]  # [(content, filename), ...]
    ):
        """
        从多个上传的文件构建向量库

        Args:
            agent: Agent名称
            file_contents: 文件内容和文件名的列表
        """
        all_docs = []

        for content, filename in file_contents:
            if not content.strip():
                continue

            doc = Document(
                page_content=content,
                metadata={"source": filename, "agent": agent}
            )
            all_docs.append(doc)

        if not all_docs:
            print(f"Warning: No valid files for {agent}")
            return False

        # 分割所有文档
        split_docs = self.text_splitter.split_documents(all_docs)

        # 构建向量库
        vectordb = FAISS.from_documents(split_docs, self.embeddings)

        self.agent_vectorstores[agent] = vectordb
        self.using_default[agent] = False

        print(f"✅ Loaded custom KB for {agent}: {len(split_docs)} chunks from {len(file_contents)} files")
        return True

    def ensure_agent_kb_loaded(self, agent: str):
        """确保agent的知识库已加载（如果没有则加载默认）"""
        if agent not in self.agent_vectorstores:
            print(f"Loading default KB for {agent}...")
            self.load_default_knowledge_base(agent)

    def rag_search(self, query: str, agent: str, top_k: int = 3) -> List[dict]:
        """
        RAG检索

        Args:
            query: 查询文本
            agent: Agent名称
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        # 确保知识库已加载
        self.ensure_agent_kb_loaded(agent)

        if agent not in self.agent_vectorstores:
            return []

        # 检索
        results = self.agent_vectorstores[agent].similarity_search_with_score(
            query, k=top_k
        )

        return [
            {
                "score": float(score),
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown")
            }
            for doc, score in results
        ]

    def reset_to_default(self, agent: str):
        """重置为默认知识库"""
        self.load_default_knowledge_base(agent)

    def get_kb_info(self, agent: str) -> dict:
        """获取知识库信息"""
        self.ensure_agent_kb_loaded(agent)

        if agent not in self.agent_vectorstores:
            return {
                "agent": agent,
                "loaded": False,
                "using_default": True,
                "num_chunks": 0
            }

        # 获取向量库中的文档数量
        vectorstore = self.agent_vectorstores[agent]
        num_chunks = vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else 0

        return {
            "agent": agent,
            "loaded": True,
            "using_default": self.using_default.get(agent, True),
            "num_chunks": num_chunks
        }


# 全局实例（用于向后兼容）
_dynamic_rag = DynamicMultiAgentRAG()


def rag_search(query: str, agent: str, top_k: int = 3) -> List[dict]:
    """全局RAG搜索接口（向后兼容）"""
    return _dynamic_rag.rag_search(query, agent, top_k)


def get_rag_instance() -> DynamicMultiAgentRAG:
    """获取RAG实例"""
    return _dynamic_rag


# 文件读取工具函数
def read_file_content(file_bytes, file_name: str) -> str:
    """
    读取上传文件的内容

    支持的格式：
    - .txt, .md: 纯文本
    - .pdf: PDF文档（需要pypdf）
    - .docx: Word文档（需要python-docx）
    """
    file_ext = os.path.splitext(file_name)[1].lower()

    try:
        if file_ext in ['.txt', '.md']:
            # 纯文本文件
            return file_bytes.decode('utf-8')

        elif file_ext == '.pdf':
            # PDF文件
            try:
                import PyPDF2
                import io

                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                return f"Error: PyPDF2 not installed. Cannot read PDF files.\nInstall with: pip install PyPDF2"

        elif file_ext == '.docx':
            # Word文档
            try:
                import docx
                import io

                doc = docx.Document(io.BytesIO(file_bytes))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except ImportError:
                return f"Error: python-docx not installed. Cannot read DOCX files.\nInstall with: pip install python-docx"

        else:
            return f"Unsupported file format: {file_ext}"

    except Exception as e:
        return f"Error reading file {file_name}: {str(e)}"


# 使用示例
if __name__ == "__main__":
    # 创建RAG实例
    rag = DynamicMultiAgentRAG()

    # 示例1：使用默认知识库
    rag.load_default_knowledge_base("ConservationAgent")
    results = rag.rag_search("forest protection", "ConservationAgent")
    print("Default KB results:", results)

    # 示例2：上传自定义内容
    custom_content = """
    Sustainable forestry practices include:
    1. Selective logging to preserve forest structure
    2. Reforestation programs
    3. Protection of old-growth forests
    4. Wildlife corridor maintenance
    """
    rag.load_from_uploaded_file("ConservationAgent", custom_content, "custom.txt")
    results = rag.rag_search("forest protection", "ConservationAgent")
    print("Custom KB results:", results)

    # 示例3：查看知识库信息
    info = rag.get_kb_info("ConservationAgent")
    print("KB Info:", info)