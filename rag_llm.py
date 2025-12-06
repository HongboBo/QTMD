from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader



# 多 Agent 知识库文件路径
AGENT_KB_FILES = {
    "ConservationAgent": "RAG/v2/ConservationAgent.txt",
    "FarmerAgent": "RAG/v2/FarmerAgent.txt",
    "CommunityAgent": "RAG/v2/CommunityAgent.txt"
}

class MultiAgentRAG:
    def __init__(self, agent_kb_files=AGENT_KB_FILES):
        self.agent_kb_files = agent_kb_files
        self.agent_vectorstores = {}
        # self.embeddings = OllamaEmbeddings(model="qwen3:latest", base_url="http://localhost:11434")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # encode_kwargs 可按需改，比如批大小
            encode_kwargs={"normalize_embeddings": True}
        )
        self._build_all_agents_vectorstores()

    def _build_all_agents_vectorstores(self):
        for agent, file_path in self.agent_kb_files.items():
            loader = TextLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)
            vectordb = FAISS.from_documents(split_docs, self.embeddings)
            self.agent_vectorstores[agent] = vectordb

    def rag_search(self, query, agent, top_k=3):
        if agent not in self.agent_vectorstores:
            raise ValueError(f"Agent {agent} not found.")
        results = self.agent_vectorstores[agent].similarity_search_with_score(query, k=top_k)
        return [
            {"score": float(score), "content": doc.page_content}
            for doc, score in results
        ]

# 单例初始化，供其他模块直接使用
rag = MultiAgentRAG()

# 对外统一接口
def rag_search(query, agent, top_k=3):
    return rag.rag_search(query, agent, top_k)