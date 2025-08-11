import os
import numpy as np
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import Dict, Any, List 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ZimLawRAGChain:
    def __init__(
        self,
        vectorstore_dir: str = "./vectorstore/faiss_index",
        model_name: str = "llama3",
        model_path: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_k: int = 20,  # Increased initial retrieval
        final_k: int = 4,  # Number of documents after re-ranking
        reranker_model: str = "BAAI/bge-reranker-base" # Can use large with more mem
    ):
        # Add existing initialization parameters
        self.top_k = top_k
        self.final_k = final_k
        self.reranker_model = reranker_model
        self._reranker = None
        self.model_configs = {
            "deepseek": {
                "name": "deepseek-r1:8b",
                "context_length": 8192 
            },
            "llama2": {
                "name": "llama2:7b",
                "context_length": 4096
            },
            "llama3": {
                "name": "llama3:8b",
                "context_length": 8192
            }
        }
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(self.model_configs.keys())}")
        
        self.model_name = model_name
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vectorstore_dir = vectorstore_dir
        self.top_k = top_k
        
        # Initialize components
        self.vectorstore = self._load_vectorstore()
        self.llm = self._initialize_llm()
        self.retriever = self._create_retriever()
        self.chain = self._create_chain()
    
    def _get_reranker(self):
        """Lazy loading of the re-ranker model"""
        if self._reranker is None:
            print("📥 Loading re-ranker model...")
            self._reranker = CrossEncoder(self.reranker_model, max_length=512)
            print("✅ Re-ranker model loaded")
        return self._reranker
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using the cross-encoder model"""
        if not docs:
            return []

        # Prepare query-document pairs for the re-ranker
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Get relevance scores
        reranker = self._get_reranker()
        scores = reranker.predict(pairs)

        # Sort documents by score
        if isinstance(scores[0], np.ndarray):
            scores = [s[0] for s in scores]
        
        # Create (score, doc) pairs and sort
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Take top k documents
        reranked_docs = [doc for _, doc in scored_docs[:self.final_k]]
        
        print(f"🔄 Re-ranked {len(docs)} documents to select top {len(reranked_docs)}")
        return reranked_docs

    def _create_retriever(self):
        """Create the retriever with improved search parameters"""
        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.top_k,
                "fetch_k": self.top_k * 2,  # Fetch more for diversity
                "lambda_mult": 0.3,  # MMR diversity factor
                "score_threshold": 0.5,
            },
            search_type="mmr"  # Use MMR for initial diversity
        )

    def _load_vectorstore(self) -> FAISS:
        """Load the FAISS vectorstore"""
        if not os.path.exists(self.vectorstore_dir):
            raise FileNotFoundError(
                f"Vectorstore not found at {self.vectorstore_dir}. "
                "Please run ingest.py first."
            )
        
        print("🔍 Loading vectorstore...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return FAISS.load_local(
            self.vectorstore_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )

    def _initialize_llm(self) -> OllamaLLM:
        """Initialize the chosen LLM via Ollama"""
        model_config = self.model_configs[self.model_name]
        model_path = self.model_path or model_config["name"]
        
        print(f"🤖 Initializing {self.model_name} model...")
        return OllamaLLM(
            model=model_path,
            temperature=self.temperature,
            num_ctx=model_config["context_length"],
            num_predict=self.max_tokens,
            callbacks=[StreamingStdOutCallbackHandler()],
            stop=["Human:", "Assistant:"]
        )

    def _create_retriever(self):
        """Create the retriever with custom search parameters"""
        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.top_k,
                "fetch_k": self.top_k * 4,
                "lambda_mult": 0.3,  # Adjust MMR diversity
                "score_threshold": 0.5,  # Minimum relevance score
            },
            search_type="similarity_score_threshold"  # Use MMR for diverse results
        )

    def _create_prompt_template(self) -> PromptTemplate:
        """Create an improved prompt template for legal Q&A"""
        template = """You are a legal assistant specialized in Zimbabwean law. Your role is to provide accurate, clear, and well-referenced legal information based on official legal documents.

        Question: {question}

        Relevant legal context:
        {context}

        Instructions:
        1. Analyze ALL provided legal sections thoroughly
        2. Cite EVERY relevant section number and act name
        3. Structure your answer with clear headings and bullet points
        4. If rights or procedures are listed in the law, enumerate them all
        5. Provide your answer in 4 parts:
            - Immediate answer to the question
            - Summary of relevant legal sections
            - List of rights or procedures mentioned in the law
            - Additional relevant sections or acts and suggestions for further reading
        5. Include cross-references between sections when relevant
        6. If the context is incomplete, indicate what additional legal sections might be relevant

        Your professional legal response:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_chain(self) -> RetrievalQA:
        """Create the RAG chain with the prompt template"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self._create_prompt_template(),
                "verbose": True
            },
            return_source_documents=True
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Enhanced answer generation with re-ranking"""
        try:
            print(f"\n❓ Question: {question}")
            
            # 1. Initial retrieval
            initial_docs = self.retriever.get_relevant_documents(
                question, 
                k=self.top_k
            )
            print(f"📚 Retrieved {len(initial_docs)} initial documents")
            
            # 2. Re-rank documents
            reranked_docs = self._rerank_documents(question, initial_docs)
            
            # 3. Create a temporary retriever with re-ranked documents
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            # 4. Generate answer using re-ranked documents
            result = self.chain.invoke({
                "query": question,
                "context": context
            })
            
            # Process sources
            sources = []
            for doc in reranked_docs:  # Use re-ranked docs for sources
                sources.append({
                    "act": doc.metadata.get("act"),
                    "chapter": doc.metadata.get("chapter"),
                    "section": doc.metadata.get("section_number"),
                    "title": doc.metadata.get("section_title"),
                    "relevance": "High"  # Adding relevance indicator
                })
            
            return {
                "question": question,
                "answer": result.get("result", "No answer generated"),
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return {
                "question": question,
                "answer": "Sorry, I encountered an error while processing your question.",
                "error": str(e)
            }

def main():
    """Test the RAG chain with sample legal questions"""
    rag_chain = ZimLawRAGChain()
    
    test_questions = [
        "What are my constitutional rights if I'm arrested?",
        "What is the process for registering a trade union?",
        "What are the requirements for fair dismissal under the Labour Act?",
        "What are the fundamental rights protected in Chapter 4 of the Constitution?"
    ]
    
    for question in test_questions:
        result = rag_chain.answer_question(question)
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        if "sources" in result:
            print("\nSources:")
            for source in result['sources']:
                print(f"- {source['act']}, {source['chapter']}, Section {source['section']}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()