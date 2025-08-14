import os
from tempfile import template
import numpy as np
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import Dict, Any, List, Optional, Tuple
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
        model_name: str = "deepseek",  # Options: 'deepseek', 'llama2', 'llama3'
        model_path: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        top_k: int = 10,  # Increased initial retrieval
        final_k: int = 4,  # Number of documents after re-ranking
        reranker_model: str = "BAAI/bge-reranker-base", # Can use large with more mem
        enable_query_rewriting: bool = True
    ):
        # Add existing initialization parameters
        self.top_k = top_k
        self.final_k = final_k
        self.enable_query_rewriting = enable_query_rewriting
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
        
    def _get_query_rewrite_prompt(self) -> str:
        """Returns the prompt template for query rewriting"""
        return """You are a legal query assistant for Zimbabwean law. Your task is to rewrite the user's question into a clear, formal, and legally precise query suitable for searching Zimbabwean legal documents.

        Original Question: {query}

        Instructions:
        1. Identify the core legal concepts and issues
        2. Use appropriate legal terminology from Zimbabwean law
        3. Make the query more specific and actionable
        4. Preserve the original intent of the question
        5. Do not assume Acts, Chapters, or sections of the legislation unless otherwise stated in the original query
        6. Use clear and concise language and limit to 200 words
        7. Respond by only giving the rewritten query, nothing more!

        Rewritten Query:
        """
    
    def _rewrite_query(self, query: str) -> Tuple[str, str]:
        """Rewrites the user query into a more formal legal query"""
        if not self.enable_query_rewriting:
            return query, None

        try:
            print("🔄 Rewriting query for legal search...")
            
            # Format the rewrite prompt
            prompt = self._get_query_rewrite_prompt().format(query=query)
            
            # Get rewritten query from LLM
            rewritten = self.llm.invoke(prompt).strip()
            
            # Validate rewritten query
            if not rewritten or len(rewritten) < 10:
                print("⚠️ Query rewriting produced invalid result, using original query")
                return query, None
                
            print(f"📝 Original: {query}")
            print(f"✨ Rewritten: {rewritten}")
            
            return rewritten, query  # Return both rewritten and original
            
        except Exception as e:
            print(f"❌ Error in query rewriting: {str(e)}")
            return query, None
     
    def _create_prompt_template(self) -> PromptTemplate:
        """Create an improved prompt template for legal Q&A"""
        template = """You are a precise and reliable legal assistant for Zimbabwean law. Your task is to answer the user's question using ONLY the information provided in the "Relevant legal context" section.
    
        ## User Question
        {question}
    
        ## Relevant Legal Context
        {context}
    
        ## Instructions
        1. **Answer Directly First:** Begin with a clear, concise answer to the user's question. If the context does not contain enough information, state: "I cannot answer that question based on the provided information."
        2. **Cite Sources Explicitly:** For every fact or right mentioned in your answer, cite the specific source. Use the format: "[Act Name, Section X]" or "[Constitution, Section Y]". Do not invent citations. ONLY USE THE SOURCES FROM THE CONTEXT PROVIDED
        3. **Structure Your Response:** Organize your answer into the following sections:
            - **Direct Answer:** A brief, direct response to the question.
            - **Legal Basis:** A summary of the relevant law, quoting key phrases if necessary, with full citations.
            - **Key Rights/Procedures:** If the law enumerates rights or procedures, list them clearly as bullet points, each with its citation.
            - **Additional Notes:** Mention any related sections or acts that might be relevant for further research, but only if they are mentioned in the context.
        4. **Be Comprehensive but Concise:** Do not omit any relevant information from the context, but avoid unnecessary verbosity.
        5. **Do NOT Speculate:** If a part of the law is missing from the context, do not guess what it says. Simply state that the information is not available in the provided context.
    
        ## Your Response
        """

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
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
        """Enhanced answer generation with query rewriting"""
        try:
            print(f"\n❓ Received question: {question}")
            
            # 1. Rewrite the query
            # rewritten_query, original_query = self._rewrite_query(question)
            
            # 2. Initial retrieval with rewritten query
            initial_docs = self.retriever.invoke(
                question, 
                k=self.top_k
            )
            print(f"📚 Retrieved {len(initial_docs)} initial documents")
            
            # 3. Re-rank documents
            reranked_docs = self._rerank_documents(question, initial_docs)
            
            # 4. Create context from reranked documents
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            # 5. Generate answer using both queries
            result = self.chain.invoke({
                "query": question,
                "context": context
            })
            
            # Process sources
            sources = []
            for doc in reranked_docs:
                sources.append({
                    "act": doc.metadata.get("act"),
                    "chapter": doc.metadata.get("chapter"),
                    "section": doc.metadata.get("section_number"),
                    "title": doc.metadata.get("section_title"),
                    "relevance": "High"
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
        "What are my constitutional rights if I'm arrested or detained?",
        "Can my boss fire me for being late?",
        "What is the process for registering a trade union?",
        "What are the requirements for fair dismissal under the Labour Act?",
        "How can I legally change my name in Zimbabwe?"
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