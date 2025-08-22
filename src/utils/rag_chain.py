import os
import numpy as np
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ZimLawRAGChain:
    def __init__(
        self,
        vectorstore_dir: str = "./vectorstore/faiss_index",
        model_name: str = "llama3",  # Options: 'deepseek', 'llama2', 'llama3'
        model_path: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        top_k: int = 15,  # Reasonable initial retrieval
        final_k: int = 8,  # Final documents for context
        reranker_model: str = "BAAI/bge-reranker-large",  # Lighter model
        enable_query_rewriting: bool = True,
        use_hybrid_search: bool = True,
        verbose: bool = False  # Control debug output
    ):
        # Store configuration
        self.top_k = top_k
        self.final_k = final_k
        self.enable_query_rewriting = enable_query_rewriting
        self.reranker_model = reranker_model
        self.use_hybrid_search = use_hybrid_search
        self.verbose = verbose  # Store verbose setting
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
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Optimized document retrieval with semantic search and optional reranking"""
        try:
            if self.verbose:
                print(f"🔍 Retrieving documents for: {query[:60]}...")
            
            # Use similarity search with score threshold
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, 
                k=self.top_k
            )
            
            # Extract documents and filter by relevance threshold
            documents = []
            for doc, score in docs_with_scores:
                # Lower scores mean higher similarity in FAISS
                if score < 1.0:  # Reasonable threshold for good matches
                    documents.append(doc)
            
            if self.verbose:
                print(f"📚 Retrieved {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error in document retrieval: {str(e)}")
            return []
        
    def _get_reranker(self):
        """Lazy loading of the re-ranker model"""
        if self._reranker is None:
            if self.verbose:
                print("📥 Loading re-ranker model...")
            self._reranker = CrossEncoder(self.reranker_model, max_length=512)
            if self.verbose:
                print("✅ Re-ranker model loaded")
        return self._reranker
    
    def _rewrite_query(self, query: str) -> str:
        """Simplified query rewriting for better legal search"""
        if not self.enable_query_rewriting:
            return query

        try:
            if self.verbose:
                print("🔄 Rewriting query for legal search...")
            
            # Improved rewrite prompt - more explicit about not assuming specifics
            prompt = f"""You are a legal assistant for Zimbabwean law. Rewrite this question to be more precise for searching legal documents:

Original: {query}

Rewrite the query to:
1. Use formal legal terminology where appropriate
2. Be clear and specific about what legal information is needed
3. Keep it concise (1-2 sentences max)
4. Focus on the core legal issue
5. NEVER reference specific sections, chapters, or acts unless they were explicitly mentioned in the original query
6. Use general terms like "under Zimbabwean law" instead of specific legislation names

Example:
- Original: "Can a child be charged with a crime?"
- Good rewrite: "What is the minimum age of criminal responsibility under Zimbabwean criminal law?"
- Bad rewrite: "Is a child under age 7 (as per Section 6 of Criminal Law Act) criminally liable?"

Rewritten query:"""
            
            # Get rewritten query from LLM
            rewritten = self.llm.invoke(prompt).strip()
            
            # Validate rewritten query
            if not rewritten or len(rewritten) < 10:
                if self.verbose:
                    print("⚠️ Using original query")
                return query
                
            if self.verbose:
                print(f"✨ Rewritten: {rewritten}")
            return rewritten
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Query rewriting failed, using original: {str(e)}")
            return query

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using the cross-encoder model"""
        if not docs:
            return []

        if self.verbose:
            print(f"🔄 Re-ranking {len(docs)} documents...")
        
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
        
        if self.verbose:
            print(f"✅ Selected top {len(reranked_docs)} documents")
        return reranked_docs

    def _load_vectorstore(self) -> FAISS:
        """Load the FAISS vectorstore with optimized settings"""
        if not os.path.exists(self.vectorstore_dir):
            raise FileNotFoundError(
                f"Vectorstore not found at {self.vectorstore_dir}. "
                "Please run ingest.py first."
            )
        
        if self.verbose:
            print("🔍 Loading vectorstore...")
        
        # Use CPU for embeddings to ensure compatibility
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},  # Use CPU for stability
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}
        )
        
        vectorstore = FAISS.load_local(
            self.vectorstore_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        if self.verbose:
            print(f"✅ Vectorstore loaded with {vectorstore.index.ntotal} documents")
        return vectorstore

    def _initialize_llm(self) -> Ollama:
        """Initialize the chosen LLM via Ollama"""
        model_config = self.model_configs[self.model_name]
        model_path = self.model_path or model_config["name"]
        
        if self.verbose:
            print(f"🤖 Initializing {self.model_name} model...")
        return Ollama(
            model=model_path,
            temperature=self.temperature,
            num_ctx=model_config["context_length"],
            num_predict=self.max_tokens,
            callbacks=[StreamingStdOutCallbackHandler()],
            stop=["Human:", "Assistant:"]
        )

    def _create_prompt_template(self) -> PromptTemplate:
        """Create an optimized prompt template for legal Q&A"""
        template = """You are a knowledgeable legal assistant for Zimbabwean law. Answer the user's question using ONLY the provided legal context.

## User Question
{question}

## Relevant Legal Context
{context}

## Instructions
1. **Direct Answer**: Start with a clear, direct answer. If insufficient information, state: "Based on the provided information, I cannot fully answer this question."
2. **Legal Basis**: Provide the relevant law with proper citations in format: [Act Name, Section X]
3. **Key Points**: List important rights, procedures, or requirements as bullet points with citations
4. **Additional Notes**: Mention related provisions only if they appear in the context

## Important
- Use ONLY information from the context provided
- Cite sources accurately using the exact act and section names from context
- Do not speculate or add information not in the context
- Be precise and comprehensive but concise

## Response
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_retriever(self):
        """Create an optimized retriever"""
        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.top_k,
                "fetch_k": self.top_k * 2,  # Fetch more candidates
                "score_threshold": 0.5,  # Reasonable threshold
            },
            search_type="similarity"  # Use similarity for better precision
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
        """Simplified and optimized answer generation"""
        try:
            if self.verbose:
                print(f"\n❓ Question: {question}")
            
            # 1. Optionally rewrite the query for better legal search
            search_query = self._rewrite_query(question)
            
            # 2. Retrieve relevant documents
            initial_docs = self._retrieve_documents(search_query)
            
            if not initial_docs:
                return {
                    "question": question,
                    "search_query": search_query,
                    "answer": "I could not find relevant information to answer your question.",
                    "sources": [],
                    "num_documents_used": 0
                }
            
            # 3. Re-rank documents for better relevance
            final_docs = self._rerank_documents(question, initial_docs)
            
            # 4. Create context from selected documents
            context = "\n\n".join([doc.page_content for doc in final_docs])
            
            # 5. Generate answer using the original question
            prompt_template = self._create_prompt_template()
            formatted_prompt = prompt_template.format(
                question=question,
                context=context
            )
            
            if self.verbose:
                print("🤖 Generating answer...")
            answer = self.llm.invoke(formatted_prompt)
            
            # 6. Extract source information
            sources = []
            for doc in final_docs:
                source_info = {
                    "act": doc.metadata.get("act", "Unknown Act"),
                    "chapter": doc.metadata.get("chapter", "Unknown Chapter"),
                    "section": doc.metadata.get("section_number", "Unknown"),
                    "title": doc.metadata.get("section_title", ""),
                    "content_type": doc.metadata.get("content_type", "general")
                }
                sources.append(source_info)
            
            return {
                "question": question,
                "search_query": search_query if search_query != question else None,
                "answer": answer,
                "sources": sources,
                "num_documents_retrieved": len(initial_docs),
                "num_documents_used": len(final_docs)
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return {
                "question": question,
                "answer": "Sorry, I encountered an error while processing your question.",
                "error": str(e),
                "sources": []
            }


# Tests moved to separate test file - no automatic execution