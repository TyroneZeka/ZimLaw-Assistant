import os
from tempfile import template
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
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
        model_name: str = "llama3",  # Options: 'deepseek', 'llama2', 'llama3'
        model_path: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        top_k: int = 20,  # Increased initial retrieval
        final_k: int = 10,  # Number of documents after re-ranking
        reranker_model: str = "BAAI/bge-reranker-large", # Can use large with more mem
        enable_query_rewriting: bool = True,
        bm25_weight: float = 0.3
    ):
        # Add existing initialization parameters
        self.bm25_weight = bm25_weight
        self._bm25_index = None
        self._document_store = []
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
        
    def _initialize_bm25(self):
        """Initialize BM25 index from documents"""
        if self._bm25_index is None:
            print("📑 Initializing BM25 index...")
            try:
                # Get documents from vectorstore using similarity search
                # Use a very large k to get all documents
                all_docs = self.vectorstore.similarity_search(
                    "", 
                    k=10000  # Large enough to get all docs
                )

                # Store documents for later reference
                self._document_store = all_docs

                # Tokenize documents for BM25
                tokenized_docs = [
                    doc.page_content.lower().split() 
                    for doc in self._document_store
                ]

                # Create BM25 index
                self._bm25_index = BM25Okapi(tokenized_docs)
                print(f"✅ BM25 index initialized with {len(all_docs)} documents")

            except Exception as e:
                print(f"❌ Error initializing BM25 index: {str(e)}")
                raise
            
    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        """Perform BM25 keyword search"""
        self._initialize_bm25()
        # Tokenize query
        tokenized_query = query.lower().split()
        # Get BM25 scores
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        # Get top-k document indices
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        # Return documents with scores
        return [
            (self._document_store[idx], bm25_scores[idx])
            for idx in top_indices
            if bm25_scores[idx] > 0
        ]
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Document]:
        """Combine semantic and keyword search results"""
        # Get semantic search results
        semantic_results = self.vectorstore.similarity_search_with_score(
            query, 
            k=top_k
        )
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, top_k)
        
        # Normalize scores
        max_semantic = max(score for _, score in semantic_results) if semantic_results else 1
        max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1
        
        # Combine and normalize scores
        combined_scores = defaultdict(float)
        
        # Add semantic scores
        for doc, score in semantic_results:
            normalized_score = score / max_semantic
            combined_scores[doc.page_content] += (1 - self.bm25_weight) * normalized_score
        
        # Add BM25 scores
        for doc, score in bm25_results:
            normalized_score = score / max_bm25
            combined_scores[doc.page_content] += self.bm25_weight * normalized_score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get final documents
        final_docs = []
        seen = set()
        for content, score in sorted_results:
            if len(final_docs) >= top_k:
                break
            
            # Find original document
            for doc in self._document_store:
                if doc.page_content == content and doc.page_content not in seen:
                    final_docs.append(doc)
                    seen.add(doc.page_content)
                    break
                
        return final_docs
        
    def _get_query_rewrite_prompt(self) -> str:
        """Returns the prompt template for query rewriting"""
        return """You are a legal query assistant for Zimbabwean law. Your task is to rewrite the user's question into a clear, formal, and legally precise query suitable for searching Zimbabwean legal documents.

        Original Question: {query}

        Instructions:
        1. Rewrite the query to be more formal and precise in legal terminology
        2. Identify the core legal concepts and issues
        3. Use appropriate legal terminology ONLY from Zimbabwean law
        4. Make the query more specific and actionable
        5. Preserve the original intent of the question
        6. Do NOT assume Acts, Chapters, or sections of the legislation unless otherwise stated in the original query
        7. Use clear and concise language and limit to 150 words
        8. Respond by only giving the rewritten query,NO NOTE, NOTHING MORE!
        9. No need to explain your reasoning!!

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

    def _get_query_variations_prompt(self) -> str:
        """Returns the prompt template for generating query variations"""
        return """You are a legal query assistant for Zimbabwean law. Your task is to generate 5 different variations of the user's question that would help retrieve relevant legal documents from different angles.

        Original Question: {query}

        Instructions:
        1. Generate 5 variations that approach the same legal issue from different perspectives
        2. Each variation should be distinct but related to the core legal question
        3. Use appropriate Zimbabwean legal terminology
        4. Focus on different aspects like:
           - Rights and obligations
           - Procedures and processes
           - Penalties and consequences
           - Definitions and requirements
           - Related legal concepts
        5. Keep each variation concise (1-2 sentences)
        6. Respond with ONLY the 5 variations, one per line, numbered 1-5
        7. Do NOT include explanations or notes
        8. Do NOT assume Acts, Chapters, or sections of the legislation unless otherwise stated in the original query

        Query Variations:
        """

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate 5 variations of the original query for multi-query retrieval"""
        try:
            print("🔄 Generating query variations for multi-query retrieval...")
            
            # Format the variations prompt
            prompt = self._get_query_variations_prompt().format(query=query)
            
            # Get variations from LLM
            variations_text = self.llm.invoke(prompt).strip()
            
            # Parse the variations
            variations = []
            lines = variations_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                           line.startswith(('1)', '2)', '3)', '4)', '5)'))):
                    # Remove numbering and clean up
                    variation = line[2:].strip()
                    if variation:
                        variations.append(variation)
            
            # Ensure we have exactly 5 variations
            if len(variations) < 5:
                print(f"⚠️ Only generated {len(variations)} variations, padding with original query")
                while len(variations) < 5:
                    variations.append(query)
            elif len(variations) > 5:
                variations = variations[:5]
            
            print(f"✨ Generated {len(variations)} query variations:")
            for i, var in enumerate(variations, 1):
                print(f"  {i}. {var}")
                
            return variations
            
        except Exception as e:
            print(f"❌ Error generating query variations: {str(e)}")
            # Fallback to original query repeated 5 times
            return [query] * 5

    def _multi_query_retrieval(self, query: str, top_k_per_query: int = 10) -> List[Document]:
        """
        Perform multi-query retrieval by generating variations and combining results
        
        Args:
            query: Original user query
            top_k_per_query: Number of documents to retrieve per query variation
            
        Returns:
            List of unique documents from all query variations
        """
        print(f"🔍 Starting multi-query retrieval with {top_k_per_query} docs per query...")
        
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        # Add the original query to the variations
        all_queries = [query] + query_variations
        
        # Store all retrieved documents with their content as key to avoid duplicates
        all_docs = {}
        query_results = {}
        
        # Retrieve documents for each query variation
        for i, query_var in enumerate(all_queries):
            print(f"📚 Retrieving for query {i+1}/{len(all_queries)}: {query_var[:50]}...")
            
            try:
                # Use hybrid search for each variation
                docs = self._hybrid_search(query_var, top_k_per_query)
                query_results[i] = docs
                
                # Add documents to combined set (use content as key to avoid duplicates)
                for doc in docs:
                    doc_key = doc.page_content[:100]  # Use first 100 chars as key
                    if doc_key not in all_docs:
                        all_docs[doc_key] = doc
                
                print(f"  ✅ Retrieved {len(docs)} documents")
                
            except Exception as e:
                print(f"  ❌ Error retrieving for variation {i+1}: {str(e)}")
                query_results[i] = []
        
        # Convert back to list
        combined_docs = list(all_docs.values())
        
        print(f"🎯 Multi-query retrieval completed:")
        print(f"  - Total queries: {len(all_queries)}")
        print(f"  - Total unique documents: {len(combined_docs)}")
        print(f"  - Average docs per query: {len(combined_docs) / len(all_queries):.1f}")
        
        return combined_docs

    def compare_retrieval_methods(self, question: str) -> Dict[str, Any]:
        """
        Compare single-query vs multi-query retrieval for a given question
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n🔍 COMPARING RETRIEVAL METHODS FOR: {question}")
        print("="*80)
        
        try:
            # Single-query retrieval
            print("\n📚 SINGLE-QUERY RETRIEVAL:")
            single_result = self.answer_question(question, use_multi_query=False)
            
            print("\n🚀 MULTI-QUERY RETRIEVAL:")
            multi_result = self.answer_question(question, use_multi_query=True)
            
            comparison = {
                "question": question,
                "single_query": {
                    "answer": single_result["answer"],
                    "num_docs_retrieved": single_result.get("num_documents_retrieved", "N/A"),
                    "num_docs_used": single_result.get("num_documents_used", "N/A"),
                    "sources": single_result.get("sources", [])
                },
                "multi_query": {
                    "answer": multi_result["answer"],
                    "num_docs_retrieved": multi_result.get("num_documents_retrieved", "N/A"),
                    "num_docs_used": multi_result.get("num_documents_used", "N/A"),
                    "sources": multi_result.get("sources", [])
                }
            }
            
            # Print comparison summary
            print(f"\n📊 COMPARISON SUMMARY:")
            print(f"Single-query - Docs retrieved: {comparison['single_query']['num_docs_retrieved']}, Used: {comparison['single_query']['num_docs_used']}")
            print(f"Multi-query  - Docs retrieved: {comparison['multi_query']['num_docs_retrieved']}, Used: {comparison['multi_query']['num_docs_used']}")
            print(f"Single-query sources: {len(comparison['single_query']['sources'])}")
            print(f"Multi-query sources: {len(comparison['multi_query']['sources'])}")
            
            return comparison
            
        except Exception as e:
            print(f"❌ Error in comparison: {str(e)}")
            return {"error": str(e), "question": question}
     
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
        6. **Do NOT Hallucinate:** Ensure all information is grounded in the provided context and legal knowledge and if you cannot answer the question according to the provided context state:"I cannot answer that question based on the provided information."
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
                "score_threshold": 0.3,  # Minimum relevance score
            },
            search_type="mmr"  # Use MMR for diverse results
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

    def answer_question(self, question: str, use_multi_query: bool = True) -> Dict[str, Any]:
        """Enhanced answer generation with multi-query retrieval"""
        try:
            print(f"\n❓ Received question: {question}")
            
            # 1. Rewrite the query for better legal search
            rewritten_query, original_query = self._rewrite_query(question)
            
            # 2. Choose retrieval strategy
            if use_multi_query:
                print("🚀 Using multi-query retrieval strategy")
                # Use multi-query retrieval
                initial_docs = self._multi_query_retrieval(question, top_k_per_query=20)
            else:
                print("📚 Using single-query retrieval strategy")
                # Use traditional single query retrieval
                initial_docs = self._hybrid_search(rewritten_query, self.top_k)
            
            print(f"📚 Retrieved {len(initial_docs)} unique documents")
            
            # 3. Re-rank documents using the rewritten query
            reranked_docs = self._rerank_documents(original_query, initial_docs)
            print(f"🎯 Selected top {len(reranked_docs)} documents after re-ranking")
            
            # 4. Create context from reranked documents
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            # 5. Generate answer using the original question
            prompt_template = self._create_prompt_template()
            formatted_prompt = prompt_template.format(
                question=question,  # Use original question for answer generation
                context=context
            )
            
            # Generate the answer
            answer = self.llm.invoke(formatted_prompt)
            
            # Process sources
            sources = []
            for doc in reranked_docs:
                sources.append({
                    "act": doc.metadata.get("act", "Unknown Act"),
                    "chapter": doc.metadata.get("chapter", "Unknown Chapter"),
                    "section": doc.metadata.get("section_number", "Unknown Section"),
                    "title": doc.metadata.get("section_title", "Unknown Title"),
                    "relevance": "High"
                })
            
            return {
                "rewritten_query": rewritten_query,
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_documents_retrieved": len(initial_docs),
                "num_documents_used": len(reranked_docs),
                "retrieval_method": "multi-query" if use_multi_query else "single-query"
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return {
                "rewritten_query": rewritten_query if 'rewritten_query' in locals() else question,
                "question": question,
                "answer": "Sorry, I encountered an error while processing your question.",
                "error": str(e),
                "retrieval_method": "multi-query" if use_multi_query else "single-query"
            }

def demo_multi_query():
    """Demonstration of multi-query retrieval functionality"""
    print("🎯 MULTI-QUERY RETRIEVAL DEMONSTRATION")
    print("="*60)
    
    # Initialize the RAG chain
    rag_chain = ZimLawRAGChain()
    
    # Example question
    demo_question = "What are my rights if I'm fired from work?"
    
    print(f"\n❓ Demo Question: {demo_question}")
    
    # Show query variations generation
    print("\n🔄 Generating query variations...")
    variations = rag_chain._generate_query_variations(demo_question)
    
    print(f"\n✨ Generated {len(variations)} variations:")
    for i, variation in enumerate(variations, 1):
        print(f"  {i}. {variation}")
    
    # Perform multi-query retrieval
    print(f"\n🚀 Performing multi-query retrieval...")
    result = rag_chain.answer_question(demo_question, use_multi_query=True)
    
    print(f"\n📊 RESULTS:")
    print(f"Documents Retrieved: {result.get('num_documents_retrieved', 'N/A')}")
    print(f"Documents Used for Answer: {result.get('num_documents_used', 'N/A')}")
    print(f"Retrieval Method: {result.get('retrieval_method', 'N/A')}")
    
    print(f"\n💡 Answer:")
    print(result['answer'])
    
    if "sources" in result and result['sources']:
        print(f"\n📚 Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['act']}, {source['chapter']}, Section {source['section']}")
    
    return result

def main():
    """Test the RAG chain with sample legal questions using both retrieval methods"""
    rag_chain = ZimLawRAGChain()
    
    test_questions = [
        "Can my boss fire me for being late?",
        "What are my constitutional rights if I'm arrested or detained?",
        "What is the process for registering a trade union?",
        "What are the requirements for fair dismissal under the Labour Act?",
        "How can I legally change my name in Zimbabwe?"
    ]
    
    for question in test_questions:
        print("\n" + "="*100)
        print(f"🔍 TESTING QUESTION: {question}")
        print("="*100)
        
        # Test multi-query retrieval
        print("\n🚀 MULTI-QUERY RETRIEVAL:")
        print("-" * 50)
        result_multi = rag_chain.answer_question(question, use_multi_query=True)
        
        print(f"\n📊 RESULTS:")
        print(f"Question: {result_multi['question']}")
        print(f"Rewritten Query: {result_multi['rewritten_query']}")
        print(f"Documents Retrieved: {result_multi.get('num_documents_retrieved', 'N/A')}")
        print(f"Documents Used: {result_multi.get('num_documents_used', 'N/A')}")
        print(f"Answer: {result_multi['answer']}")
        
        if "sources" in result_multi:
            print(f"\n📚 Sources ({len(result_multi['sources'])}):")
            for i, source in enumerate(result_multi['sources'], 1):
                print(f"  {i}. {source['act']}, {source['chapter']}, Section {source['section']}")
        
        print("\n" + "="*100)
        
        # Add a small delay between questions to make output readable
        import time
        time.sleep(1)

if __name__ == "__main__":
    # Uncomment the line below to run the demo
    # demo_multi_query()
    main()
    rag_chain = ZimLawRAGChain()
    
    test_questions = [
        "Can my boss fire me for being late?",
        "What are my constitutional rights if I'm arrested or detained?",
        "What is the process for registering a trade union?",
        "What are the requirements for fair dismissal under the Labour Act?",
        "How can I legally change my name in Zimbabwe?"
    ]
    
    for question in test_questions:
        print("\n" + "="*100)
        print(f"🔍 TESTING QUESTION: {question}")
        print("="*100)
        
        # Test multi-query retrieval
        print("\n🚀 MULTI-QUERY RETRIEVAL:")
        print("-" * 50)
        result_multi = rag_chain.answer_question(question, use_multi_query=True)
        
        print(f"\n📊 RESULTS:")
        print(f"Question: {result_multi['question']}")
        print(f"Rewritten Query: {result_multi['rewritten_query']}")
        print(f"Documents Retrieved: {result_multi.get('num_documents_retrieved', 'N/A')}")
        print(f"Documents Used: {result_multi.get('num_documents_used', 'N/A')}")
        print(f"Answer: {result_multi['answer']}")
        
        if "sources" in result_multi:
            print(f"\n📚 Sources ({len(result_multi['sources'])}):")
            for i, source in enumerate(result_multi['sources'], 1):
                print(f"  {i}. {source['act']}, {source['chapter']}, Section {source['section']}")
        
        print("\n" + "="*100)
        
        # Add a small delay between questions to make output readable
        import time
        time.sleep(1)

if __name__ == "__main__":
    main()