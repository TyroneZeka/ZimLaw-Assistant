import os
from typing import Dict, Any, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ZimLawRAGChain:
    def __init__(
        self,
        vectorstore_dir: str = "./vectorstore/faiss_index",
        model_name: str = "deepseek-r1:8b",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_k: int = 4
    ):
        self.vectorstore_dir = vectorstore_dir
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        # Initialize components
        self.vectorstore = self._load_vectorstore()
        self.llm = self._initialize_llm()
        self.retriever = self._create_retriever()
        self.chain = self._create_chain()

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

    def _initialize_llm(self) -> Ollama:
        """Initialize the Deepseek LLM via Ollama"""
        print(f"🤖 Initializing {self.model_name}...")
        return Ollama(
            model=self.model_name,
            temperature=self.temperature,
            num_ctx=4096,  # Larger context window for legal texts
            num_predict=self.max_tokens,
            callbacks=[StreamingStdOutCallbackHandler()],
            stop=["Human:", "Assistant:"]  # Prevent continuing the conversation
        )

    def _create_retriever(self):
        """Create the retriever with custom search parameters"""
        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.top_k,
                "fetch_k": self.top_k * 3,  # Fetch more docs then filter
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
        """Get an answer for a legal question with source citations"""
        try:
            print(f"\n❓ Question: {question}")
            result = self.chain.invoke({
                "query": question
            })
            
            # Extract sources for citation
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "act": doc.metadata.get("act"),
                        "chapter": doc.metadata.get("chapter"),
                        "section": doc.metadata.get("section_number"),
                        "title": doc.metadata.get("section_title")
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