import os
import json
import glob
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ... existing imports ...

@dataclass
class LegalSection:
    """Represents a section from a legal document"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]

class LegalDocumentIngester:
    def __init__(
        self,
        json_dir: str = "./data/clean",
        vectorstore_dir: str = "./vectorstore/faiss_index",
        chunk_size: int = 800
    ):
        self.json_dir = json_dir
        self.vectorstore_dir = vectorstore_dir
        self.chunk_size = chunk_size
        os.makedirs(vectorstore_dir, exist_ok=True)

    def normalize_document(self, doc: Dict) -> Dict:
        """Normalize document structure regardless of input format"""
        metadata = {
            "title": doc.get("title") or doc.get("metadata", {}).get("title", "Unknown"),
            "chapter": doc.get("chapter") or doc.get("metadata", {}).get("chapter", "Unknown"),
            "source": doc.get("source_file") or doc.get("metadata", {}).get("source_url", ""),
            "commencement": doc.get("metadata", {}).get("commencement", "Unknown"),
            "version_date": doc.get("metadata", {}).get("version_date", "Unknown")
        }
        
        sections = doc.get("sections", [])
        return {"metadata": metadata, "sections": sections}

    def load_json_documents(self) -> List[Dict]:
        """Load and normalize all JSON documents from the data directory"""
        json_files = glob.glob(os.path.join(self.json_dir, "*.json"))
        documents = []
        
        print(f"📁 Loading {len(json_files)} JSON documents...")
        for file_path in tqdm(json_files, desc="Loading Documents"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    normalized_doc = self.normalize_document(doc)
                    documents.append(normalized_doc)
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        return documents

    def process_section(self, section: Dict, doc_metadata: Dict) -> Optional[LegalSection]:
        """Process a single section into a structured format"""
        try:
            # Handle different section content keys
            content = section.get('text') or section.get('content', '')
            if not content.strip():
                return None

            # Create clean section number
            section_number = str(section.get('section', '')).strip()
            if not section_number:
                section_number = str(len(doc_metadata.get('processed_sections', [])) + 1)

            # Create unique ID
            section_id = f"{doc_metadata['title']}_{section_number}".replace(' ', '_').lower()
            
            # Create section title
            title = section.get('title', '').strip()
            if not title and content:
                # Use first line as title if none provided
                title = content.split('\n')[0][:100]

            # Combine metadata
            metadata = {
                "act": doc_metadata['title'],
                "chapter": doc_metadata['chapter'],
                "section_number": section_number,
                "section_title": title,
                "source": doc_metadata['source']
            }
            
            return LegalSection(
                id=section_id,
                title=title,
                content=content.strip(),
                metadata=metadata
            )
        except Exception as e:
            print(f"❌ Error processing section: {e}")
            return None

    def prepare_sections(self, documents: List[Dict]) -> List[LegalSection]:
        """Extract and prepare all sections from documents"""
        sections = []
        
        print("📑 Processing document sections...")
        for doc in tqdm(documents, desc="Processing Documents"):
            doc_metadata = doc['metadata']
            doc_metadata['processed_sections'] = []
            
            for section in doc['sections']:
                legal_section = self.process_section(section, doc_metadata)
                if legal_section:
                    sections.append(legal_section)
                    doc_metadata['processed_sections'].append(legal_section.id)
        
        return sections

    # ... rest of the existing methods (create_embeddings, create_search_document, create_vectorstore) ...

    def create_embeddings(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the embeddings model"""
        print(f"🧠 Loading embedding model: {model_name}...")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_search_document(self, section: LegalSection) -> Dict:
        """Create a formatted document for search"""
        # Create a well-structured text that preserves context
        text = f"""
Act: {section.metadata['act']}
Chapter: {section.metadata['chapter']}
Section {section.metadata['section_number']}: {section.title}

{section.content}
""".strip()

        return {
            "text": text,
            "metadata": {
                "id": section.id,
                **section.metadata
            }
        }

    def create_vectorstore(self, sections: List[LegalSection], embeddings):
        """Create and save the FAISS vectorstore"""
        print("🔍 Creating FAISS vectorstore...")
        
        # Prepare documents for vectorstore
        documents = [
            self.create_search_document(section)
            for section in tqdm(sections, desc="Preparing Search Documents")
        ]
        
        # Create vectorstore
        vectorstore = FAISS.from_texts(
            texts=[doc["text"] for doc in documents],
            embedding=embeddings,
            metadatas=[doc["metadata"] for doc in documents]
        )
        
        # Save vectorstore
        vectorstore.save_local(self.vectorstore_dir)
        print(f"✅ Vectorstore saved to {self.vectorstore_dir}")
        return vectorstore

    def run_pipeline(self):
        """Run the complete ingestion pipeline"""
        print("=== Starting Legal Document Ingestion Pipeline ===")
        
        # 1. Load JSON documents
        documents = self.load_json_documents()
        if not documents:
            print("❌ No documents found to process")
            return
        
        # 2. Process sections
        sections = self.prepare_sections(documents)
        print(f"✅ Processed {len(sections)} sections")
        
        # 3. Initialize embeddings
        embeddings = self.create_embeddings()
        
        # 4. Create and save vectorstore
        vectorstore = self.create_vectorstore(sections, embeddings)
        
        print("=== Ingestion Pipeline Complete ===")
        return vectorstore

def main():
    ingester = LegalDocumentIngester()
    ingester.run_pipeline()

if __name__ == "__main__":
    main()