import os
import json
import glob
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class LegalSection:
    """Represents a section from a legal document"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    
@dataclass
class SemanticChunkingConfig:
    """Configuration for semantic chunking strategy"""
    max_chunk_size: int = 1000  # Maximum size for a single chunk
    min_chunk_size: int = 200   # Minimum size for a chunk to be meaningful
    overlap_size: int = 100     # Overlap between chunks for context preservation
    similarity_threshold: float = 0.7  # Similarity threshold for semantic grouping
    sentence_model: str = "all-MiniLM-L6-v2"  # Lightweight model for chunking
    preserve_structure: bool = True  # Whether to preserve legal structure markers


class SemanticChunker:
    """Advanced semantic chunking specifically designed for legal documents"""
    
    def __init__(self, config: SemanticChunkingConfig):
        self.config = config
        self.sentence_model = SentenceTransformer(config.sentence_model)
        self.legal_markers = [
            r'\([0-9]+\)',  # (1), (2), etc.
            r'\([a-z]+\)',  # (a), (b), etc.
            r'\([ivx]+\)',  # (i), (ii), (iii), etc.
            r'Section \d+',  # Section markers
            r'Chapter \d+',  # Chapter markers
            r'Part [IVX]+',  # Part markers
            r'Article \d+',  # Article markers
        ]
    
    def identify_legal_boundaries(self, text: str) -> List[int]:
        """Identify natural legal document boundaries like subsections, paragraphs"""
        boundaries = [0]  # Start of document
        
        # Find legal structure markers
        for pattern in self.legal_markers:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                if pos not in boundaries:
                    boundaries.append(pos)
        
        # Find paragraph boundaries (double newlines)
        for match in re.finditer(r'\n\s*\n', text):
            pos = match.start()
            if pos not in boundaries:
                boundaries.append(pos)
        
        # Find sentence boundaries for fine-grained splitting
        for match in re.finditer(r'[.!?]\s+[A-Z]', text):
            pos = match.start() + 1
            if pos not in boundaries:
                boundaries.append(pos)
        
        boundaries.append(len(text))  # End of document
        return sorted(set(boundaries))
    
    def create_semantic_segments(self, text: str) -> List[str]:
        """Create initial segments based on legal structure and semantics"""
        boundaries = self.identify_legal_boundaries(text)
        segments = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end].strip()
            
            if segment and len(segment) >= 50:  # Minimum meaningful segment
                segments.append(segment)
        
        return segments
    
    def compute_semantic_similarity(self, segments: List[str]) -> np.ndarray:
        """Compute semantic similarity matrix between segments"""
        if not segments:
            return np.array([])
        
        # Get embeddings for all segments
        embeddings = self.sentence_model.encode(segments)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def merge_similar_segments(self, segments: List[str], similarity_matrix: np.ndarray) -> List[str]:
        """Merge semantically similar adjacent segments"""
        if len(segments) <= 1:
            return segments
        
        merged_segments = []
        current_chunk = segments[0]
        
        for i in range(1, len(segments)):
            # Check similarity with previous segment
            similarity = similarity_matrix[i-1, i]
            
            # Check if merging would exceed max chunk size
            potential_merge = current_chunk + " " + segments[i]
            
            if (similarity > self.config.similarity_threshold and 
                len(potential_merge) <= self.config.max_chunk_size):
                # Merge with current chunk
                current_chunk = potential_merge
            else:
                # Start new chunk
                if len(current_chunk.strip()) >= self.config.min_chunk_size:
                    merged_segments.append(current_chunk.strip())
                current_chunk = segments[i]
        
        # Add final chunk
        if len(current_chunk.strip()) >= self.config.min_chunk_size:
            merged_segments.append(current_chunk.strip())
        
        return merged_segments
    
    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """Create overlapping chunks to preserve context"""
        if not chunks or len(chunks) <= 1:
            return chunks
        
        overlapping_chunks = []
        
        for i, chunk in enumerate(chunks):
            overlapping_chunks.append(chunk)
            
            # Create overlap with next chunk if it exists
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Get last N characters from current chunk
                overlap_start = chunk[-self.config.overlap_size:] if len(chunk) > self.config.overlap_size else chunk
                # Get first N characters from next chunk
                overlap_end = next_chunk[:self.config.overlap_size] if len(next_chunk) > self.config.overlap_size else next_chunk
                
                overlap_chunk = overlap_start + " " + overlap_end
                
                if (len(overlap_chunk) >= self.config.min_chunk_size and 
                    overlap_chunk not in overlapping_chunks):
                    overlapping_chunks.append(overlap_chunk)
        
        return overlapping_chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Main method to chunk text using semantic similarity"""
        print(f"📝 Semantic chunking text of length {len(text)}")
        
        # Step 1: Create initial segments based on legal structure
        segments = self.create_semantic_segments(text)
        print(f"  📋 Created {len(segments)} initial segments")
        
        if not segments:
            return []
        
        # Step 2: Compute semantic similarity
        similarity_matrix = self.compute_semantic_similarity(segments)
        
        # Step 3: Merge similar segments
        merged_chunks = self.merge_similar_segments(segments, similarity_matrix)
        print(f"  🔗 Merged into {len(merged_chunks)} semantic chunks")
        
        # Step 4: Handle oversized chunks with recursive splitting
        final_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.overlap_size,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for chunk in merged_chunks:
            if len(chunk) > self.config.max_chunk_size:
                # Use recursive splitter for oversized chunks
                sub_chunks = text_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # Step 5: Create overlapping chunks for better context
        if self.config.overlap_size > 0:
            final_chunks = self.create_overlapping_chunks(final_chunks)
        
        print(f"  ✅ Final semantic chunks: {len(final_chunks)}")
        return final_chunks


class LegalDocumentIngester:
    def __init__(
        self,
        json_dir: str = "./data/clean",
        vectorstore_dir: str = "./vectorstore/faiss_index",
        chunking_config: Optional[SemanticChunkingConfig] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ):
        self.json_dir = json_dir
        self.vectorstore_dir = vectorstore_dir
        self.chunking_config = chunking_config or SemanticChunkingConfig()
        self.embedding_model = embedding_model
        self.semantic_chunker = SemanticChunker(self.chunking_config)
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

    def create_enhanced_content(self, section: Dict, doc_metadata: Dict) -> str:
        """Create enhanced content with legal context for better semantic understanding"""
        content = section.get('text') or section.get('content', '')
        title = section.get('title', '').strip()
        section_number = str(section.get('section', '')).strip()
        
        # Create rich context for better embeddings
        enhanced_content = f"""
        Legal Document: {doc_metadata['title']}
        Chapter: {doc_metadata['chapter']}
        Section {section_number}: {title}
        
        Content:
        {content}
        
        Context: This section is part of {doc_metadata['title']}, specifically addressing {title.lower()} matters under {doc_metadata['chapter']}.
        """.strip()
        
        return enhanced_content

    def process_section_with_semantic_chunking(self, section: Dict, doc_metadata: Dict) -> List[LegalSection]:
        """Process a single section with semantic chunking"""
        try:
            content = section.get('text') or section.get('content', '')
            if not content.strip():
                return []

            # Create enhanced content with legal context
            enhanced_content = self.create_enhanced_content(section, doc_metadata)
            
            # Apply semantic chunking
            chunks = self.semantic_chunker.chunk_text(enhanced_content)
            
            if not chunks:
                return []

            # Create section metadata
            section_number = str(section.get('section', '')).strip()
            if not section_number:
                section_number = str(len(doc_metadata.get('processed_sections', [])) + 1)

            title = section.get('title', '').strip()
            if not title and content:
                title = content.split('\n')[0][:100]

            base_metadata = {
                "act": doc_metadata['title'],
                "chapter": doc_metadata['chapter'],
                "section_number": section_number,
                "section_title": title,
                "source": doc_metadata['source'],
                "commencement": doc_metadata.get('commencement', 'Unknown'),
                "version_date": doc_metadata.get('version_date', 'Unknown')
            }

            # Create LegalSection objects for each chunk
            legal_sections = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_metadata['title']}_{section_number}_chunk_{i+1}".replace(' ', '_').lower()
                
                chunk_metadata = {
                    **base_metadata,
                    "chunk_id": i + 1,
                    "total_chunks": len(chunks),
                    "is_chunked": len(chunks) > 1,
                    "chunk_type": "semantic"
                }
                
                chunk_title = f"{title} (Chunk {i+1})" if len(chunks) > 1 else title
                
                legal_sections.append(LegalSection(
                    id=chunk_id,
                    title=chunk_title,
                    content=chunk.strip(),
                    metadata=chunk_metadata
                ))
            
            return legal_sections
            
        except Exception as e:
            print(f"❌ Error processing section with semantic chunking: {e}")
            return []

    def prepare_sections(self, documents: List[Dict]) -> List[LegalSection]:
        """Extract and prepare all sections with semantic chunking"""
        sections = []
        
        print("📑 Processing document sections with semantic chunking...")
        for doc in tqdm(documents, desc="Processing Documents"):
            doc_metadata = doc['metadata']
            doc_metadata['processed_sections'] = []
            
            for section in doc['sections']:
                legal_sections = self.process_section_with_semantic_chunking(section, doc_metadata)
                sections.extend(legal_sections)
                
                # Track processed sections
                for legal_section in legal_sections:
                    doc_metadata['processed_sections'].append(legal_section.id)
        
        return sections

    def create_embeddings(self):
        """Initialize the embeddings model with enhanced configuration"""
        print(f"🧠 Loading embedding model: {self.embedding_model}...")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16,
                'show_progress_bar': True
            }
        )

    def create_search_document(self, section: LegalSection) -> Dict:
        """Create a formatted document for search with enhanced metadata"""
        # Create context-rich text for better semantic search
        context_info = ""
        if section.metadata.get("is_chunked"):
            context_info = f" [Semantic Chunk {section.metadata['chunk_id']} of {section.metadata['total_chunks']}]"
        
        # Enhanced text format for better retrieval
        formatted_text = f"""
        Legal Authority: {section.metadata['act']}
        Legal Division: {section.metadata['chapter']}
        Section {section.metadata['section_number']}: {section.metadata['section_title']}{context_info}
        
        Legal Content:
        {section.content}
        
        Legal Reference: {section.metadata['act']}, {section.metadata['chapter']}, Section {section.metadata['section_number']}
        Document Type: Zimbabwean Legal Document
        Chunk Type: {section.metadata.get('chunk_type', 'standard')}
        """.strip()

        return {
            "text": formatted_text,
            "metadata": {
                "id": section.id,
                **section.metadata,
                "content_length": len(section.content),
                "has_legal_markers": any(marker in section.content for marker in ['(a)', '(b)', '(1)', '(2)']),
                "content_type": self._classify_content_type(section.content)
            }
        }
    
    def _classify_content_type(self, content: str) -> str:
        """Classify the type of legal content for better retrieval"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['right', 'rights', 'entitled', 'freedom']):
            return "rights_and_freedoms"
        elif any(word in content_lower for word in ['procedure', 'process', 'steps', 'application']):
            return "procedures"
        elif any(word in content_lower for word in ['penalty', 'fine', 'imprisonment', 'conviction']):
            return "penalties"
        elif any(word in content_lower for word in ['definition', 'means', 'interpret', 'defined']):
            return "definitions"
        elif any(word in content_lower for word in ['duty', 'obligation', 'shall', 'must']):
            return "obligations"
        else:
            return "general_provision"

    def create_vectorstore(self, sections: List[LegalSection], embeddings):
        """Create and save the enhanced FAISS vectorstore with metadata"""
        print("🔍 Creating enhanced FAISS vectorstore...")
        
        # Prepare documents for vectorstore
        documents = [
            self.create_search_document(section)
            for section in tqdm(sections, desc="Preparing Enhanced Search Documents")
        ]
        
        # Create vectorstore with enhanced texts and metadata
        vectorstore = FAISS.from_texts(
            texts=[doc["text"] for doc in documents],
            embedding=embeddings,
            metadatas=[doc["metadata"] for doc in documents]
        )
        
        # Save vectorstore
        vectorstore.save_local(self.vectorstore_dir)
        print(f"✅ Enhanced vectorstore saved to {self.vectorstore_dir}")
        
        # Print statistics
        self._print_ingestion_statistics(sections, documents)
        
        return vectorstore
    
    def _print_ingestion_statistics(self, sections: List[LegalSection], documents: List[Dict]):
        """Print detailed statistics about the ingestion process"""
        print("\n📊 INGESTION STATISTICS:")
        print(f"  Total sections processed: {len(sections)}")
        print(f"  Total documents created: {len(documents)}")
        
        # Chunk statistics
        chunked_sections = [s for s in sections if s.metadata.get('is_chunked', False)]
        print(f"  Sections with semantic chunking: {len(chunked_sections)}")
        
        # Content type distribution
        content_types = {}
        for doc in documents:
            content_type = doc['metadata'].get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        print(f"  Content type distribution:")
        for content_type, count in sorted(content_types.items()):
            print(f"    {content_type}: {count}")
        
        # Size statistics
        content_lengths = [doc['metadata']['content_length'] for doc in documents]
        avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        print(f"  Average content length: {avg_length:.0f} characters")
        print(f"  Min content length: {min(content_lengths) if content_lengths else 0}")
        print(f"  Max content length: {max(content_lengths) if content_lengths else 0}")

    def run_pipeline(self):
        """Run the complete enhanced ingestion pipeline"""
        print("=== Starting Enhanced Legal Document Ingestion Pipeline ===")
        
        # 1. Load JSON documents
        documents = self.load_json_documents()
        if not documents:
            print("❌ No documents found to process")
            return
        
        # 2. Process sections with semantic chunking
        sections = self.prepare_sections(documents)
        print(f"✅ Processed {len(sections)} sections with semantic chunking")
        
        # 3. Initialize embeddings
        embeddings = self.create_embeddings()
        
        # 4. Create and save enhanced vectorstore
        vectorstore = self.create_vectorstore(sections, embeddings)
        
        print("=== Enhanced Ingestion Pipeline Complete ===")
        return vectorstore


def test_semantic_chunking():
    """Test the semantic chunking functionality"""
    print("🧪 Testing Semantic Chunking...")
    
    config = SemanticChunkingConfig(
        max_chunk_size=800,
        min_chunk_size=200,
        similarity_threshold=0.7
    )
    
    chunker = SemanticChunker(config)
    
    # Sample legal text
    sample_text = """
    Section 50: Rights of arrested and detained persons
    
    (1) Any person who is arrested must be informed at the time of arrest of the reason for the arrest.
    
    (2) Any person who is arrested must be permitted, without delay, at the expense of the State, to contact their spouse or partner, or a relative or legal practitioner, or anyone else of their choice.
    
    (3) Any person who is arrested must be treated humanely and with respect for their inherent dignity.
    
    (4) Any person who is arrested must be released unconditionally or on reasonable conditions, pending a charge or trial, unless there are compelling reasons justifying their continued detention.
    """
    
    chunks = chunker.chunk_text(sample_text)
    
    print(f"Original text length: {len(sample_text)}")
    print(f"Number of chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (length: {len(chunk)}):")
        print(f"'{chunk[:100]}...'")


if __name__ == "__main__":
    # Uncomment to test semantic chunking
    # test_semantic_chunking()
    
    # Run the enhanced ingestion pipeline
    config = SemanticChunkingConfig(
        max_chunk_size=1000,
        min_chunk_size=200,
        overlap_size=100,
        similarity_threshold=0.7,
        preserve_structure=True
    )
    
    ingester = LegalDocumentIngester(
        chunking_config=config,
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    ingester.run_pipeline()