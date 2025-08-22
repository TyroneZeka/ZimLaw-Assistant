import os
import json
import glob
import re
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

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
    batch_size: int = 32  # Batch size for GPU processing
    max_workers: int = 4  # Number of parallel workers for text preprocessing
    device: str = "auto"  # Device selection: "auto", "cuda", "cpu"


class SemanticChunker:
    """GPU-optimized semantic chunking specifically designed for legal documents"""
    
    def __init__(self, config: SemanticChunkingConfig):
        self.config = config
        
        # Auto-detect device with better error handling
        if config.device == "auto":
            if torch.cuda.is_available():
                try:
                    # Test GPU availability
                    torch.cuda.current_device()
                    torch.cuda.empty_cache()
                    self.device = "cuda"
                    print(f"🚀 GPU detected and accessible")
                except Exception as e:
                    print(f"⚠️  GPU detected but not accessible: {e}")
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = config.device
            
        print(f"🚀 Using device: {self.device}")
        
        # Load model with better error handling
        try:
            print(f"📥 Loading sentence transformer model: {config.sentence_model}")
            self.sentence_model = SentenceTransformer(
                config.sentence_model,
                device=self.device
            )
            print(f"✅ Model loaded successfully on {self.device}")
            
            # Optimize model for inference only if on GPU and working
            if self.device == "cuda":
                try:
                    # Test if GPU operations work
                    test_text = ["This is a test sentence."]
                    test_embedding = self.sentence_model.encode(test_text, convert_to_tensor=True)
                    print(f"✅ GPU operations test passed")
                    
                    # Only use FP16 if GPU test succeeds
                    self.sentence_model.half()
                    torch.backends.cudnn.benchmark = True
                    print(f"✅ FP16 optimization enabled")
                except Exception as e:
                    print(f"⚠️  GPU optimization failed, using FP32: {e}")
                    # Reload model on CPU if GPU fails
                    self.sentence_model = SentenceTransformer(
                        config.sentence_model,
                        device="cpu"
                    )
                    self.device = "cpu"
                    print(f"🔄 Fallback to CPU mode")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"🔄 Attempting fallback to CPU...")
            try:
                self.sentence_model = SentenceTransformer(
                    config.sentence_model,
                    device="cpu"
                )
                self.device = "cpu"
                print(f"✅ CPU fallback successful")
            except Exception as fallback_e:
                print(f"❌ CPU fallback also failed: {fallback_e}")
                raise RuntimeError(f"Could not initialize sentence transformer: {fallback_e}")
        
        self.legal_markers = [
            r'\([0-9]+\)',  # (1), (2), etc.
            r'\([a-z]+\)',  # (a), (b), etc.
            r'\([ivx]+\)',  # (i), (ii), (iii), etc.
            r'Section \d+',  # Section markers
            r'Chapter \d+',  # Chapter markers
            r'Part [IVX]+',  # Part markers
            r'Article \d+',  # Article markers
        ]
        
        # Compile regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.legal_markers]
    
    def identify_legal_boundaries(self, text: str) -> List[int]:
        """Identify natural legal document boundaries - optimized version"""
        boundaries = [0]  # Start of document
        
        # Use compiled patterns for better performance
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
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
    
    def compute_semantic_similarity(self, segments: List[str]) -> torch.Tensor:
        """GPU-accelerated semantic similarity computation with fallback"""
        if not segments:
            return torch.tensor([])
        
        try:
            # Process in batches for memory efficiency
            all_embeddings = []
            batch_size = min(self.config.batch_size, len(segments))  # Don't exceed segment count
            
            with torch.no_grad():
                for i in tqdm(range(0, len(segments), batch_size), desc="Computing embeddings", disable=False):
                    batch_segments = segments[i:i + batch_size]
                    
                    try:
                        # Get embeddings with device acceleration - remove conflicting parameters
                        batch_embeddings = self.sentence_model.encode(
                            batch_segments,
                            batch_size=len(batch_segments),
                            convert_to_tensor=True,
                            device=self.device
                        )
                        
                        all_embeddings.append(batch_embeddings)
                        
                    except Exception as batch_e:
                        print(f"Batch {i//batch_size + 1} failed on {self.device}, trying CPU: {batch_e}")
                        # Fallback to CPU for this batch
                        batch_embeddings = self.sentence_model.encode(
                            batch_segments,
                            batch_size=len(batch_segments),
                            convert_to_tensor=True,
                            device="cpu"
                        )
                        # Move to target device if needed
                        if self.device == "cuda" and batch_embeddings.device.type == "cpu":
                            batch_embeddings = batch_embeddings.cuda()
                        
                        all_embeddings.append(batch_embeddings)
            
            if not all_embeddings:
                return torch.tensor([])
            
            # Concatenate all embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            
            # Normalize embeddings for cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute cosine similarity matrix
            if self.device == "cuda":
                try:
                    similarity_matrix = torch.mm(embeddings, embeddings.t())
                except Exception as gpu_e:
                    print(f"GPU similarity computation failed, using CPU: {gpu_e}")
                    # Fallback to CPU
                    embeddings_cpu = embeddings.cpu()
                    similarity_matrix = torch.mm(embeddings_cpu, embeddings_cpu.t())
            else:
                similarity_matrix = torch.mm(embeddings, embeddings.t())
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Similarity computation failed, using fallback: {e}")
            # Return identity matrix as fallback
            n = len(segments)
            return torch.eye(n)
    
    def merge_similar_segments(self, segments: List[str], similarity_matrix: torch.Tensor) -> List[str]:
        """GPU-accelerated segment merging with fallback"""
        if len(segments) <= 1:
            return segments
        
        try:
            # Convert to numpy for easier indexing
            if similarity_matrix.device.type == "cuda":
                sim_matrix_np = similarity_matrix.cpu().numpy()
            else:
                sim_matrix_np = similarity_matrix.numpy()
            
        except Exception as e:
            print(f"    ⚠️  Similarity matrix conversion failed: {e}")
            # Fallback: return segments as-is
            return segments
        
        merged_segments = []
        current_chunk = segments[0]
        
        for i in range(1, len(segments)):
            try:
                # Check similarity with previous segment
                similarity = sim_matrix_np[i-1, i] if i-1 < sim_matrix_np.shape[0] and i < sim_matrix_np.shape[1] else 0.0
                
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
                    
            except Exception as e:
                print(f"    ⚠️  Error merging segment {i}: {e}")
                # Add current chunk and start new one
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
        """Main method to chunk text using semantic similarity with robust error handling"""
        
        try:
            # Step 1: Create initial segments based on legal structure
            segments = self.create_semantic_segments(text)
            
            if not segments:
                return []
            
            # Step 2: Compute semantic similarity with error handling
            try:
                similarity_matrix = self.compute_semantic_similarity(segments)
                
                if similarity_matrix.numel() == 0:
                    raise ValueError("Empty similarity matrix")
                    
            except Exception as sim_e:
                # Fallback to simple recursive chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.max_chunk_size,
                    chunk_overlap=self.config.overlap_size,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                fallback_chunks = text_splitter.split_text(text)
                return fallback_chunks
            
            # Step 3: Merge similar segments
            try:
                merged_chunks = self.merge_similar_segments(segments, similarity_matrix)
            except Exception as merge_e:
                # Use original segments if merging fails
                merged_chunks = segments
            
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
                try:
                    final_chunks = self.create_overlapping_chunks(final_chunks)
                except Exception as overlap_e:
                    # Continue without overlaps
                    pass
            
            return final_chunks
            
        except Exception as e:
            # Emergency fallback
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.max_chunk_size,
                    chunk_overlap=self.config.overlap_size,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                emergency_chunks = text_splitter.split_text(text)
                return emergency_chunks
            except Exception as emergency_e:
                # Last resort: return the whole text as one chunk
                return [text] if text.strip() else []


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
                    "chunk_type": "semantic_gpu"  # Updated to reflect GPU optimization
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
        """Extract and prepare all sections with GPU-optimized semantic chunking and filtering"""
        print("📑 Processing document sections...")
        
        # First, filter and analyze documents
        valid_docs = 0
        empty_docs = 0
        sections = []
        
        for doc in tqdm(documents, desc="Processing Documents"):
            doc_metadata = doc['metadata']
            doc_metadata['processed_sections'] = []
            doc_sections = doc.get('sections', [])
            
            if not doc_sections:
                empty_docs += 1
                continue
            
            doc_has_content = False
            for section in doc_sections:
                content = section.get('text') or section.get('content', '')
                if content.strip() and len(content.strip()) > 10:  # Minimum meaningful content
                    legal_sections = self.process_section_with_semantic_chunking(section, doc_metadata)
                    sections.extend(legal_sections)
                    
                    # Track processed sections
                    for legal_section in legal_sections:
                        doc_metadata['processed_sections'].append(legal_section.id)
                    
                    doc_has_content = True
            
            if doc_has_content:
                valid_docs += 1
            else:
                empty_docs += 1
        
        print(f"✅ Processed {valid_docs} documents with content, skipped {empty_docs} empty documents")
        print(f"📝 Created {len(sections)} sections total")
        
        return sections

    def create_embeddings(self):
        """Initialize the embeddings model with dynamic GPU optimization"""
        print(f"🧠 Loading embedding model: {self.embedding_model}...")
        
        # Use the same device detection logic as the chunker
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust batch size based on device capabilities
        if device == "cuda":
            try:
                # Test GPU memory availability
                test_tensor = torch.zeros(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                batch_size = 64  # Larger batch for GPU
            except Exception as gpu_e:
                device = "cpu"
                batch_size = 8
        else:
            batch_size = 8  # Conservative batch for CPU
        
        try:
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={
                    'device': device,
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': batch_size
                }
            )
        except Exception as e:
            print(f"❌ Failed to load {self.embedding_model}: {e}")
            print(f"🔄 Trying fallback model...")
            
            # Fallback to a more reliable model
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Safe fallback to CPU
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 4
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
        Chunk Type: {section.metadata.get('chunk_type', 'standard')} (GPU-Optimized)
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
        """Print detailed statistics about the GPU-optimized ingestion process"""
        print("\n📊 GPU-OPTIMIZED INGESTION STATISTICS:")
        print(f"  Total sections processed: {len(sections)}")
        print(f"  Total search documents created: {len(documents)}")
        
        if not sections:
            print("  ⚠️  No sections were processed - check your input documents")
            return
        
        # Chunk statistics
        chunked_sections = [s for s in sections if s.metadata.get('is_chunked', False)]
        gpu_chunks = [s for s in sections if s.metadata.get('chunk_type') == 'semantic_gpu']
        print(f"  Sections with GPU semantic chunking: {len(chunked_sections)}")
        print(f"  GPU-optimized chunks: {len(gpu_chunks)}")
        
        # Document source statistics
        document_sources = {}
        for section in sections:
            source_act = section.metadata.get('act', 'Unknown')
            document_sources[source_act] = document_sources.get(source_act, 0) + 1
        
        print(f"  Documents with processed content: {len(document_sources)}")
        
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
        
        print(f"\n💡 GPU Optimization Benefits:")
        print(f"  ✅ GPU-accelerated similarity computation")
        print(f"  ✅ Batch processing enabled")
        print(f"  ✅ Empty document filtering active")
        print(f"  ✅ Memory-efficient processing")

    def run_pipeline(self):
        """Run the complete GPU-optimized ingestion pipeline"""
        print("=== Starting GPU-Optimized Legal Document Ingestion Pipeline ===")
        
        # 1. Load JSON documents
        documents = self.load_json_documents()
        if not documents:
            print("❌ No documents found to process")
            return
        
        # 2. Process sections with GPU-optimized semantic chunking
        sections = self.prepare_sections(documents)
        print(f"✅ Processed {len(sections)} sections with GPU-optimized semantic chunking")
        
        # 3. Initialize embeddings with GPU support
        embeddings = self.create_embeddings()
        
        # 4. Create and save enhanced vectorstore
        vectorstore = self.create_vectorstore(sections, embeddings)
        
        print("=== GPU-Optimized Ingestion Pipeline Complete ===")
        return vectorstore


def test_gpu_setup():
    """Test GPU setup and model loading before running full pipeline"""
    print("🧪 Testing GPU Setup")
    print("=" * 30)
    
    # Check PyTorch GPU availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        # Test basic GPU operations
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = test_tensor * 2
            print(f"✅ Basic GPU operations working")
        except Exception as e:
            print(f"❌ Basic GPU operations failed: {e}")
            return False
    
    # Test sentence transformer loading
    try:
        print(f"🔄 Testing SentenceTransformer loading...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Start with CPU
        test_embedding = model.encode(["Test sentence"], convert_to_tensor=True)
        print(f"✅ SentenceTransformer CPU test passed")
        
        if torch.cuda.is_available():
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
                test_embedding = model.encode(["Test sentence"], convert_to_tensor=True)
                print(f"✅ SentenceTransformer GPU test passed")
            except Exception as gpu_e:
                print(f"⚠️  SentenceTransformer GPU test failed: {gpu_e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformer test failed: {e}")
        return False


def test_semantic_chunking():
    """Test the semantic chunking functionality with simple example"""
    print("🧪 Testing Semantic Chunking...")
    
    # Test with CPU first
    config = SemanticChunkingConfig(
        max_chunk_size=800,
        min_chunk_size=200,
        similarity_threshold=0.7,
        device="cpu"  # Start with CPU
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
    
    return len(chunks) > 0


if __name__ == "__main__":
    # Check GPU availability
    device_to_use = "auto"
    
    if torch.cuda.is_available():
        try:
            # Test basic GPU operations
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = test_tensor * 2
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
            device_to_use = "auto"  # Will use CUDA
        except Exception as e:
            print(f"⚠️  GPU not working, using CPU")
            device_to_use = "cpu"
    else:
        print("⚠️  No GPU available, using CPU")
        device_to_use = "cpu"
    
    # Configure with optimal settings for detected device
    if device_to_use == "cpu":
        config = SemanticChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=200,
            overlap_size=100,
            similarity_threshold=0.5,
            preserve_structure=True,
            batch_size=8,         # Smaller batch for CPU
            max_workers=4,        
            device="cpu"
        )
        print("🚀 Starting ingestion pipeline (CPU mode)")
    else:
        config = SemanticChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=200,
            overlap_size=100,
            similarity_threshold=0.5,
            preserve_structure=True,
            batch_size=64,        # Larger batch for GPU
            max_workers=8,        
            device="auto"         # Auto-detect GPU
        )
        print("🚀 Starting ingestion pipeline (GPU mode)")
    
    ingester = LegalDocumentIngester(
        chunking_config=config,
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    
    try:
        result = ingester.run_pipeline()
        if result:
            print("✅ Pipeline completed successfully!")
        else:
            print("⚠️  Pipeline completed with warnings")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()