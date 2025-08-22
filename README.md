# ZimLaw Assistant

An advanced AI-powered legal assistant for Zimbabwe, built with RAG (Retrieval-Augmented Generation) and conditioned prompting to help citizens understand their rights under the Constitution and key laws.

## Features

### Core Functionality
- **Legal Question Answering**: Ask questions about your legal rights and get detailed, structured responses
- **Source Citations**: All answers include references to specific laws, sections, and chapters
- **Real-time Streaming**: Watch responses generate in real-time for better user experience
- **Conditioned Generation**: Uses few-shot prompting for consistent, high-quality legal advice format

### User Interface
- **Modern Design**: Professional gradient-based UI with responsive layout
- **Chat Interface**: Conversational format with newest-first message ordering
- **Quick Actions**: Pre-configured buttons for common legal topics
- **Dark Sidebar**: Enhanced navigation with improved contrast and accessibility
- **Mobile Responsive**: Works seamlessly across different screen sizes

### Technical Features
- **Hybrid Search**: Combines semantic similarity with re-ranking for better retrieval
- **Query Rewriting**: Automatically improves search queries for better results
- **Smart Formatting**: Automatic formatting with bold headers and structured content
- **Error Handling**: Comprehensive error handling with user-friendly messages

## Tech Stack

- **LLM**: Llama 3 (via Ollama) with streaming support
- **Embeddings**: all-MiniLM-L6-v2 for semantic search
- **Vector Database**: FAISS for efficient similarity search
- **Re-ranking**: BGE-reranker-large for improved result quality
- **Framework**: LangChain for LLM orchestration
- **UI**: Streamlit with custom CSS styling
- **Fonts**: Inter font family for modern typography

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Git for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TyroneZeka/ZimLaw-Assistant.git
   cd ZimLaw-Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama model**
   ```bash
   ollama pull llama3
   ```

4. **Ingest legal documents** (if not already done)
   ```bash
   python src/utils/ingest.py
   ```

5. **Launch the application**
   ```bash
   streamlit run src/app.py
   ```

6. **Access the app** at `http://localhost:8501`

## Usage Examples

### Sample Questions
- "What are my rights if I'm arrested?"
- "Can a 10-year-old be charged with a crime?"
- "What happens if someone is tried twice for the same crime?"
- "What constitutes criminal negligence?"

### Response Format
Each response includes:
- **Direct Answer**: Clear, concise answer to your question
- **Legal Basis**: Relevant laws and sections
- **Key Provisions/Rights**: Structured breakdown of important points
- **Additional Notes**: Context and practical implications

## Project Structure

```
ZimLaw-Assistant/
├── src/
│   ├── app.py                          # Main Streamlit application
│   ├── conditioned_answer_generator.py # Few-shot prompting system
│   └── utils/
│       ├── rag_chain.py               # RAG implementation
│       └── ingest.py                  # Document processing
├── data/
│   ├── clean/                         # Processed legal documents
│   └── legal_finetune_dataset.json   # Training dataset
├── vectorstore/
│   └── faiss_index/                  # Vector database
├── screenshots/
│   └── demo.png                      # Application screenshot
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Development

### Testing
Run the conditioning system tests:
```bash
python test_conditioned_generator.py
```

### Configuration
- Model selection in `src/utils/rag_chain.py`
- UI styling in `src/app.py` CSS section
- Few-shot examples in `src/conditioned_answer_generator.py`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit with descriptive messages: `git commit -m "feat: add new feature"`
6. Push to your fork: `git push origin feature-name`
7. Create a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Zimbabwe Legal Information Institute for legal document sources
- Ollama for local LLM deployment
- Streamlit for the web framework
- LangChain for LLM orchestration

## Demo

![ZimLaw Assistant Screenshot](screenshots/demo.png)

*Modern interface with real-time streaming responses and professional legal formatting*