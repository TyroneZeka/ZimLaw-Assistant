#!/usr/bin/env python3
"""
Test script for the Conditioned Answer Generator

This script demonstrates the improved answer quality using few-shot prompting
with examples from Zimbabwean criminal law and constitutional provisions.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.rag_chain import ZimLawRAGChain
from src.conditioned_answer_generator import ConditionedAnswerGenerator


def test_conditioned_answers():
    """Test the conditioned answer generator with various legal questions."""
    
    print("🎯 Testing Conditioned Answer Generator")
    print("=" * 60)
    
    try:
        # Initialize RAG chain
        print("🔧 Initializing RAG chain...")
        rag_chain = ZimLawRAGChain(
            top_k=10,
            final_k=5,
            enable_query_rewriting=True,
            verbose=False  # Clean output
        )
        
        # Initialize conditioned generator
        print("🎨 Initializing conditioned answer generator...")
        conditioned_generator = ConditionedAnswerGenerator(
            rag_chain.llm, 
            rag_chain,
            verbose=False  # Clean output
        )
        
        # Test questions (mix of examples and new questions)
        test_questions = [
            "What rights does a person have if they are arrested?",  # From examples
            "Can a 6-year-old child be charged with a crime?",  # Related to examples
            "What is the difference between intention and negligence in criminal law?",  # New question
            "When can Zimbabwe prosecute crimes committed in other countries?",  # From examples
            "What happens if someone is tried twice for the same crime?",  # Related to examples
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}: {question}")
            print('='*80)
            
            # Get conditioned answer
            print("\n🎨 CONDITIONED ANSWER (Few-Shot Prompting):")
            print("-" * 50)
            conditioned_result = conditioned_generator.generate_answer(question)
            print(conditioned_result['conditioned_answer'])
            
            print(f"\n📊 Metadata:")
            print(f"  Examples used: {conditioned_result.get('num_examples_used', 0)}")
            print(f"  Method: {conditioned_result.get('retrieval_method', 'Unknown')}")
            
            print(f"\n{'='*80}")
            
            # Small delay for readability
            import time
            time.sleep(1)
        
        print("\n✅ All tests completed successfully!")
        
        # Show available example questions
        print(f"\n📚 Available Few-Shot Examples ({len(conditioned_generator.few_shot_examples)}):")
        example_questions = conditioned_generator.get_example_questions()
        for i, q in enumerate(example_questions, 1):
            print(f"  {i}. {q}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def compare_answer_quality():
    """Compare regular RAG answers with conditioned answers."""
    
    print("\n🔬 ANSWER QUALITY COMPARISON")
    print("=" * 60)
    
    try:
        # Initialize components
        rag_chain = ZimLawRAGChain()
        conditioned_generator = ConditionedAnswerGenerator(rag_chain.llm, rag_chain)
        
        # Questions for comparison
        comparison_questions = [
            "What is criminal negligence?",
            "When can a child be held criminally responsible?",
            "What constitutes criminal conduct?"
        ]
        
        for question in comparison_questions:
            print(f"\n{'='*80}")
            print(f"COMPARISON: {question}")
            print('='*80)
            
            comparison_result = conditioned_generator.compare_answers(question)
            
            if comparison_result.get('comparison_available'):
                print("\n📊 REGULAR RAG ANSWER:")
                print("-" * 30)
                print(comparison_result['regular_answer'][:500] + "..." if len(comparison_result['regular_answer']) > 500 else comparison_result['regular_answer'])
                
                print("\n🎨 CONDITIONED ANSWER:")
                print("-" * 30)
                print(comparison_result['conditioned_answer'][:500] + "..." if len(comparison_result['conditioned_answer']) > 500 else comparison_result['conditioned_answer'])
                
                print(f"\n📚 Sources: {len(comparison_result.get('regular_sources', []))} retrieved")
            else:
                print(f"❌ Comparison failed: {comparison_result.get('error', 'Unknown error')}")
            
            print(f"\n{'='*80}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")


if __name__ == "__main__":
    # Run tests
    test_conditioned_answers()
    
    # Optionally run comparison (commented out to keep output manageable)
    # compare_answer_quality()
