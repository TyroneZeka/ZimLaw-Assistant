#!/usr/bin/env python3
"""
Conditioned Answer Generator for ZimLaw Assistant

This module implements a few-shot prompting strategy to improve answer quality
by providing the LLM with high-quality examples of the expected format.
"""

from typing import Dict, Any, List
from langchain.schema import Document


class ConditionedAnswerGenerator:
    """
    A class that uses few-shot prompting to condition the LLM to provide
    high-quality, well-structured legal answers in a consistent format.
    """
    
    def __init__(self, llm, rag_chain, verbose=False):
        """
        Initialize the conditioned answer generator.
        
        Args:
            llm: The language model instance (Ollama)
            rag_chain: The RAG chain for document retrieval
            verbose: Whether to show debug output
        """
        self.llm = llm
        self.rag_chain = rag_chain
        self.verbose = verbose
        self.few_shot_examples = self._create_few_shot_examples()
    
    def _create_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Create high-quality few-shot examples based on Zimbabwean criminal law and constitution.
        
        Returns:
            List of example Q&A pairs demonstrating the expected format
        """
        return [
            {
                "question": "What rights does a person have if they are arrested?",
                "answer": """**Direct Answer:** A person who is arrested has several fundamental constitutional rights that must be respected.

**Legal Basis:** According to the Constitution, Section 50, arrested persons have comprehensive rights including the right to be informed of the reason for arrest, the right to remain silent, and the right to legal representation.

**Key Rights:**
• Right to be informed of the reason for arrest immediately [Constitution, Section 50(1)(a)]
• Right to remain silent and not be compelled to make confessions [Constitution, Section 50(1)(b)]
• Right to contact and consult with a legal practitioner [Constitution, Section 50(1)(c)]
• Right to be brought before a court within 48 hours [Constitution, Section 50(2)]
• Right to be presumed innocent until proven guilty [Criminal Law (Codification and Reform) Act, Section 3(3)(a)]

**Additional Notes:** These rights are fundamental and cannot be waived. Any violation may result in evidence being excluded from court proceedings."""
            },
            
            {
                "question": "At what age can a child be held criminally responsible in Zimbabwe?",
                "answer": """**Direct Answer:** A child below seven years of age cannot be held criminally responsible, while children aged seven and above may face criminal charges depending on their capacity.

**Legal Basis:** The Criminal Law (Codification and Reform) Act establishes clear age thresholds for criminal responsibility.

**Key Provisions:**
• Children under 7 years: Deemed to lack criminal capacity completely [Criminal Law (Codification and Reform) Act, Section 7]
• Children 7-13 years: May be held responsible if prosecution proves criminal capacity
• Children 14 years and above: Presumed to have criminal capacity [Criminal Law (Codification and Reform) Act, Section 8]

**Additional Notes:** For children between 7-13 years, the court must determine whether the child had sufficient understanding to distinguish between right and wrong at the time of the alleged offense."""
            },
            
            {
                "question": "What constitutes criminal conduct under Zimbabwean law?",
                "answer": """**Direct Answer:** Criminal conduct may consist of either an act or an omission, provided it meets the legal requirements for criminal liability.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 10, defines criminal conduct and establishes when omissions can constitute crimes.

**Key Elements:**
• Acts: Any positive conduct that violates criminal law
• Omissions: Failure to act when legally required to do so [Criminal Law (Codification and Reform) Act, Section 10(2)]

**Legal Duty to Act Arises From:**
• Protective or family relationships requiring protection of another's life or safety [Section 10(2)(b)(i)]
• Prior conduct creating dangerous situations [Section 10(2)(b)(ii)]
• Assuming control over dangerous situations by contract [Section 10(2)(b)(iii)]
• Holding public office with specific duties [Section 10(2)(b)(iv)]

**Additional Notes:** Mere moral obligations do not create legal duties sufficient for criminal liability based on omissions."""
            },
            
            {
                "question": "What is the test for criminal intention in Zimbabwe?",
                "answer": """**Direct Answer:** Criminal intention is determined using a subjective test that examines whether the accused actually intended the conduct or consequence.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 13, establishes the subjective test for intention in criminal cases.

**Key Principles:**
• Subjective test: Court decides if the person actually possessed the required intention [Section 13(1)]
• Must intend the specific conduct or consequence that occurred
• Motive is generally irrelevant to criminal liability [Section 13(2)]

**Application:**
• Court considers all relevant factors influencing the accused's state of mind
• Direct evidence of intention not required - can be inferred from circumstances
• Test focuses on what the accused actually intended, not what a reasonable person would have intended

**Additional Notes:** This subjective approach differs from objective tests and emphasizes the actual mental state of the accused at the time of the offense."""
            },
            
            {
                "question": "When can someone be tried for a crime committed outside Zimbabwe?",
                "answer": """**Direct Answer:** Zimbabwe can prosecute crimes committed wholly or partly outside its borders in specific circumstances involving territorial jurisdiction.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 6, establishes extraterritorial jurisdiction for Zimbabwean courts.

**Jurisdiction Exists When:**
• Crime committed wholly inside Zimbabwe [Section 6(1)(a)]
• Crime partly outside Zimbabwe but completed inside Zimbabwe [Section 6(1)(b)]
• Crime against public security or State safety [Section 6(1)(c)(i)]
• Crime produced harmful effects in Zimbabwe [Section 6(1)(c)(ii)(A)]
• Crime intended to produce harmful effects in Zimbabwe [Section 6(1)(c)(ii)(B)]
• Crime committed with realization of risk of harmful effects in Zimbabwe [Section 6(1)(c)(ii)(C)]

**Additional Notes:** This section does not limit other enactments regulating territorial jurisdiction or making special provisions for extraterritorial crimes [Section 6(2)]."""
            },
            
            {
                "question": "What is the double jeopardy rule in Zimbabwe?",
                "answer": """**Direct Answer:** No person can be tried again for the same offense after being properly convicted or acquitted by a competent court.

**Legal Basis:** This principle is established in the Criminal Law (Codification and Reform) Act and constitutional provisions protecting against double jeopardy.

**Key Protection:**
• No retrial after conviction or acquittal by competent court [Criminal Law (Codification and Reform) Act, Section referenced in constitutional rights]
• Must be based on valid indictment, summons, or charge
• Extends to related offenses that could have been charged in original trial

**Requirements for Protection:**
• Trial by competent court
• Valid charges that could support judgment
• Final disposition (conviction or acquittal)

**Additional Notes:** This protection is fundamental to the criminal justice system and prevents the state from repeatedly prosecuting the same person for the same conduct."""
            },
            
            {
                "question": "What constitutes negligence in criminal law?",
                "answer": """**Direct Answer:** Criminal negligence is determined by an objective test examining whether the accused's conduct fell below the standard of a reasonable person in the same circumstances.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 16, establishes comprehensive tests for criminal negligence.

**Test Components:**
• **For Acts:** Whether a reasonable person would not have performed the act, or whether accused failed to exercise reasonable care [Section 16(1)(a)]
• **For Omissions:** Whether a reasonable person would not have omitted to act [Section 16(1)(b)]
• **For Consequences:** Two-part test examining failure to realize risk and whether reasonable person would have guarded against it [Section 16(1)(c)]

**Specific Applications:**
• Culpable homicide cases
• Negligently causing serious bodily harm
• Negligently causing serious property damage [Section 16(2)]

**Additional Notes:** The objective standard means personal limitations or inexperience generally do not excuse negligent conduct if a reasonable person would have acted differently."""
            },
            
            {
                "question": "What does 'realisation of real risk' mean in criminal law?",
                "answer": """**Direct Answer:** Realisation of real risk is a subjective test with two components: awareness of risk and reckless continuation of conduct despite that awareness.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 15, defines this concept which replaces the common law test for constructive intention.

**Two Components:**
• **Awareness:** Whether accused realized there was more than a remote risk that conduct might cause consequences or that circumstances existed [Section 15(1)(a)]
• **Recklessness:** Whether accused continued conduct despite realizing the risk [Section 15(1)(b)]

**Legal Implications:**
• Supersedes common law tests for constructive/legal intention [Section 15(4)]
• If awareness proved, recklessness inferred from actual consequences [Section 15(3)]
• Components may be implicit in statutory language [Section 15(2)]

**Additional Notes:** This test provides a clear framework for determining criminal liability in cases involving conscious risk-taking behavior."""
            },
            
            {
                "question": "Can Roman-Dutch law still be applied in Zimbabwe's criminal courts?",
                "answer": """**Direct Answer:** Roman-Dutch criminal law no longer applies in Zimbabwe to the extent that the Criminal Law Code expressly or impliedly replaces it.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 4, addresses the relationship between the new Code and pre-existing Roman-Dutch law.

**Key Provisions:**
• Roman-Dutch criminal law from Cape Colony (as of 1891) no longer applies where Code covers the matter [Section 4(1)]
• Courts may still reference judicial decisions and legal writings for interpretive guidance [Section 4(2)]
• Guidance may come from relevant aspects of former law or foreign criminal law [Section 4(2)(a)(b)]

**Practical Effect:**
• Code takes precedence over common law
• Historical authorities only used for interpretation, not as binding law
• Comprehensive codification reduces reliance on case law

**Additional Notes:** This represents a significant shift toward statutory criminal law while preserving interpretive resources from legal tradition."""
            },
            
            {
                "question": "How does the Code interact with other criminal legislation?",
                "answer": """**Direct Answer:** The Criminal Law Code works alongside other criminal enactments, with specific Code provisions applying to all criminal liability determinations.

**Legal Basis:** The Criminal Law (Codification and Reform) Act, Section 5, establishes the relationship between the Code and other criminal legislation.

**Key Principles:**
• Other criminal enactments remain valid and applicable [Section 5(1)]
• Section 5 and Chapters II and XII-XVI of Code apply to all criminal liability [Section 5(2)]
• Universal application unless expressly excluded by other enactment

**Practical Application:**
• Specialized criminal laws retain their specific provisions
• General principles from Code (capacity, intention, causation) apply universally
• Consistent standards across all criminal legislation

**Additional Notes:** This creates a unified framework for criminal liability while preserving specialized criminal statutes in areas like traffic, taxation, and regulatory offenses."""
            }
        ]
    
    def _format_few_shot_examples(self) -> str:
        """
        Format the few-shot examples into a prompt string.
        
        Returns:
            Formatted string containing all examples
        """
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"""### Example {i}
**Question:** {example['question']}
**Answer:**
{example['answer']}

"""
        return examples_text
    
    def _create_conditioning_prompt(self, context: str, question: str) -> str:
        """
        Create the full conditioning prompt with examples, context, and question.
        
        Args:
            context: Retrieved legal context
            question: User's question
            
        Returns:
            Complete prompt for the LLM
        """
        examples = self._format_few_shot_examples()
        
        prompt = f"""You are a precise legal assistant for Zimbabwean law. Provide your answer in the same style and format as the examples below.

{examples}

### Your Task
Use the following context to answer the user's question. Follow the style and format of the examples above.

**Context:**
{context}

**Question:** {question}

**Answer:**
"""
        return prompt
    
    def generate_answer_stream(self, question: str):
        """
        Generate a conditioned answer using few-shot prompting with streaming.
        
        Args:
            question: User's legal question
            
        Yields:
            Chunks of the generated answer
        """
        try:
            if self.verbose:
                print(f"🎯 Generating conditioned answer (streaming) for: {question}")
            
            # Get relevant context using the RAG chain
            rag_result = self.rag_chain.answer_question(question)
            
            # Extract context from retrieved documents
            if 'sources' in rag_result and rag_result['sources']:
                # Reconstruct context from sources if available
                context_pieces = []
                for source in rag_result['sources'][:5]:  # Use top 5 sources
                    context_piece = f"[{source['act']}, {source['chapter']}, Section {source['section']}]: {source.get('title', '')}"
                    context_pieces.append(context_piece)
                context = "\n\n".join(context_pieces)
            else:
                # Fallback to any existing context
                context = "No specific legal context retrieved."
            
            # Create the conditioning prompt
            conditioning_prompt = self._create_conditioning_prompt(context, question)
            
            # Generate answer using streaming
            if self.verbose:
                print("🤖 Generating conditioned response (streaming)...")
            
            # Use the stream method for streaming response
            for chunk in self.llm.stream(conditioning_prompt):
                yield chunk
                
        except Exception as e:
            print(f"❌ Error in conditioned answer generation: {str(e)}")
            yield f"Sorry, I encountered an error while generating a conditioned response: {str(e)}"

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate a conditioned answer using few-shot prompting.
        
        Args:
            question: User's legal question
            
        Returns:
            Dictionary containing the conditioned answer and metadata
        """
        try:
            if self.verbose:
                print(f"🎯 Generating conditioned answer for: {question}")
            
            # Get relevant context using the RAG chain
            rag_result = self.rag_chain.answer_question(question)
            
            # Extract context from retrieved documents
            if 'sources' in rag_result and rag_result['sources']:
                # Reconstruct context from sources if available
                context_pieces = []
                for source in rag_result['sources'][:5]:  # Use top 5 sources
                    context_piece = f"[{source['act']}, {source['chapter']}, Section {source['section']}]: {source.get('title', '')}"
                    context_pieces.append(context_piece)
                context = "\n\n".join(context_pieces)
            else:
                # Fallback to any existing context
                context = "No specific legal context retrieved."
            
            # Create the conditioning prompt
            conditioning_prompt = self._create_conditioning_prompt(context, question)
            
            # Generate answer using the conditioned prompt
            if self.verbose:
                print("🤖 Generating conditioned response...")
            conditioned_answer = self.llm.invoke(conditioning_prompt)
            
            return {
                "question": question,
                "conditioned_answer": conditioned_answer,
                "context_used": context,
                "num_examples_used": len(self.few_shot_examples),
                "retrieval_method": "few_shot_conditioning"
            }
            
        except Exception as e:
            print(f"❌ Error in conditioned answer generation: {str(e)}")
            return {
                "question": question,
                "conditioned_answer": "Sorry, I encountered an error while generating a conditioned response.",
                "error": str(e),
                "retrieval_method": "few_shot_conditioning"
            }
    
    def get_example_questions(self) -> List[str]:
        """
        Get a list of example questions for testing purposes.
        
        Returns:
            List of example questions from the few-shot examples
        """
        return [example["question"] for example in self.few_shot_examples]
    
    def compare_answers(self, question: str) -> Dict[str, Any]:
        """
        Compare regular RAG answer with conditioned answer.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing both answers for comparison
        """
        try:
            # Get regular RAG answer
            regular_result = self.rag_chain.answer_question(question)
            
            # Get conditioned answer
            conditioned_result = self.generate_answer(question)
            
            return {
                "question": question,
                "regular_answer": regular_result.get("answer", "No answer generated"),
                "conditioned_answer": conditioned_result.get("conditioned_answer", "No answer generated"),
                "regular_sources": regular_result.get("sources", []),
                "conditioned_context": conditioned_result.get("context_used", ""),
                "comparison_available": True
            }
            
        except Exception as e:
            print(f"❌ Error in answer comparison: {str(e)}")
            return {
                "question": question,
                "error": str(e),
                "comparison_available": False
            }


def create_conditioned_generator(llm, rag_chain):
    """
    Factory function to create a ConditionedAnswerGenerator instance.
    
    Args:
        llm: The language model instance
        rag_chain: The RAG chain instance
        
    Returns:
        ConditionedAnswerGenerator instance
    """
    return ConditionedAnswerGenerator(llm, rag_chain)


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 ConditionedAnswerGenerator module loaded successfully!")
    print(f"📚 Module contains {len(ConditionedAnswerGenerator(None, None).few_shot_examples)} few-shot examples")
    
    # Display example questions
    generator = ConditionedAnswerGenerator(None, None)
    example_questions = generator.get_example_questions()
    
    print("\n📋 Available example questions:")
    for i, question in enumerate(example_questions, 1):
        print(f"  {i}. {question}")
