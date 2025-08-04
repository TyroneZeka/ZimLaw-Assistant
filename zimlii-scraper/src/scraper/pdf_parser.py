import fitz  # PyMuPDF
import re
import os
import json
from typing import List, Dict, Any, Optional

class SimpleLegalParser:
    def __init__(self, output_dir: str = "./data/clean"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Patterns to identify and skip unwanted content
        self.skip_patterns = [
            r'^\s*table\s+of\s+contents\s*$',
            r'^\s*index\s+to\s+act\s*$',
            r'^\s*arrangement\s+of\s+sections\s*$',
            r'^\s*page\s+\d+\s*$',
            # r'^\s*\d+\s*$',  # Standalone numbers (often page numbers)
            r'CONSTITUTION\s+OF\s+ZIMBABWE',
            r'Government\s+Gazette',
            r'^\s*_+\s*$',   # Lines of underscores
            r'^\s*={3,}\s*$' # Lines of equal signs
        ]
        # Start markers for actual content
        self.content_markers = [
            r'chapter\s+1',
            r'part\s+1',
            r'interpretation',
            r'preliminary',
            # r'short\s+title',
            # r'^\s*1\s+[A-Z]',
            r'^\s*(\d+)\s*+([A-Z][^(\n]*?)(?:\s*$|\s+)'
        ]
        # Pattern for section headings: number followed by space and capital letter, no parentheses or dots in the number part
        self.section_heading_pattern = re.compile(r'^\s*(\d+)\s+([A-Z][^(\n]*?)')

    def debug_section(self, line: str) -> None:
        """Debug helper to print section matching attempts"""
        match = self.section_heading_pattern.match(line)
        if match:
            print(f"✓ Matched section: {line}")
            print(f"  Number: {match.group(1)}")
            print(f"  Title: {match.group(2)}")
        else:
            print(f"✗ Failed to match: {line}")
    
    def is_skip_line(self, line: str) -> bool:
        """Check if line should be skipped"""
        if not line.strip():
            return True
        return any(re.search(pattern, line.lower()) for pattern in self.skip_patterns)

    def is_content_start(self, line: str) -> bool:
        """Check if line indicates start of main content"""
        return any(re.search(pattern, line.lower()) for pattern in self.content_markers)

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        print(f"📄 Processing: {pdf_path}")
        doc = fitz.open(pdf_path)
        sections = []
        current_section_title = None
        current_section_content = []
        in_content = False
        pending_single_digit_section_number = None # Store a section number line waiting for its title
    
        print("Scanning for sections...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            # Corrected line split character
            lines = text.split('\n') 
            for line in lines:
                original_line = line # Keep original for potential debug
                line = line.strip()
                
                # --- Section Heading Logic (Two-line format) ---
                # Check if we have a pending single-digit number and this line could be its title
                if pending_single_digit_section_number is not None:
                    # Combine the stored number with the current line to form the full heading
                    combined_line = f"{pending_single_digit_section_number} {line}"
                    print(f"[DEBUG] Combining '{pending_single_digit_section_number}' + '{line}' -> '{combined_line}'")
                    
                    match = self.section_heading_pattern.match(combined_line)
                    if match:
                        print(f"✓ Matched two-line section: {combined_line}")
                        # Save previous section if exists
                        if current_section_title:
                            sections.append({
                                "title": current_section_title,
                                "content": '\n'.join(current_section_content).strip() # Corrected newline
                            })
                        # Start new section
                        current_section_title = combined_line
                        current_section_content = []
                        pending_single_digit_section_number = None # Clear the pending number
                        continue # Move to the next line, don't process this line further as a content line
                    else:
                        # If it doesn't match, the pending number was likely a page number or mis-identified.
                        # We should probably discard the pending number and process this line normally.
                        # However, let's be cautious. If the *next* line matches a pattern, we might reconsider.
                        # For now, let's assume it was a page number and clear it.
                        # A more sophisticated approach might involve buffering more lines.
                        print(f"[DEBUG] Combined line '{combined_line}' did not match section pattern. Discarding number '{pending_single_digit_section_number}'.")
                        pending_single_digit_section_number = None
                        # Fall through to process the current 'line' normally
                
                # --- General Line Processing ---
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for content start if we haven't found it yet
                if not in_content:
                    in_content = self.is_content_start(line)
                    if in_content:
                        print(f"Content start detected at: {line}")
    
                # Once we're in content, process lines
                if in_content:
                    # Check for single-digit section numbers (potential first line of two-line heading)
                    # Do this check *before* is_skip_line to prevent skipping valid section numbers.
                    # Only store it if we don't already have one pending.
                    if pending_single_digit_section_number is None and re.match(r'^\s*\d\s*$', line): # Exactly one digit
                         print(f"[DEBUG] Found potential single-digit section number line: '{line}'")
                         pending_single_digit_section_number = line.strip() # Store the stripped number
                         continue # Don't process this line further yet
                     
                    # Skip unwanted lines (but not potential section numbers anymore)
                    if self.is_skip_line(line):
                        print(f"[DEBUG] Skipped line: '{line}'") # Optional debug
                        continue
                    
                    # Check for regular (single-line) section headings
                    match = self.section_heading_pattern.match(line)
                    if match:
                        print(f"✓ Matched single-line section: {line}")
                        # Save previous section if exists
                        if current_section_title:
                            sections.append({
                                "title": current_section_title,
                                "content": '\n'.join(current_section_content).strip() # Corrected newline
                            })
                        # Start new section
                        current_section_title = line
                        current_section_content = []
                    elif line: # Add non-empty lines to current section content
                         current_section_content.append(line)
    
        # --- End of Processing ---
        # Handle any remaining pending single-digit number 
        # (This would be an edge case where the document ends with just a number line)
        if pending_single_digit_section_number is not None:
            print(f"[WARNING] Ended with unprocessed single-digit number: '{pending_single_digit_section_number}'")
            # Depending on requirements, you might add it as a section title with empty content,
            # or just discard it. Let's discard for now.
            
        # Don't forget the last section's content
        if current_section_title is not None:
            # Append content regardless of whether current_section_content is empty
            sections.append({
                "title": current_section_title,
                "content": '\n'.join(current_section_content).strip() # Corrected newline
            })
    
        return {
            "title": os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title(),
            "source_file": pdf_path,
            "sections": sections  # Return the structured sections
        }



    def save_output(self, parsed_content: Dict[str, Any]):
        """Save parsed content to both text and JSON formats"""
        base_name = re.sub(r'[^a-z0-9]+', '_', parsed_content["title"].lower())
        
        # Save as JSON (easier to handle structured data)
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_content, f, indent=2, ensure_ascii=False)
        print(f"📄 Saved JSON to: {json_path}")

        # Save as text (structured format)
        txt_path = os.path.join(self.output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"DOCUMENT: {parsed_content['title']}\n")
            f.write(f"SOURCE: {parsed_content['source_file']}\n")
            f.write(f"{'='*80}\n\n")
            
            # Iterate through structured sections
            for section in parsed_content.get("sections", []):
                f.write(f"--- {section['title']} ---\n\n")
                f.write(f"{section['content']}\n\n")
                
        print(f"📄 Saved text to: {txt_path}")

def main():
    parser = SimpleLegalParser()
    # Process all PDFs in the raw directory
    raw_dir = "./data/raw"
    for filename in os.listdir(raw_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(raw_dir, filename)
            try:
                content = parser.parse_pdf(pdf_path)
                parser.save_output(content)
                print(f"✅ Successfully processed: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()