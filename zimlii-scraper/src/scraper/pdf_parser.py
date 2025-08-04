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
        # Update the skip_patterns in __init__
        self.skip_patterns = [
            r'^\s*table\s+of\s+contents\s*$',
            r'^\s*index\s+to\s+act\s*$',
            r'^\s*arrangement\s+of\s+sections\s*$',
            r'^\s*page\s+\d+\s*$',
            r'^\s*\d+\s*$',  # This will catch standalone page numbers
            r'^CONSTITUTION\s+OF\s+ZIMBABWE\s*$',  # Updated to match header exactly
            r'^\s*CONSTITUTION\s+OF\s+ZIMBABWE\s*\d+\s*$',  # Matches header with page number
            r'Government\s+Gazette',
            r'^\s*_+\s*$',   # Lines of underscores
            r'^\s*={3,}\s*$', # Lines of equal signs
            # Add these new patterns
            r'^\s*\[.*\]\s*$',  # Matches anything in square brackets
            r'^\s*\d+\s*$'      # Matches standalone numbers (like page numbers)
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
            
    def is_header_or_footer(self, block) -> bool:
        """Check if a text block is likely a header or footer based on its position"""
        # block[1] is y0 (top position), block[3] is y1 (bottom position)
        # Typical A4 page height is around 842 points
        page_height = 842
        header_zone = 50  # First 50 points from top
        footer_zone = 792  # Last 50 points of page
        
        y_pos = block[1]  # Top position of the text block
        
        return (y_pos < header_zone) or (y_pos > footer_zone)
    
    def is_skip_line(self, line: str) -> bool:
        """Check if line should be skipped"""
        if not line.strip():
            return True
        return any(re.search(pattern, line.lower()) for pattern in self.skip_patterns)

    def is_content_start(self, line: str) -> bool:
        """Check if line indicates start of main content"""
        return any(re.search(pattern, line.lower()) for pattern in self.content_markers)

    # ... (Keep your existing __init__, is_skip_line, is_content_start methods) ...

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        print(f"📄 Processing: {pdf_path}")
        doc = fitz.open(pdf_path)
        sections = []
        current_section_title = None
        current_section_content = []
        in_content = False
        pending_section_number = None # Variable to hold a single-digit number line

        print("Scanning for sections...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            # Use the correct newline character for splitting
            lines = text.split('\n') 
            for line in lines:
                original_line = line # Keep original for potential debugging
                line = line.strip()

                # --- Potential Section Header Debug ---
                # Check for potential section headers (BEFORE skipping or other logic)
                # This helps us see if lines like "1" or "1 The Republic" are present
                if re.match(r'^\s*\d+\s*[A-Z]?', line): # Allow number-only or number+title start
                     print(f"[DEBUG] Potential section part found: '{original_line}' (stripped: '{line}')")

                # --- Check for Single-Digit Section Number ---
                # Do this check early, before is_skip_line, to catch potential section numbers
                # Only check if we are in content and don't already have a pending number
                if in_content and pending_section_number is None:
                    # Check if the line is exactly a single digit (allowing surrounding whitespace)
                    if re.match(r'^\s*\d\s*$', line):
                        print(f"[INFO] Found potential single-digit section number: '{line}'")
                        pending_section_number = line.strip() # Store the clean number
                        # Important: Continue to the next line, do not process this line further yet
                        continue 

                # --- Skip Unwanted Lines ---
                # Now apply the skip logic to the current line
                # Note: A line that was just stored as pending_section_number will not be re-processed here
                # because we used 'continue' above.
                if self.is_skip_line(line):
                    print(f"[DEBUG] Skipped line: '{original_line}'") # Optional debug
                    continue

                # --- Content Start Detection ---
                if not in_content:
                    in_content = self.is_content_start(line)
                    if in_content:
                        print(f"Content start detected at: {line}")

                # --- Process Lines Once In Content ---
                if in_content:
                    # --- Handle Pending Single-Digit Number ---
                    # If we had stored a single-digit number, try combining it with the current line
                    if pending_section_number is not None:
                        combined_line = f"{pending_section_number} {line}"
                        print(f"[DEBUG] Trying combined line: '{combined_line}'")

                        # Test the combined line against the section pattern
                        match = self.section_heading_pattern.match(combined_line)
                        if match:
                            print(f"✓ Matched TWO-LINE section: '{combined_line}'")
                            print(f"  Number: {match.group(1)}")
                            print(f"  Title: {match.group(2)}")

                            # Save the previous section (if any)
                            if current_section_title:
                                sections.append({
                                    "title": current_section_title,
                                    "content": '\n'.join(current_section_content).strip() # Use correct newline
                                })

                            # Start the NEW section with the combined title
                            current_section_title = combined_line
                            current_section_content = []
                            pending_section_number = None # Clear the pending number
                            # Important: Continue, do not process 'line' as content for the *new* section
                            continue 
                        else:
                            # If the combination didn't work, the stored number was likely a page number.
                            print(f"[INFO] Combined line '{combined_line}' did NOT match section pattern. Discarding number '{pending_section_number}'.")
                            # We should probably add the stored number as content now? Or just discard?
                            # Let's discard for simplicity and assume it was a page number.
                            pending_section_number = None
                            # Fall through to process the current 'line' normally (it might be content or another header)

                    # --- Check for Regular (Single-Line) Section Headers ---
                    # This check happens if:
                    # 1. There was no pending number, OR
                    # 2. The pending number combination failed and we fell through
                    match = self.section_heading_pattern.match(line)
                    if match:
                        print(f"✓ Matched SINGLE-LINE section: '{line}'")
                        print(f"  Number: {match.group(1)}")
                        print(f"  Title: {match.group(2)}")

                        # Save the previous section (if any)
                        if current_section_title:
                            sections.append({
                                "title": current_section_title,
                                "content": '\n'.join(current_section_content).strip() # Use correct newline
                            })

                        # Start the NEW section
                        current_section_title = line
                        current_section_content = []

                    # --- Add Content Lines ---
                    # This part runs if:
                    # 1. The line was not a section header (single or combined).
                    # 2. We have an active section (current_section_title is set).
                    elif current_section_title and line: # Add non-empty lines to content
                         current_section_content.append(line)

        # --- End of File Processing ---
        # Handle any leftover pending number (unlikely, but good practice)
        if pending_section_number is not None:
            print(f"[WARNING] Reached end of file with unprocessed number: '{pending_section_number}'")
            # Optionally add it as content or discard. Discarding for now.

        # Don't forget the LAST section's content
        if current_section_title is not None:
            sections.append({
                "title": current_section_title,
                "content": '\n'.join(current_section_content).strip() # Use correct newline
            })

        return {
            "title": os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title(),
            "source_file": pdf_path,
            "sections": sections  # Return the structured sections
        }

# ... (Keep your existing save_output and main methods) ...



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