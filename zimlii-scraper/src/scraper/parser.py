# filepath: d:\Workspace\ZimLaw-Assistant\src\utils\zimlii_scraper.py
# src/utils/zimlii_scraper.py

import requests
from bs4 import BeautifulSoup
import os
import json

# Setup
OUTPUT_DIR = "./data/raw/acts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_page_metadata(soup, url):
    """Extract document-level metadata"""
    title_tag = soup.find('h1') or soup.find('h2', class_='doc-title')
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Act"

    chapter_tag = soup.find(string=lambda x: x and "Chapter" in x and ":" in x)
    chapter = chapter_tag.strip() if chapter_tag else "Unknown"

    commencement_tag = soup.find(string=lambda x: x and "Commenced on" in x)
    commencement = commencement_tag.strip() if commencement_tag else "Unknown"

    return {
        "title": title,
        "chapter": chapter,
        "commencement": commencement,
        "source_url": url,
        "version_date": "31 December 2016"  # From footer note
    }
    
def clean_text(text: str) -> str:
    """Enhanced text cleaning"""
    if not text:
        return ""
        
    # Fix common spacing issues
    replacements = {
        "theacquiring authority": "the acquiring authority",
        "Theacquiring authority": "The acquiring authority",
        "acquiring authority acquiring authority": "acquiring authority",
        "designated valuation officer designated valuation officer": "designated valuation officer",
        "theMinister": "the Minister",
        "theCouncil": "the Council",
    }
    
    cleaned = text
    
    # Apply replacements
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Fix multiple spaces and normalize
    cleaned = " ".join(cleaned.split())
    
    return cleaned.strip()

def scrape_zimlii_act(url):
    """Scrape a ZimLII Act page with improved section handling"""
    print(f"🔍 Fetching: {url}")
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch {url}: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    metadata = get_page_metadata(soup, url)
    
    sections = []
    current_main_section = None
    
    # Process main sections
    for section_div in soup.find_all('section', class_='akn-section'):
        section_data = extract_section_data(section_div, current_main_section)
        if section_data:
            if section_data['section_type'] == 'main':
                current_main_section = section_data['section']
            sections.append(section_data)
    
    # Process schedules
    schedules = extract_schedules(soup)
    if schedules:
        sections.extend(schedules)
        
    print(f"✅ Successfully extracted {len(sections)} sections")
    return sections, metadata

def extract_section_number(section_div) -> tuple:
    """Extract proper section number and type from section div"""
    section_num = section_div.find('span', class_='akn-num')
    if not section_num:
        return None, None
        
    num_text = section_num.get_text(strip=True).strip()
    
    # Handle different section number formats
    if num_text.startswith('(') and num_text.endswith(')'):
        # Subsection like (1) or (a)
        num = num_text.strip('()')
        section_type = 'subsection'
    elif '.' in num_text:
        # Main section like "1." or "3D."
        num = num_text.rstrip('.')
        section_type = 'main'
    else:
        num = num_text
        section_type = 'other'
        
    return num, section_type

def extract_section_data(section_div, current_main_section=None):
    """Extract section data using HTML IDs for accurate section numbering"""
    # Get section ID from the HTML
    section_id = section_div.get('id', '')
    if not section_id:
        return None
    
    # Parse section number from ID (e.g., "sec_3" -> "3")
    main_section = section_id.split('_')[1] if section_id.startswith('sec_') else None
    
    # Get section title
    title_tag = section_div.find(['h2', 'h3'])
    if not title_tag:
        return None
    
    title = title_tag.get_text(strip=True)
    if title.startswith(f"{main_section}."):
        title = title[len(f"{main_section}."):]
    title = title.strip()
    # Process content based on structure
    content = []
    
    # Handle subsections
    subsections = section_div.find_all('section', class_='akn-subsection')
    if subsections:
        for subsec in subsections:
            subsec_content = process_subsection(subsec)
            if subsec_content:
                content.append(subsec_content)
    else:
        # Direct content if no subsections
        main_content = extract_section_content(section_div)
        if main_content:
            content.append(main_content)
    
    return {
        "section": main_section,
        "section_type": "main",
        "title": title,
        "text": clean_text(" ".join(content)),
    }

def extract_section_content(section_div):
    """Extract section content with improved text cleaning"""
    content_parts = []
    
    # Process direct content
    direct_content = section_div.find('span', class_='akn-content', recursive=False)
    if direct_content:
        content_parts.append(extract_content_block(direct_content))
    
    # Process subsections
    for subsec in section_div.find_all('section', class_='akn-subsection', recursive=False):
        subsec_content = process_subsection(subsec)
        if subsec_content:
            content_parts.append(subsec_content)
    
    # Process paragraphs
    for para in section_div.find_all('section', class_='akn-paragraph', recursive=False):
        para_content = process_paragraph_block(para)
        if para_content:
            content_parts.append(para_content)
    
    return " ".join(filter(None, content_parts))

def process_paragraph_block(para_block):
    """Process an akn-paragraph block and its content"""
    content_parts = []

    # Get the paragraph number/letter (e.g., (a), (b))
    num_tag = para_block.find('span', class_='akn-num')
    if num_tag:
        num_text = num_tag.get_text(strip=True)
        content_parts.append(f"{num_text}")

    # Get the paragraph content
    content_span = para_block.find('span', class_='akn-content')
    if content_span:
        # Use extract_content_block to handle nested <span class="akn-p"> tags
        para_text = extract_content_block(content_span)
        if para_text:
            content_parts.append(para_text)

    return " ".join(content_parts)

def process_subsection(subsec):
    """Process a subsection with improved ID-based handling"""
    subsec_id = subsec.get('id', '')
    if not subsec_id:
        return None
    
    content_parts = []
    
    # Get subsection number
    num_tag = subsec.find('span', class_='akn-num')
    if num_tag:
        content_parts.append(num_tag.get_text(strip=True))
    
    # Process main content
    content_span = subsec.find('span', class_='akn-content')
    if content_span:
        # Process paragraphs within content
        paragraphs = content_span.find_all('span', class_='akn-p')
        if paragraphs:
            for para in paragraphs:
                para_text = clean_text(extract_content_block(para))
                if para_text:
                    content_parts.append(para_text)
        else:
            # Direct content if no paragraphs
            content_text = clean_text(extract_content_block(content_span))
            if content_text:
                content_parts.append(content_text)
    
    return " ".join(content_parts)

def extract_content_block(element):
    """Extract clean text content with improved term handling"""
    if not element:
        return ""
    
    content = []
    
    for node in element.descendants:
        if node.name is None and node.string:  # Text node
            text = node.string.strip()
            if text:
                content.append(text)
        elif node.name == 'span' and 'akn-term' in node.get('class', []):
            # Handle terms with proper spacing
            term_text = node.get_text(strip=True)
            content.append(f" {term_text} ")
    
    return " ".join(content).strip()

def extract_schedules(soup):
    """Extract schedule content from akn-attachments"""
    schedules = []
    attachments = soup.find('span', class_='akn-attachments')
    
    if not attachments:
        return []
        
    for schedule in attachments.find_all('div', class_='akn-attachment'):
        # Get schedule header
        header = schedule.find('h2', class_='akn-heading')
        subheader = schedule.find('h2', class_='akn-subheading')
        
        schedule_title = ""
        if header:
            schedule_title = header.get_text(strip=True)
        if subheader:
            schedule_title += f" - {subheader.get_text(strip=True)}"
            
        parts = []
        # Process each part in the schedule
        for part in schedule.find_all('section', class_='akn-part'):
            part_data = extract_schedule_part(part)
            if part_data:
                parts.append(part_data)
                
        schedule_data = {
            "section": "Schedule",
            "title": schedule_title,
            "parts": parts,
            "text": "\n\n".join([f"{part['title']}\n{part['content']}" for part in parts]),
            "type": "schedule"
        }
        
        schedules.append(schedule_data)
        
    return schedules

def extract_schedule_part(part):
    """Extract content from a schedule part"""
    title = part.find('h2')
    title_text = title.get_text(strip=True) if title else ""
    
    content = []
    
    # Get all paragraphs in this part
    paragraphs = part.find_all(['span', 'section'], class_=['akn-p', 'akn-paragraph', 'akn-intro'])
    
    for para in paragraphs:
        # Skip editorial remarks
        if para.find('span', class_='akn-remark'):
            continue
            
        # Get paragraph number if exists
        num = para.find('span', class_='akn-num')
        num_text = num.get_text(strip=True) if num else ""
        
        # Get paragraph content
        para_content = para.find('span', class_='akn-content')
        if para_content:
            text = para_content.get_text(strip=True)
        else:
            text = para.get_text(strip=True)
            
        if text and not text.startswith('['):
            if num_text:
                content.append(f"{num_text} {text}")
            else:
                content.append(text)
    
    return {
        "title": title_text,
        "content": "\n".join(content)
    }

def save_to_text(sections, metadata, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write metadata
        f.write(f"{'='*80}\n")
        f.write(f"TITLE: {metadata['title']}\n")
        f.write(f"CHAPTER: {metadata['chapter']}\n")
        f.write(f"COMMENCED: {metadata['commencement']}\n")
        f.write(f"SOURCE: {metadata['source_url']}\n")
        f.write(f"{'='*80}\n\n")

        # Write sections with cleaner formatting
        for sec in sections:
            f.write(f"{sec['title']}\n")
            if sec.get('type') == 'schedule':
                # Special formatting for schedules
                f.write(f"{sec['text']}\n")
            else:
                f.write(f"{sec['text']}\n")
            f.write(f"[Reference: {sec.get('source_url', metadata['source_url'])}]\n")
            f.write("\n" + "-"*60 + "\n")
    
    print(f"📄 Saved clean text to {filepath}")

def save_to_json(sections, metadata, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    data = {
        "metadata": metadata,
        "sections": sections
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"📄 Saved structured JSON to {filepath}")

# === MAIN ===
if __name__ == "__main__":
    # Test with Dangerous Drugs Act
    URL = "https://zimlii.org/akn/zw/act/2004/7/eng@2016-12-31"

    result = scrape_zimlii_act(URL)
    if result:
        sections, metadata = result

        # Save outputs
        act_slug = metadata["title"].replace(" ", "_").lower()
        # save_to_text(sections, metadata, f"{act_slug}.txt")
        save_to_json(sections, metadata, f"{act_slug}.json")

    print("✅ Scraping completed.")