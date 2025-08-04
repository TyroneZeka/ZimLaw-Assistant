# src/utils/zimlii_scraper.py

import requests
from bs4 import BeautifulSoup
import time
import os
import json
from urllib.parse import urljoin

# Setup
OUTPUT_DIR = "./data/clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_page_metadata(soup, url):
    """Extract document-level metadata"""
    # Title: e.g., "Dangerous Drugs Act"
    title_tag = soup.find('h1') or soup.find('h2', class_='doc-title')
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Act"

    # Chapter: Look for "Chapter 15:02"
    chapter_tag = soup.find(string=lambda x: x and "Chapter" in x and ":" in x)
    chapter = chapter_tag.strip() if chapter_tag else "Unknown"

    # Commencement date
    commencement_tag = soup.find(string=lambda x: x and "Commenced on" in x)
    commencement = commencement_tag.strip() if commencement_tag else "Unknown"

    return {
        "title": title,
        "chapter": chapter,
        "commencement": commencement,
        "source_url": url,
        "version_date": "31 December 2016"  # From footer note
    }

def is_repealed_section(section_div):
    """Check if section is repealed"""
    text = section_div.get_text(strip=True)
    if "[section repealed by" in text.lower():
        return True
    return False


def scrape_zimlii_act(url, filename_prefix=None):
    """
    Scrape a ZimLII Act page (e.g., Dangerous Drugs Act)
    Returns list of sections with metadata
    """
    print(f"🔍 Fetching: {url}")
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch {url}: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract document metadata
    metadata = get_page_metadata(soup, url)
    print(f"📘 Found: {metadata['title']} ({metadata['chapter']})")

    sections = []
    all_section_divs = soup.find_all('section', class_='akn-section')

    print(f"📦 Found {len(all_section_divs)} section blocks")

    for div in all_section_divs:
        num_tag = div.find('span', class_='akn-num')
        section_number = num_tag.get_text(strip=True).rstrip('.') if num_tag else None

        if not section_number:
            continue  
        
        # Clean section number: remove trailing dots, e.g., "14H."
        section_number = section_number.rstrip('.')

        # Skip repealed sections
        if is_repealed_section(div):
            print(f"🗑️ Skipping repealed section: {section_number}")
            continue

        # Extract title
        title_tag = div.find('span', class_='akn-num')
        title = title_tag.get_text(strip=True) if title_tag else "No Title"

        # Extract clean content
        content = extract_section_content(div)
        if not content:
            continue

        # Build section object
        section_data = {
            "section": section_number,
            "title": title,
            "text": content,
            "part": None,  # Will infer from context later
            "source_url": url,
            "document": metadata["title"],
            "chapter": metadata["chapter"]
        }

        # Try to infer Part (e.g., "Part V – Control of dangerous drugs")
        parent = div
        for _ in range(5):
            parent = parent.find_parent()
            if parent and parent.name == 'akn-section' and 'akn-part' in parent.get('class', []):
                part_header = parent.find(['h2'], string=True)
                if part_header:
                    section_data["part"] = part_header.get_text(strip=True)
                break

        sections.append(section_data)

    print(f"✅ Successfully extracted {len(sections)} valid sections")
    return sections, metadata

def extract_section_content(section_div):
    """Extract section content including all nested elements"""
    content = []
    
    # Get section header from h2 or h3
    section_header = section_div.find(['h2', 'h3'])
    if section_header:
        header_text = section_header.get_text(strip=True)
        content.append(f"{header_text}")  # Remove "Section" prefix as it's in the header
    
    # Check if section is repealed
    if is_repealed_section(section_div):
        return None
    
    # Get main content (excluding subsections)
    main_content_div = section_div.find('span', class_='akn-content')
    if main_content_div:
        main_text = extract_content_block(main_content_div)
        if main_text:
            content.append(main_text)
    
    # Process subsections
    subsections = section_div.find_all('section', class_='akn-subsection')
    for subsec in subsections:
        subsec_content = process_subsection(subsec)
        if subsec_content:
            content.append(subsec_content)
            
    return "\n".join(filter(None, content))

def process_subsection(subsec):
    """Process a subsection and its nested elements"""
    content = []
    
    # Get subsection number without extra parentheses
    num_tag = subsec.find('span', class_='akn-num')
    if num_tag:
        num_text = num_tag.get_text(strip=True).strip('()')
        content.append(f"({num_text})")
    
    # Get main subsection content
    subsec_content_div = subsec.find('span', class_='akn-content')
    if subsec_content_div:
        text = subsec_content_div.get_text(strip=True)
        if text and not is_repealed_text(text):
            content.append(text)
    
    # Handle nested paragraphs
    paragraphs = subsec.find_all(['span', 'div'], class_='akn-p')
    for p in paragraphs:
        # Skip if it's just a number
        if not p.get('class') or 'akn-num' not in p.get('class'):
            text = p.get_text(strip=True)
            if text:
                content.append(text)
    
    return " ".join(filter(None, content))

def is_repealed_text(text):
    """Check if text indicates a repealed section"""
    repealed_markers = [
        "[section repealed",
        "[subsection repealed",
        "[repealed by",
        "***"
    ]
    return any(marker in text.lower() for marker in repealed_markers)

def extract_content_block(element):
    """Extract clean text content from an element"""
    if not element:
        return ""
        
    content = []
    
    # Get all text nodes, excluding specific classes and repealed content
    for node in element.descendants:
        if node.name is None and node.string and node.string.strip():
            text = node.string.strip()
            if text and not is_repealed_text(text):
                if not any(c in node.parent.get('class', []) for c in ['akn-num', 'akn-remark']):
                    content.append(text)
                    
    return " ".join(content).strip()

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
            # if sec["part"]:
            #     f.write(f"\nPART: {sec['part']}\n\n")
            
            # Write section header if present
            if "header" in sec:
                f.write(f"{sec['header']}\n")
            
            # f.write(f"Section {sec['section']}\n")
            f.write(f"{sec['text']}\n")
            f.write(f"[Reference: {sec['source_url']}#{sec['section']}]\n")
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
    URL = "https://zimlii.org/akn/zw/act/1955/28/eng@2016-12-31"

    result = scrape_zimlii_act(URL, filename_prefix="dangerous_drugs")
    if result:
        sections, metadata = result

        # Save outputs
        act_slug = metadata["title"].replace(" ", "_").lower()
        save_to_text(sections, metadata, f"{act_slug}.txt")
        save_to_json(sections, metadata, f"{act_slug}.json")

    print("✅ Scraping completed.")
    time.sleep(1)