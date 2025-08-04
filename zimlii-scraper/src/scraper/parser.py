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

def scrape_zimlii_act(url):
    """Scrape a ZimLII Act page (e.g., Dangerous Drugs Act)"""
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
    
    # Process main sections
    for div in all_section_divs:
        section_data = extract_section_data(div)
        if section_data:
            sections.append(section_data)
            
    # Process schedules
    schedules = extract_schedules(soup)
    if schedules:
        sections.extend(schedules)

    print(f"✅ Successfully extracted {len(sections)} sections and schedules")
    return sections, metadata

def extract_section_data(section_div, last_section_num=[0]):
    section_number_tag = section_div.find('span', class_='akn-num')
    section_title = section_div.find(['h2', 'h3'])
    
    # Return None only if the title is missing (a section must have a title)
    if not section_title:
        return None

    section_title_text = section_title.get_text(strip=True)

    # Logic for determining the section number
    if section_number_tag:
        num_text = section_number_tag.get_text(strip=True).rstrip('.').strip()
        try:
            current_num = int(num_text)
            last_section_num[0] = current_num  # Update the counter to this new number
        except ValueError:
            current_num = num_text
    else:
        last_section_num[0] += 1
        current_num = last_section_num[0]

    content = extract_section_content(section_div)

    return {
        "section": str(current_num), 
        "title": section_title_text,
        "text": content,
        "source_url": section_div.find('a', class_='source-url')['href'] if section_div.find('a', class_='source-url') else None
    }

def extract_section_content(section_div):
    """Extract section content including all nested elements"""
    content = []

    # Process all direct child sections (like akn-subsection, akn-paragraph)
    # Use recursive=False to only get immediate children
    child_sections = section_div.find_all('section', recursive=False)
    for child_sec in child_sections:
        paragraph_content = []

        # Extract text from akn-intro, if present
        intro = child_sec.find('span', class_=['akn-intro', 'akn-content'])
        if intro:
            intro_text = intro.get_text(strip=True)
            if intro_text:
                paragraph_content.append(intro_text)

        # Extract text from akn-paragraph, if present
        # This finds any section with class akn-paragraph that is a direct child of child_sec
        para_blocks = child_sec.find_all('section', class_='akn-paragraph', recursive=False)
        for para_block in para_blocks:
            # Get the paragraph letter (a), (b), etc.
            num_tag = para_block.find('span', class_='akn-num')
            # Get the paragraph content
            content_span = para_block.find('span', class_='akn-content')
            if content_span:
                para_text = content_span.get_text(strip=True)
                if num_tag and para_text:
                    paragraph_content.append(f"{num_tag.get_text(strip=True)} {para_text}")
                elif para_text:
                    paragraph_content.append(para_text)

        # If we found content for this child section, join it and add to main content
        if paragraph_content:
            content.append(" ".join(paragraph_content))

    # Also check if there is any direct akn-content in the main section (for simpler sections)
    direct_content = section_div.find('span', class_='akn-content', recursive=False)
    if direct_content:
        text = direct_content.get_text(strip=True)
        if text:
            content.append(text)

    return " ".join(filter(None, content))

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
    """Process a subsection and its nested elements"""
    content = []

    # Get subsection header
    num_tag = subsec.find('span', class_='akn-num')
    if num_tag:
        num_text = num_tag.get_text(strip=True).strip('()')
        content.append(f"({num_text})")

    # Get main subsection content
    subsec_content_div = subsec.find('span', class_='akn-content')
    if subsec_content_div:
        text = subsec_content_div.get_text(strip=True)
        if text:
            content.append(text)

    return " ".join(filter(None, content))

def extract_content_block(element):
    """Extract clean text content from an element"""
    if not element:
        return ""
        
    content = []
    
    # Get all text nodes, excluding specific classes
    for node in element.descendants:
        if node.name is None and node.string and node.string.strip():
            text = node.string.strip()
            if text and not any(c in node.parent.get('class', []) for c in ['akn-num', 'akn-remark']):
                content.append(text)
                    
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
        save_to_text(sections, metadata, f"{act_slug}.txt")
        save_to_json(sections, metadata, f"{act_slug}.json")

    print("✅ Scraping completed.")