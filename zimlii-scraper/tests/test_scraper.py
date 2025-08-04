import requests
from bs4 import BeautifulSoup
import os
import json

# Setup
OUTPUT_DIR = "./data/clean"
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
    """Scrape a ZimLII Act page"""
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
        section_data = extract_section_data(div)
        if section_data:
            sections.append(section_data)

    print(f"✅ Successfully extracted {len(sections)} valid sections")
    return sections, metadata

def extract_section_data(section_div):
    """Extract data from a section"""
    section_header = section_div.find(['h2', 'h3'])
    section_title = section_header.get_text(strip=True) if section_header else "No Title"

    # Extract main content
    content = extract_section_content(section_div)
    if not content:
        return None

    # Build section object
    return {
        "title": section_title,
        "text": content,
        "source_url": section_div.find('a', class_='akn-link')['href'] if section_div.find('a', class_='akn-link') else None
    }

def extract_section_content(section_div):
    """Extract section content including all nested elements"""
    content = []

    # Get main content
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

    # Extract lower part of the page
    lower_part = extract_lower_part(section_div)
    if lower_part:
        content.append(lower_part)

    return "\n".join(filter(None, content))

def process_subsection(subsec):
    """Process a subsection and its nested elements"""
    content = []

    # Get subsection header
    subsec_header = subsec.find('span', class_='akn-num')
    if subsec_header:
        content.append(f"{subsec_header.get_text(strip=True)}")

    # Get main subsection content
    subsec_content_div = subsec.find('span', class_='akn-content')
    if subsec_content_div:
        text = subsec_content_div.get_text(strip=True)
        if text:
            content.append(text)

    return "\n".join(filter(None, content))

def extract_lower_part(section_div):
    """Extract the lower part of the section"""
    lower_content = []
    lower_div = section_div.find('div', class_='akn-footer')
    if lower_div:
        for p in lower_div.find_all('p'):
            lower_content.append(p.get_text(strip=True))
    return "\n".join(lower_content)

def extract_content_block(element):
    """Extract clean text content from an element"""
    if not element:
        return ""
        
    content = []
    
    for node in element.descendants:
        if node.name is None and node.string and node.string.strip():
            text = node.string.strip()
            if text:
                content.append(text)
                    
    return " ".join(content).strip()

def save_to_text(sections, metadata, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"TITLE: {metadata['title']}\n")
        f.write(f"CHAPTER: {metadata['chapter']}\n")
        f.write(f"COMMENCED: {metadata['commencement']}\n")
        f.write(f"SOURCE: {metadata['source_url']}\n")
        f.write(f"{'='*80}\n\n")

        for sec in sections:
            f.write(f"{sec['title']}\n")
            f.write(f"{sec['text']}\n")
            f.write(f"[Reference: {sec['source_url']}]\n")
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
    URL = "https://zimlii.org/akn/zw/act/1955/28/eng@2016-12-31"

    result = scrape_zimlii_act(URL)
    if result:
        sections, metadata = result

        # Save outputs
        act_slug = metadata["title"].replace(" ", "_").lower()
        save_to_text(sections, metadata, f"{act_slug}.txt")
        save_to_json(sections, metadata, f"{act_slug}.json")

    print("✅ Scraping completed.")