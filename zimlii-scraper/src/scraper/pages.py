import requests
from bs4 import BeautifulSoup
import csv
import time
import logging
import os
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZimLIIScraper:
    def __init__(self):
        self.base_url = "https://zimlii.org/legislation/"
        self.domain = "https://zimlii.org"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.output_dir = "./data/raw"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_page(self, url):
        """Fetch a page with error handling and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url}: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def get_total_pages(self):
        """Get the total number of pages from pagination"""
        response = self.get_page(self.base_url)
        if not response:
            return 0
        
        soup = BeautifulSoup(response.content, 'html.parser')
        pagination = soup.find('ul', class_='pagination')
        if not pagination:
            return 0
            
        # Find all page numbers and get the highest one
        page_items = pagination.find_all('li', class_='page-item')
        pages = []
        for item in page_items:
            link = item.find('a')
            if link and link.get_text().isdigit():
                pages.append(int(link.get_text()))
        
        return max(pages) if pages else 0

    def scrape_legislation_links(self):
        """Scrape all legislation links from all pages"""
        legislation_data = []
        total_pages = self.get_total_pages()
        
        if not total_pages:
            logger.error("Could not determine total number of pages")
            return []

        logger.info(f"Found {total_pages} pages to scrape")

        for page_num in range(1, total_pages + 1):
            url = f"{self.base_url}?page={page_num}"
            logger.info(f"Scraping page {page_num} of {total_pages}")
            
            response = self.get_page(url)
            if not response:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', class_='doc-table')
            
            if not table:
                logger.warning(f"No table found on page {page_num}")
                continue

            # Find all rows in the table
            rows = table.find_all('tr')
            
            for row in rows:
                title_cell = row.find('td', class_='cell-title')
                if not title_cell:
                    continue
                    
                link = title_cell.find('a')
                if not link:
                    continue

                href = link.get('href')
                if not href:
                    continue

                title = link.get_text(strip=True)
                full_url = urljoin(self.domain, href)
                
                citation_cell = row.find('td', class_='cell-citation')
                citation = citation_cell.get_text(strip=True) if citation_cell else ""

                legislation_data.append({
                    'title': title,
                    'url': full_url,
                    'citation': citation
                })

            # Be nice to the server
            time.sleep(1)

        return legislation_data

    def save_to_csv(self, data):
        """Save the scraped data to a CSV file"""
        if not data:
            logger.warning("No data to save")
            return

        filepath = os.path.join(self.output_dir, 'zimlii_legislation.csv')
        fieldnames = ['title', 'url', 'citation']

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Successfully saved {len(data)} items to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save CSV file: {e}")

def main():
    scraper = ZimLIIScraper()
    legislation_data = scraper.scrape_legislation_links()
    scraper.save_to_csv(legislation_data)

if __name__ == "__main__":
    main()