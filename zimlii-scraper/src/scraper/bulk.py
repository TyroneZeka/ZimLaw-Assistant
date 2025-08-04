import sys
import time
import random
import csv
import logging
from pathlib import Path
from datetime import datetime
from parser import scrape_zimlii_act, save_to_text, save_to_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BulkScraper:
    def __init__(self):
        self.input_file = Path('./data/raw/zimlii_legislation.csv')
        self.output_dir = Path('./data/raw/acts')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scraping parameters
        self.delay_between_requests = random.uniform(3, 5)  # seconds
        self.batch_size = 10  # acts per batch
        self.break_time = 30  # seconds between batches
        
        # Track progress
        self.processed = set()
        self.load_progress()

    def load_progress(self):
        """Load previously processed acts"""
        progress_file = self.output_dir / 'progress.txt'
        if progress_file.exists():
            self.processed = set(progress_file.read_text().splitlines())
            logger.info(f"Loaded {len(self.processed)} previously processed acts")

    def save_progress(self, act_url):
        """Save progress after each successful scrape"""
        progress_file = self.output_dir / 'progress.txt'
        with open(progress_file, 'a') as f:
            f.write(f"{act_url}\n")
        self.processed.add(act_url)

    def get_acts_to_scrape(self):
        """Read CSV and filter out already processed acts"""
        acts = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['url'] not in self.processed:
                    acts.append(row)
        return acts

    def scrape_acts(self):
        acts = self.get_acts_to_scrape()
        total_acts = len(acts)
        logger.info(f"Found {total_acts} acts to scrape")

        for i, act in enumerate(acts, 1):
            try:
                # Check if we need a break between batches
                if i % self.batch_size == 0:
                    logger.info(f"Taking a {self.break_time}s break after batch...")
                    time.sleep(self.break_time)

                logger.info(f"Processing {i}/{total_acts}: {act['title']}")
                
                # Scrape the act
                result = scrape_zimlii_act(act['url'])
                if result:
                    sections, metadata = result
                    
                    # Generate filenames
                    act_slug = metadata["title"].replace(" ", "_").lower()
                    txt_file = f"{act_slug}.txt"
                    json_file = f"{act_slug}.json"
                    
                    # Save outputs
                    save_to_text(sections, metadata, txt_file)
                    save_to_json(sections, metadata, json_file)
                    
                    # Record progress
                    self.save_progress(act['url'])
                    
                    logger.info(f"[SUCCESS] Successfully processed {act['title']}")
                else:
                    logger.error(f"[ERROR] Failed to process {act['title']}")

                # Random delay between requests
                time.sleep(self.delay_between_requests)

            except Exception as e:
                logger.error(f"Error processing {act['title']}: {str(e)}")
                continue

        logger.info("Bulk scraping completed!")

def main():
    start_time = datetime.now()
    logger.info(f"Starting bulk scraping at {start_time}")
    
    scraper = BulkScraper()
    scraper.scrape_acts()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Completed scraping at {end_time} (Duration: {duration})")

if __name__ == "__main__":
    main()