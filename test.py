from sentence_transformers import SentenceTransformer, util
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd

# -------- CONFIGURATION --------
company_search_urls = {
    "Google": "https://www.linkedin.com/jobs/search/?f_C=1441&location=United%20States&f_TP=1",
    "Microsoft": "https://www.linkedin.com/jobs/search/?f_C=1035&location=United%20States&f_TP=1"
}

resume_keywords = [
    "machine learning", "deep learning", "computer vision", "pytorch", "tensorflow",
    "object detection", "classification", "docker", "aws", "llm", "nlp", "cnn",
    "transformer", "yolo", "fastapi", "flask", "sagemaker", "cloud computing",
    "model deployment", "data engineering"
]

MIN_MATCH_THRESHOLD = 70.0  # percent
MAX_YEARS_EXP = 6  # semantic match threshold

# -------- SETUP MODEL --------
model = SentenceTransformer("all-MiniLM-L6-v2")
resume_embedding = model.encode(" ".join(resume_keywords), convert_to_tensor=True)

# -------- SETUP DRIVER --------
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# -------- UTILITY FUNCTIONS --------
def scroll_to_bottom():
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def semantic_match_score(text):
    job_embedding = model.encode(text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_embedding, job_embedding)
    return round(float(score[0][0]) * 100, 2)

def is_senior_role(text):
    senior_text = f"Roles requiring more than {MAX_YEARS_EXP} years experience or senior titles"
    role_embedding = model.encode(senior_text, convert_to_tensor=True)
    job_embedding = model.encode(text, convert_to_tensor=True)
    similarity = float(util.pytorch_cos_sim(job_embedding, role_embedding)[0][0])
    return similarity > 0.5

def scrape_and_analyze_jobs():
    results = []
    for company, url in company_search_urls.items():
        print(f"\nScraping jobs for {company}...")
        driver.get(url)
        time.sleep(3)
        scroll_to_bottom()
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        job_cards = soup.select("a.base-card__full-link")

        seen = set()
        for job in job_cards:
            title = job.get('aria-label')
            link = job.get('href').split('?')[0]
            if link in seen:
                continue
            seen.add(link)

            # Visit job page to get description
            driver.get(link)
            time.sleep(2)
            job_soup = BeautifulSoup(driver.page_source, 'html.parser')
            desc_elem = job_soup.select_one("div.description__text")
            job_desc = desc_elem.get_text(separator=" ", strip=True) if desc_elem else ""

            match_score = semantic_match_score(job_desc)
            if match_score >= MIN_MATCH_THRESHOLD and not is_senior_role(job_desc):
                results.append({
                    "company": company,
                    "title": title,
                    "link": link,
                    "match_score": match_score
                })
    return results

# -------- RUN & DISPLAY --------
final_jobs = scrape_and_analyze_jobs()
df = pd.DataFrame(final_jobs)
print("\nTop Matching Jobs:")
print(df)
driver.quit()
