import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
from tqdm import tqdm

BASE_URL = "https://www.unegui.mn"
CATEGORY_PATH = "/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/2-r/?page="

OUTPUT_CSV = "unegui_data.csv"
OUTPUT_JSON = "unegui_data.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}

UB_DISTRICTS = [
    "Баянзүрх", "Сүхбаатар", "Чингэлтэй",
    "Баянгол", "Хан-Уул", "Сонгинохайрхан",
    "Налайх", "Багануур", "Багахангай"
]

def clean_text(el):
    if el:
        return el.get_text(strip=True).replace("\xa0", " ").strip()
    return ""

def parse_price(price_str):
    if not price_str:
        return None
    price_str = price_str.replace(",", "").replace("₮", "").strip()
    match = re.search(r"(\d+(\.\d+)?)", price_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def parse_area(area_str):
    if not area_str:
        return None
    s = area_str.lower()
    s = s.replace("мкв", "м2").replace("м.кв", "м2").replace("м²", "м2")
    match = re.search(r"(\d+(\.\d+)?)(?=\s*м2|\s*$)", s)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def extract_district(address):
    if not address:
        return None
    for dist in UB_DISTRICTS:
        if dist in address:
            return dist
    return None

def scrape_listing(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ Зарын хуудас авахад алдаа: {url} (код {res.status_code})")
            return None
        soup = BeautifulSoup(res.text, "html.parser")

        props = {}

        for row in soup.select(".price-title+ .property"):
            k = clean_text(row.select_one(".name"))
            v = clean_text(row.select_one(".value"))
            if k:
                props[k] = v

        if not props:
            for tr in soup.select("table tr"):
                tds = tr.find_all(["td", "th"])
                if len(tds) >= 2:
                    k = clean_text(tds[0]).replace(":", "")
                    v = clean_text(tds[1])
                    if k:
                        props[k] = v

            for li in soup.select("ul li"):
                txt = clean_text(li)
                if ":" in txt:
                    parts = txt.split(":", 1)
                    k = parts[0].strip()
                    v = parts[1].strip()
                    props[k] = v

        price_text = clean_text(soup.select_one(".price"))
        if not price_text:
            alt_price = soup.select_one(".announcement-price")
            if alt_price:
                price_text = clean_text(alt_price)
        props["price_text"] = price_text
        props["price_numeric"] = parse_price(price_text)

        area_text = props.get("Талбай", "")
        props["area_numeric"] = parse_area(area_text)

        address_tag = soup.find("span", itemprop="address")
        props["address_text"] = clean_text(address_tag) if address_tag else None
        props["district"] = extract_district(props["address_text"])

        props["url"] = url
        return props
    except Exception as e:
        print(f"⚠️ Алдаа (scrape_listing): {e}")
        return None

def scrape_all_pages(pages=20, limit=None):
    all_data = []
    count = 0
    seen_urls = set()  # Давтагдсан URL-г хадгалах

    for page in range(1, pages + 1):
        url = f"{BASE_URL}{CATEGORY_PATH}{page}"
        print(f"📄 Хуудас {page}: {url}")
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ Хуудас авахад алдаа гарлаа: Код {res.status_code}")
            continue
        soup = BeautifulSoup(res.text, "html.parser")

        links = [a['href'] for a in soup.find_all('a', href=True) if "/adv/" in a['href']]
        links = list(dict.fromkeys(links))  # давхардлыг арилгах

        for link in tqdm(links, desc=f"Page {page}"):
            if limit is not None and count >= limit:
                return all_data

            full_url = link if link.startswith("http") else BASE_URL + link
            if full_url in seen_urls:
                continue  # Давтагдсан бол алгасах

            data = scrape_listing(full_url)
            if data:
                all_data.append(data)
                seen_urls.add(full_url)
                count += 1
            time.sleep(0.5)
    return all_data

def main():
    all_data = scrape_all_pages(pages=20, limit=None)  # 20 хуудас бүх зар
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"✅ Амжилттай {len(df)} зар хадгалагдлаа: {OUTPUT_CSV}, {OUTPUT_JSON}")
    else:
        print("⚠️ Мэдээлэл олдсонгүй.")

if __name__ == "__main__":
    main()
