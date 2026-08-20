import os
import re
import time
import urllib.parse
import pandas as pd
import requests
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CAR_DATA_PATH = BASE_DIR / "data" / "Car Dealer list.xlsx"
OUTPUT_DIR = BASE_DIR / "static" / "images" / "cars"

# Headers for API requests
HEADERS = {
    "User-Agent": "CarRecommendationApp/2.0 (contact@carrecommendationsystem.com; developer test)"
}

# Standard segment fallbacks in case download fails completely
FALLBACKS = {
    "hatchback": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=600&q=80",
    "sedan": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=600&q=80",
    "suv": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=600&q=80",
    "muv": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=600&q=80",
    "ev": "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80",
    "luxury": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=600&q=80"
}

def get_filename(brand: str, model: str) -> str:
    """Helper to convert Brand and Model to a clean filename: brand_model.jpg"""
    name = f"{brand}_{model}".lower().strip()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_{2,}', '_', name)
    return f"{name}.jpg"

def fetch_wikipedia_image(brand: str, model: str) -> str | None:
    """Query Wikipedia for the exact model's image URL"""
    # Create search terms based on brand and model
    queries = [
        f"{brand} {model}",
        f"Suzuki {model}" if brand == "Maruti Suzuki" else f"{brand} {model}",
        f"Toyota {model}" if "Innova" in model else f"{brand} {model}",
        f"Wuling Air EV" if "Comet EV" in model else f"{brand} {model}",
        f"{model} (car)",
        model
    ]
    
    # De-duplicate queries list while maintaining order
    queries = list(dict.fromkeys(queries))
    
    for query in queries:
        try:
            # 1. Search Wikipedia for matching page title
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json"
            r = requests.get(search_url, headers=HEADERS, timeout=5).json()
            search_results = r.get("query", {}).get("search", [])
            
            if search_results:
                title = search_results[0]["title"]
                
                # 2. Get main image thumbnail from PageImages API
                img_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=pageimages&format=json&pithumbsize=800"
                res = requests.get(img_url, headers=HEADERS, timeout=5).json()
                pages = res.get("query", {}).get("pages", {})
                
                for page_id, page_data in pages.items():
                    if "thumbnail" in page_data:
                        src = page_data["thumbnail"]["source"]
                        # Check that it's not a generic svg or icon
                        if src.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            print(f"  Wiki Match: Found page '{title}' image")
                            return src
        except Exception:
            pass
    return None

def fetch_bing_image(query: str) -> str | None:
    """Scrape Bing Images for a backup car picture, filtering out logos and diagrams"""
    url = "https://www.bing.com/images/search?q=" + urllib.parse.quote(query)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        html = response.text
        
        matches = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', html)
        if not matches:
            matches = re.findall(r'"murl":"(http[^"]+)"', html)
            
        for match in matches:
            img_url = urllib.parse.unquote(match).replace("\\", "")
            # Filter out obvious non-car pages or small icons
            url_lower = img_url.lower()
            if any(term in url_lower for term in ["logo", "icon", "vector", "diagram", "structure", "formula", "sign", "emblem"]):
                continue
            return img_url
    except Exception:
        pass
    return None

def download_and_save(url: str, save_path: Path) -> bool:
    """Download file from URL and save to path"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            # Check content length is reasonable (> 2KB)
            if len(r.content) > 2048:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return True
    except Exception:
        pass
    return False

def main():
    print("Starting exact Car Image Downloader (Wiki + Bing)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CAR_DATA_PATH.exists():
        print(f"Car dataset not found at: {CAR_DATA_PATH}")
        return
        
    df = pd.read_excel(CAR_DATA_PATH)
    df = df.dropna(subset=["Brand", "Model"])
    df = df[df["Brand"].astype(str).str.strip() != ""]
    
    cars = df[["Brand", "Model", "Segment"]].drop_duplicates()
    print(f"Found {len(cars)} unique car models to download.")
    
    # Add custom BMW 3 Series fallback row for default list
    cars = pd.concat([
        cars, 
        pd.DataFrame([{"Brand": "BMW", "Model": "3 Series", "Segment": "Luxury"}])
    ], ignore_index=True).drop_duplicates(subset=["Brand", "Model"])
    
    success_count = 0
    
    for idx, row in cars.iterrows():
        brand = str(row["Brand"]).strip()
        model = str(row["Model"]).strip()
        segment = str(row["Segment"]).lower().strip()
        
        filename = get_filename(brand, model)
        save_path = OUTPUT_DIR / filename
        
        print(f"[{idx+1}/{len(cars)}] {brand} {model} ({segment})")
        
        # Step 1: Try Wikipedia API
        img_url = fetch_wikipedia_image(brand, model)
        downloaded = False
        
        if img_url:
            downloaded = download_and_save(img_url, save_path)
            if downloaded:
                print(f"  -> SUCCESS: Loaded from Wikipedia")
                success_count += 1
                
        # Step 2: Try Bing Image Search as backup
        if not downloaded:
            print("  Wiki match failed. Trying Bing search backup...")
            query = f"{brand} {model} car white color front view"
            img_url = fetch_bing_image(query)
            if img_url:
                downloaded = download_and_save(img_url, save_path)
                if downloaded:
                    print(f"  -> SUCCESS: Loaded from Bing backup")
                    success_count += 1
                    
        # Step 3: Segment Fallback
        if not downloaded:
            print("  All web queries failed. Loading segment fallback...")
            fallback_url = FALLBACKS.get(segment, FALLBACKS["suv"])
            downloaded = download_and_save(fallback_url, save_path)
            if downloaded:
                print(f"  -> FALLBACK: Loaded category default")
                
        # Small delay between downloads
        time.sleep(0.5)
        
    print(f"\nCompleted! Downloaded exact matches for {success_count} / {len(cars)} models.")

if __name__ == "__main__":
    main()
