"""
One-off scraper for PartSelect.

Pulls structured data + prose for ~40 parts into:
  data/parts.json           list of part records
  data/docs/{PS#}.txt       prose blob per part for RAG embedding

Source set:
  - PS11752778 (the named example: refrigerator door shelf bin)
  - All parts listed under model WDT780SAEM1 (dishwasher) — provides dishwasher coverage
  - A handful of seed fridge parts so the fridge category isn't underrepresented

Uses curl_cffi (Chrome TLS impersonation) because PartSelect sits behind Akamai
and rejects plain requests/curl. The site is server-rendered HTML, so no
headless browser is needed once we get past the TLS check.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from curl_cffi import requests

BASE = "https://www.partselect.com"
DATA_DIR = Path(__file__).parent / "data"
DOCS_DIR = DATA_DIR / "docs"

# Hard seeds: the explicitly named part, plus a few fridge parts so we cover
# both appliance categories in addition to the WDT780SAEM1 dishwasher pull.
SEED_PART_URLS = [
    "/PS11752778-Whirlpool-WPW10321304-Refrigerator-Door-Shelf-Bin.htm",
    "/PS733947-Whirlpool-WP2188656-Refrigerator-Crisper-Drawer-with-Humidity-Control.htm",
    "/PS11743427-Whirlpool-W10503278-Refrigerator-Ice-Maker-Assembly.htm",
    "/PS429725-Whirlpool-2255709-Refrigerator-Door-Shelf-Bin.htm",
]

SEED_MODEL_URLS = [
    "/Models/WDT780SAEM1/",   # dishwasher — task brief specifies this model
    "/Models/WRS325SDHZ05/",  # side-by-side refrigerator
    "/Models/KDPE234GBS/",    # KitchenAid dishwasher
    "/Models/WRF555SDFZ/",    # french-door refrigerator
]

MAX_PARTS = 40
SLEEP_SECONDS = 1.5


@dataclass
class Part:
    part_number: str               # PS-prefix
    manufacturer_part_number: str
    name: str
    appliance_type: str            # "Refrigerator" or "Dishwasher" (inferred)
    price: str | None
    in_stock: bool
    brand: str | None              # primary manufacturer
    brands_fits: list[str] = field(default_factory=list)
    description: str = ""
    symptoms: list[str] = field(default_factory=list)
    replaces_part_numbers: list[str] = field(default_factory=list)
    sample_models: list[str] = field(default_factory=list)   # subset of cross-ref (capped)
    install_difficulty: str | None = None
    install_video_id: str | None = None
    image_url: str | None = None
    source_url: str = ""
    repair_stories: list[str] = field(default_factory=list)  # short user-submitted notes


def fetch(url: str) -> str:
    """GET a partselect.com URL with Chrome impersonation. Returns HTML or raises."""
    full = url if url.startswith("http") else BASE + url
    r = requests.get(full, impersonate="chrome120", timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {full}")
    return r.text


# ---------------------------------------------------------------------------
# Model page → list of part URLs
# ---------------------------------------------------------------------------

def extract_part_urls_from_model(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for a in soup.select('a[href*="/PS"]'):
        href = a.get("href", "")
        m = re.match(r"^(/PS\d+-[^?#]+\.htm)", href)
        if not m:
            continue
        path = m.group(1)
        ps = re.match(r"^/(PS\d+)-", path).group(1)
        if ps in seen:
            continue
        seen.add(ps)
        urls.append(path)
    return urls


# ---------------------------------------------------------------------------
# Part page → Part record
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


def _section_column(soup: BeautifulSoup, label_pattern: str) -> Tag | None:
    """Find a .pd__wrap column whose bold header matches the pattern."""
    for div in soup.select(".pd__wrap.row .col-md-6"):
        header = div.select_one(".bold")
        if header and re.search(label_pattern, header.get_text(" ", strip=True), re.I):
            return div
    return None


def _section_text(soup: BeautifulSoup, label_pattern: str) -> str | None:
    """Return the text content of a labeled column, minus the header."""
    div = _section_column(soup, label_pattern)
    if not div:
        return None
    header = div.select_one(".bold")
    full = _clean(div.get_text(" ", strip=True))
    if header:
        head_text = _clean(header.get_text(" ", strip=True))
        if full.startswith(head_text):
            return full[len(head_text):].strip(": ").strip()
    return full


def _appliance_type_from_name(name: str, product_types: str | None) -> str:
    if product_types:
        if re.search(r"dishwasher", product_types, re.I):
            return "Dishwasher"
        if re.search(r"refrigerator|fridge|freezer", product_types, re.I):
            return "Refrigerator"
    if re.search(r"dishwasher", name, re.I):
        return "Dishwasher"
    if re.search(r"refrigerator|fridge|freezer", name, re.I):
        return "Refrigerator"
    return "Unknown"


def parse_part(html: str, source_url: str) -> Part | None:
    soup = BeautifulSoup(html, "html.parser")

    ps_el = soup.select_one('[itemprop=productID]')
    mpn_el = soup.select_one('[itemprop=mpn]')
    name_el = soup.find("h1")
    if not (ps_el and mpn_el and name_el):
        return None

    part_number = _clean(ps_el.get_text())
    mpn = _clean(mpn_el.get_text())
    name = _clean(name_el.get_text())

    price_el = soup.select_one(".js-partPrice") or soup.select_one("[itemprop=price]")
    price = _clean(price_el.get_text()) if price_el else None
    if price and not price.startswith("$"):
        price = "$" + price

    in_stock_el = soup.find(string=re.compile(r"In Stock|Out of Stock", re.I))
    in_stock = bool(in_stock_el and "In Stock" in in_stock_el)

    desc_el = soup.select_one("[itemprop=description]")
    description = _clean(desc_el.get_text(" ", strip=True)) if desc_el else ""

    # pd__wrap.row holds labeled columns; symptoms is a real <ul>
    symptoms: list[str] = []
    sym_col = _section_column(soup, r"fixes the following symptoms")
    if sym_col:
        symptoms = [_clean(li.get_text(" ", strip=True)) for li in sym_col.select("li")]

    products_text = _section_text(soup, r"works with the following products")
    replaces_text = _section_text(soup, rf"replaces these")

    replaces: list[str] = []
    if replaces_text:
        replaces = [_clean(p).rstrip(",") for p in re.split(r"[,\s]+", replaces_text) if _clean(p)]

    # Brand info via itemprop=brand
    brand = None
    brands_fits: list[str] = []
    brand_span = soup.select_one('[itemprop=brand] [itemprop=name]')
    if brand_span:
        brand = _clean(brand_span.get_text())
        # Sibling span lists "for Brand, Brand, ..."
        sib = brand_span.find_parent('[itemprop=brand]') if False else None
        for_span = soup.find(string=re.compile(r"^\s*for\s+\w", re.I))
        if for_span:
            ft = _clean(for_span)
            ft = re.sub(r"^for\s+", "", ft, flags=re.I)
            brands_fits = [b.strip() for b in re.split(r",|\band\b", ft) if b.strip()]

    # Model cross reference — sample first 10
    sample_models: list[str] = []
    for a in soup.select('a[href^="/Models/"]'):
        mm = re.match(r"^/Models/([^/]+)/?", a.get("href", ""))
        if mm:
            mdl = mm.group(1)
            if mdl not in sample_models:
                sample_models.append(mdl)
        if len(sample_models) >= 10:
            break

    # Install difficulty — sits as a text node next to a bold "Difficulty Level:" sibling
    install_difficulty = None
    for bold in soup.select("div.bold"):
        if "Difficulty Level" in bold.get_text():
            parent = bold.parent
            if parent:
                # Strip the header text from the parent's combined text
                full = _clean(parent.get_text(" ", strip=True))
                head = _clean(bold.get_text(" ", strip=True))
                if full.startswith(head):
                    rest = full[len(head):].strip(": ").strip()
                    if rest:
                        install_difficulty = rest
            break

    # Video — PartSelect uses .yt-video with a data attribute holding the YT id
    video_el = soup.select_one(".yt-video[data-yt-init], [data-yt-init]")
    install_video_id = video_el.get("data-yt-init") if video_el else None

    image_el = soup.select_one("img[itemprop=image]")
    image_url = image_el.get("src") if image_el else None

    # Repair stories (capped)
    repair_stories: list[str] = []
    for rs in soup.select(".repair-story")[:3]:
        txt = _clean(rs.get_text(" ", strip=True))
        if 30 < len(txt) < 600:
            repair_stories.append(txt)

    appliance_type = _appliance_type_from_name(name, products_text)

    return Part(
        part_number=part_number,
        manufacturer_part_number=mpn,
        name=name,
        appliance_type=appliance_type,
        price=price,
        in_stock=in_stock,
        brand=brand,
        brands_fits=brands_fits,
        description=description,
        symptoms=symptoms,
        replaces_part_numbers=replaces,
        sample_models=sample_models,
        install_difficulty=install_difficulty,
        install_video_id=install_video_id,
        image_url=image_url,
        source_url=BASE + source_url if source_url.startswith("/") else source_url,
        repair_stories=repair_stories,
    )


# ---------------------------------------------------------------------------
# Doc generation for RAG
# ---------------------------------------------------------------------------

def part_to_doc(p: Part) -> str:
    lines: list[str] = []
    lines.append(f"# {p.name}")
    lines.append(f"PartSelect Number: {p.part_number}")
    lines.append(f"Manufacturer Part Number: {p.manufacturer_part_number}")
    lines.append(f"Appliance Type: {p.appliance_type}")
    if p.brand:
        lines.append(f"Brand: {p.brand}")
    if p.brands_fits:
        lines.append(f"Fits Brands: {', '.join(p.brands_fits)}")
    if p.price:
        lines.append(f"Price: {p.price}")
    if p.install_difficulty:
        lines.append(f"Installation Difficulty: {p.install_difficulty}")
    lines.append("")
    if p.description:
        lines.append("## Description")
        lines.append(p.description)
        lines.append("")
    if p.symptoms:
        lines.append("## Symptoms This Part Fixes")
        for s in p.symptoms:
            lines.append(f"- {s}")
        lines.append("")
    if p.replaces_part_numbers:
        lines.append("## Replaces Part Numbers")
        lines.append(", ".join(p.replaces_part_numbers))
        lines.append("")
    if p.sample_models:
        lines.append("## Sample Compatible Models")
        lines.append(", ".join(p.sample_models))
        lines.append("")
    if p.repair_stories:
        lines.append("## Customer Repair Stories")
        for r in p.repair_stories:
            lines.append(f"- {r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_part_urls() -> list[str]:
    urls: list[str] = list(SEED_PART_URLS)
    for model_url in SEED_MODEL_URLS:
        print(f"[model] {model_url}", file=sys.stderr)
        html = fetch(model_url)
        urls.extend(extract_part_urls_from_model(html))
        time.sleep(SLEEP_SECONDS)
    # Dedupe by PS number, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        m = re.match(r"^/?(PS\d+)", u.lstrip("/"))
        if not m:
            continue
        ps = m.group(1)
        if ps in seen:
            continue
        seen.add(ps)
        out.append(u if u.startswith("/") else "/" + u)
        if len(out) >= MAX_PARTS:
            break
    return out


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    urls = collect_part_urls()
    print(f"[plan] {len(urls)} part URLs to fetch", file=sys.stderr)

    parts: list[Part] = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}", file=sys.stderr)
        try:
            html = fetch(url)
            part = parse_part(html, url)
            if part is None:
                print(f"  ! could not parse, skipping", file=sys.stderr)
                continue
            parts.append(part)
            doc = part_to_doc(part)
            (DOCS_DIR / f"{part.part_number}.txt").write_text(doc, encoding="utf-8")
        except Exception as e:
            print(f"  ! error: {e}", file=sys.stderr)
        time.sleep(SLEEP_SECONDS)

    out_path = DATA_DIR / "parts.json"
    out_path.write_text(
        json.dumps([asdict(p) for p in parts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] wrote {len(parts)} parts to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
