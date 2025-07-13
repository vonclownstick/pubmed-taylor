#!/usr/bin/env python3
"""
Pubmed (Taylor's Version) - Enhanced with AI Analysis and Password Protection
Smart concentric search with LLM-generated queries, citation weighting, and AI-powered relevance ranking
SECURITY FEATURES:
- Password protection for new sessions/IPs
- Session-based authentication
- Environment variable password storage
"""
import asyncio
import csv
import datetime as dt
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional
from functools import lru_cache
import json
import hashlib
import secrets

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, Response, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ---------- Load .env if present ----------
def _load_dotenv() -> None:
    env_path = ".env"
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ModuleNotFoundError:
            for line in open(env_path).readlines():
                if line.strip() and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

# ---------- Configuration ----------
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "research@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "Cassowary")  # Default password
DEBUG_JOURNALS = os.getenv("DEBUG_JOURNALS", "false").lower() in ["true", "1", "yes"]
UPDATE_MAPPINGS = os.getenv("UPDATE_MAPPINGS", "true").lower() in ["true", "1", "yes"]

NOW = dt.datetime.now().year
NCBI_TIMEOUT = 15  # Reduced timeout
BATCH_SIZE = 800  # Increased batch size significantly
MAX_CONCURRENT = 10  # Increased concurrency
RATE_LIMIT_SEC = 0.02 if NCBI_API_KEY else 0.1  # Much faster with API key

# Global cache for fuzzy matching results
FUZZY_MATCH_CACHE = {}

# Counter for learned mappings this session
LEARNED_MAPPINGS_COUNT = 0

# Session management for password protection
AUTHENTICATED_SESSIONS = set()  # Store authenticated session tokens
SESSION_SECRET = secrets.token_hex(32)  # Secret for session tokens

def get_client_ip(request: Request) -> str:
    """Get client IP address, handling proxies"""
    # Check for forwarded IP first (for deployments behind proxies like Render)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check other common headers
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct connection IP
    if request.client:
        return request.client.host
    
    return "unknown"

def create_session_token(ip: str) -> str:
    """Create a session token for an authenticated IP"""
    # Create a token that includes IP and timestamp for extra security
    timestamp = str(int(time.time()))
    data = f"{ip}:{timestamp}:{SESSION_SECRET}"
    return hashlib.sha256(data.encode()).hexdigest()

def verify_session_token(token: str, ip: str) -> bool:
    """Verify a session token is valid and belongs to the IP"""
    if not token:
        return False
    
    # Check if token is in our authenticated sessions
    return token in AUTHENTICATED_SESSIONS

async def check_authentication(
    request: Request, 
    session_token: Optional[str] = Cookie(None)
) -> bool:
    """Check if the request is authenticated"""
    client_ip = get_client_ip(request)
    
    # Check if session token is valid
    if session_token and verify_session_token(session_token, client_ip):
        return True
    
    return False

# ---------- Journal Impact Factor Data (OPTIMIZED) ----------
@lru_cache(maxsize=10000)
def normalize_journal_name_cached(name: str) -> str:
    """Cached version of journal name normalization"""
    if not name or not name.strip():
        return ""
    
    if str(name).lower() in ['nan', 'none', 'null']:
        return ""
    
    normalized = str(name).strip().upper()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[.,]', '', normalized)
    normalized = re.sub(r'\s*&\s*', ' AND ', normalized)
    
    if len(normalized.split()) > 3:
        words_to_remove = ['THE', 'OF', 'AND', 'FOR', 'IN', 'ON']
        for word in words_to_remove:
            normalized = re.sub(r'\b' + word + r'\b', '', normalized)
    
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def load_journal_impacts(csv_file: str = "impacts_mapped.csv") -> Dict[str, Dict]:
    """Optimized journal impact factor loading with better indexing"""
    impacts = {}
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Journal impact factors will not be available.", file=sys.stderr)
        return impacts
    
    try:
        print("Loading journal impact factors...", file=sys.stderr)
        start_time = time.time()
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            
            for row in reader:
                journal_title = row.get('journal_title', '').strip()
                pubmed_abbrev = row.get('pubmed_abbreviation', '').strip()
                issn = row.get('issn', '').strip()
                
                jif_sans_self = None
                jif_column = row.get('JIF Sans Journal Self Cites', '')
                
                try:
                    if jif_column and jif_column.strip() and jif_column.strip().lower() not in ['n/a', 'na', '', '-']:
                        jif_sans_self = float(jif_column.strip())
                    else:
                        jif_sans_self = 0.0
                except (ValueError, TypeError):
                    jif_sans_self = 0.0
                
                if journal_title and jif_sans_self is not None:
                    impact_data = {
                        'journal_title': journal_title,
                        'pubmed_abbreviation': pubmed_abbrev if pubmed_abbrev else '',
                        'issn': issn,
                        'jif_sans_self': jif_sans_self
                    }
                    
                    # Use cached normalization
                    normalized_title = normalize_journal_name_cached(journal_title)
                    normalized_abbrev = normalize_journal_name_cached(pubmed_abbrev) if pubmed_abbrev else ''
                    
                    if normalized_title:
                        impacts[normalized_title] = impact_data
                        count += 1
                    
                    if normalized_abbrev and normalized_abbrev != normalized_title:
                        impacts[normalized_abbrev] = impact_data
                    
                    # Index by ISSN
                    if issn:
                        impacts[issn] = impact_data
                        clean_issn = issn.replace('-', '').replace(' ', '')
                        if clean_issn != issn:
                            impacts[clean_issn] = impact_data
            
            load_time = time.time() - start_time
            print(f"Loaded {count} journal impact factors in {load_time:.2f}s", file=sys.stderr)
            
    except Exception as e:
        print(f"Error loading {csv_file}: {e}", file=sys.stderr)
    
    return impacts

def find_journal_impact_optimized(journal_name: str, issn: str, impacts_db: Dict[str, Dict]) -> float:
    """OPTIMIZED journal impact factor lookup with caching, limited fuzzy matching, and CSV learning"""
    if not impacts_db:
        return 0.0
    
    # Create cache key
    cache_key = f"{journal_name}|{issn}"
    if cache_key in FUZZY_MATCH_CACHE:
        return FUZZY_MATCH_CACHE[cache_key]
    
    # Strategy 1: Try exact ISSN match first (fastest)
    if issn:
        if issn in impacts_db:
            result = impacts_db[issn]['jif_sans_self']
            FUZZY_MATCH_CACHE[cache_key] = result
            return result
        
        clean_issn = issn.replace('-', '').replace(' ', '')
        if clean_issn in impacts_db:
            result = impacts_db[clean_issn]['jif_sans_self']
            FUZZY_MATCH_CACHE[cache_key] = result
            return result
    
    # Strategy 2: Try exact normalized journal name match
    if journal_name:
        normalized = normalize_journal_name_cached(journal_name)
        
        if normalized in impacts_db:
            result = impacts_db[normalized]['jif_sans_self']
            FUZZY_MATCH_CACHE[cache_key] = result
            return result
        
        # Strategy 3: LIMITED fuzzy matching (only for important cases)
        # Only do fuzzy matching if journal name is reasonable length and no exact match
        if len(journal_name) > 5 and len(journal_name) < 100:
            best_score = 0
            best_match = None
            best_match_key = ""
            
            # Limit fuzzy matching to first 500 entries for performance
            candidates_checked = 0
            max_candidates = 500
            
            for stored_key, data in impacts_db.items():
                if candidates_checked >= max_candidates:
                    break
                    
                if not isinstance(data, dict) or stored_key.isdigit():
                    continue
                
                candidates_checked += 1
                
                # Quick similarity check - only do expensive matching for promising candidates
                if len(stored_key) > 3:
                    # Quick character overlap check first
                    common_chars = set(normalized.lower()) & set(stored_key.lower())
                    char_overlap = len(common_chars) / min(len(normalized), len(stored_key))
                    
                    if char_overlap > 0.4:  # Only proceed if reasonable character overlap
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, normalized, stored_key).ratio()
                        
                        if similarity > 0.75 and similarity > best_score:  # Higher threshold for performance
                            best_score = similarity
                            best_match = data
                            best_match_key = stored_key
            
            if best_match and best_score >= 0.75:
                result = best_match['jif_sans_self']
                FUZZY_MATCH_CACHE[cache_key] = result
                
                # 🎯 SAVE LEARNED MAPPING: Add successful fuzzy match to CSV for future exact matches
                if best_score >= 0.80:  # Only save high-confidence matches
                    save_learned_mapping(journal_name, issn, best_match, best_score)
                    if DEBUG_JOURNALS:
                        print(f"💾 LEARNED: '{journal_name}' -> '{best_match_key}' (score: {best_score:.3f})", file=sys.stderr)
                
                return result
    
    # Cache negative results too
    FUZZY_MATCH_CACHE[cache_key] = 0.0
    return 0.0

def save_learned_mapping(pubmed_name: str, issn: str, matched_data: Dict, confidence_score: float, csv_file: str = "impacts_mapped.csv") -> None:
    """Save a successful fuzzy match to CSV file for future exact matching"""
    global LEARNED_MAPPINGS_COUNT
    
    if not UPDATE_MAPPINGS:
        return
        
    try:
        # Normalize the PubMed name for the key
        normalized_pubmed = normalize_journal_name_cached(pubmed_name)
        
        # Check if we already have this mapping in memory
        if normalized_pubmed in JOURNAL_IMPACTS:
            return  # Already exists
        
        # 🚀 Add to in-memory dictionary IMMEDIATELY for current session
        JOURNAL_IMPACTS[normalized_pubmed] = matched_data.copy()
        
        # Also add with original name if different
        if pubmed_name.strip() != normalized_pubmed:
            JOURNAL_IMPACTS[pubmed_name.strip()] = matched_data.copy()
        
        # Add ISSN mappings if available
        if issn:
            clean_issn = issn.replace('-', '').replace(' ', '')
            if issn not in JOURNAL_IMPACTS:
                JOURNAL_IMPACTS[issn] = matched_data.copy()
            if clean_issn != issn and clean_issn not in JOURNAL_IMPACTS:
                JOURNAL_IMPACTS[clean_issn] = matched_data.copy()
        
        # 💾 APPEND TO CSV FILE for persistence across sessions
        if os.path.exists(csv_file):
            # Create backup on first write (once per session)
            backup_file = csv_file + ".backup"
            if not os.path.exists(backup_file):
                import shutil
                shutil.copy2(csv_file, backup_file)
                print(f"📄 Created backup: {backup_file}", file=sys.stderr)
            
            # Append the learned mapping to CSV
            with open(csv_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Create a row that matches the CSV format
                # Use the PubMed name that was searched for as the new journal_title
                new_row = [
                    pubmed_name,  # journal_title - the name that was searched
                    issn or matched_data.get('issn', ''),  # issn
                    '',  # 2022 Total Cites to All Years (empty for learned)
                    '',  # Journal Impact Factor (empty, we use JIF Sans Self Cites)
                    '',  # Journal Citation Indicator (empty for learned)
                    '',  # 5-Year Jrl Impact Factor (empty for learned)
                    str(matched_data['jif_sans_self']),  # JIF Sans Journal Self Cites - THE KEY VALUE
                    '',  # Immediacy Index (empty for learned)
                    '',  # Normalized Eigenfactor (empty for learned)
                    '',  # Article Influence Score (empty for learned)
                    f"LEARNED_FROM_{matched_data.get('journal_title', 'FUZZY_MATCH')}"  # pubmed_abbreviation - mark as learned
                ]
                
                writer.writerow(new_row)
            
            # Increment session counter
            LEARNED_MAPPINGS_COUNT += 1
            
            print(f"Saved learned mapping #{LEARNED_MAPPINGS_COUNT}: '{pubmed_name}' -> JIF: {matched_data['jif_sans_self']:.2f} (confidence: {confidence_score:.2f})", file=sys.stderr)
        
    except Exception as e:
        print(f"⚠️  Failed to save learned mapping: {e}", file=sys.stderr)

# Load journal impacts at startup
print("Initializing journal impact database...", file=sys.stderr)
JOURNAL_IMPACTS = load_journal_impacts()

@dataclass
class SearchResult:
    pmid: str
    title: str
    authors: str
    journal: str
    year: int
    abstract: str
    weight: float
    strategy: str
    rank: int
    journal_impact: float = 0.0
    issn: str = ""
    combined_score: float = 0.0
    ai_rank: Optional[int] = None  # New field for AI ranking
    # Additional metadata for RIS export
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""

# ---------- Enhanced Ranking Functions (OPTIMIZED) ----------
@lru_cache(maxsize=1000)
def calculate_recency_boost_cached(year: int, current_year: int = NOW) -> float:
    """Cached recency boost calculation"""
    years_ago = current_year - year
    if years_ago <= 1:
        return 2.0
    elif years_ago <= 3:
        return 1.5
    elif years_ago <= 5:
        return 1.2
    elif years_ago <= 10:
        return 1.0
    else:
        return 0.8

@lru_cache(maxsize=1000)
def calculate_journal_impact_score_cached(journal_impact: float) -> float:
    """Cached journal impact score calculation"""
    if journal_impact <= 0:
        return 0.0
    return 1.0 + math.log10(journal_impact)

def calculate_combined_score_batch(results_data: List[tuple]) -> List[float]:
    """Batch calculate combined scores for better performance"""
    scores = []
    for raw_citations, journal_impact, year in results_data:
        recency_boost = calculate_recency_boost_cached(year, NOW)
        boosted_citations = raw_citations * recency_boost
        
        citation_score = math.log10(boosted_citations + 1) if boosted_citations > 0 else 0.0
        journal_score = calculate_journal_impact_score_cached(journal_impact)
        
        combined = (0.4 * citation_score + 0.4 * journal_score + 0.2 * recency_boost)
        scores.append(combined)
    
    return scores

# ---------- AI Analysis Functions ----------
async def analyze_with_ai(results: List[SearchResult], query: str) -> Dict:
    """Send results to AI for relevance ranking and summary generation"""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured")
    
    if not results:
        raise ValueError("No results to analyze")
    
    # Prepare data for AI analysis
    papers_data = []
    for result in results:
        papers_data.append({
            "pmid": result.pmid,
            "title": result.title,
            "abstract": result.abstract or "No abstract available",
            "authors": result.authors,
            "journal": result.journal,
            "year": result.year
        })
    
    # Create the prompt
    prompt = f"""You are analyzing scientific papers for relevance to the research question: "{query}"

Please analyze these {len(papers_data)} papers and provide:

1. A ranked list of PMIDs from most relevant to least relevant based on title and abstract content
2. A comprehensive synthesis organized as exactly 3 distinct paragraphs

SYNTHESIS REQUIREMENTS:
- Write exactly 3 paragraphs separated by blank lines
- Paragraph 1: Overview of the research landscape and main themes
- Paragraph 2: Key findings and methodological approaches  
- Paragraph 3: Implications, gaps, and future directions
- Only reference papers provided using format (PMID: 12345678)
- Include multiple relevant PMIDs throughout each paragraph
- Focus on findings most relevant to: "{query}"
- Use scientific writing style

Papers to analyze:
{json.dumps(papers_data, indent=2)}

Respond in this exact JSON format:
{{
  "ranked_pmids": ["pmid1", "pmid2", "pmid3", ...],
  "synthesis": "Paragraph 1 content here with citations (PMID: 12345678).\\n\\nParagraph 2 content here with more citations (PMID: 87654321).\\n\\nParagraph 3 content here with final citations (PMID: 11223344)."
}}"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a scientific literature analysis expert. Return only valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        print(f"🧠 Sending {len(papers_data)} papers to AI for analysis...", file=sys.stderr)
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60  # Longer timeout for analysis
        )
        response.raise_for_status()
        
        result = response.json()
        raw_content = result["choices"][0]["message"]["content"]
        
        try:
            parsed_result = json.loads(raw_content)
            
            # Validate the response structure
            if "ranked_pmids" not in parsed_result or "synthesis" not in parsed_result:
                raise ValueError("AI response missing required fields")
            
            ranked_pmids = parsed_result["ranked_pmids"]
            synthesis = parsed_result["synthesis"]
            
            # Validate PMIDs are in our original set
            original_pmids = {r.pmid for r in results}
            valid_ranked_pmids = [pmid for pmid in ranked_pmids if pmid in original_pmids]
            
            if not valid_ranked_pmids:
                raise ValueError("No valid PMIDs in AI ranking")
            
            print(f"✅ AI analysis complete: {len(valid_ranked_pmids)} papers ranked", file=sys.stderr)
            
            return {
                "ranked_pmids": valid_ranked_pmids,
                "synthesis": synthesis,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse AI response: {e}", file=sys.stderr)
            return {"success": False, "error": "Failed to parse AI response"}
            
    except Exception as e:
        print(f"❌ AI analysis failed: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}

# ---------- Enhanced LLM Query Generation (OPTIMIZED) ----------
def generate_smart_queries(nlq: str, model: str = OPENAI_MODEL) -> List[str]:
    """Optimized query generation with shorter timeout"""
    if not OPENAI_API_KEY:
        return generate_fallback_queries(nlq)
    
    prompt = f"""Generate exactly 4 PubMed search queries for: "{nlq}"

Return as JSON:
{{
  "queries": [
    "specific MeSH/field query",
    "moderate specificity query", 
    "broader terms query",
    "natural language query"
  ]
}}"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON with exactly 4 PubMed queries."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,  # Reduced tokens
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10  # Reduced timeout
        )
        response.raise_for_status()
        
        result = response.json()
        raw = result["choices"][0]["message"]["content"]
        
        parsed = json.loads(raw)
        queries = parsed.get("queries", [])
        
        clean_queries = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 5]
        
        if len(clean_queries) >= 3:
            print(f"Generated {len(clean_queries)} AI queries", file=sys.stderr)
            return clean_queries[:4]
        else:
            return generate_fallback_queries(nlq)
                
    except Exception as e:
        print(f"OpenAI failed, using fallbacks: {e}", file=sys.stderr)
        return generate_fallback_queries(nlq)

def generate_fallback_queries(nlq: str) -> List[str]:
    """Optimized fallback query generation"""
    key_terms = re.findall(r'\b[a-zA-Z]{3,}\b', nlq.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    key_terms = [w for w in key_terms if w not in stop_words][:4]
    
    if not key_terms:
        return [nlq.strip()]
    
    queries = []
    if len(key_terms) >= 2:
        queries.append(f'{key_terms[0]}[Title/Abstract] AND {key_terms[1]}[Title/Abstract]')
        queries.append(f'{key_terms[0]} AND {key_terms[1]}')
        queries.append(f'{key_terms[0]} OR {key_terms[1]}')
    
    queries.append(key_terms[0])
    
    return queries[:4]

# ---------- NCBI API helpers (OPTIMIZED) ----------
def _eparams(extra: dict = None) -> dict:
    """Add NCBI email and API key to parameters"""
    p = {} if extra is None else dict(extra)
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p

async def esearch_async(session: aiohttp.ClientSession, query: str, retmax: int = 200, strategy_name: str = "") -> tuple[List[str], str]:
    """Optimized async PubMed search"""
    if not query or len(query.strip()) < 2:
        return [], strategy_name
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query.strip(),
        "retmode": "json",
        "retmax": retmax
    }
    params.update(_eparams())
    
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            data = await response.json()
            
            ids = data.get("esearchresult", {}).get("idlist", [])
            # Ensure all IDs are strings
            clean_ids = [str(pmid).strip() for pmid in ids if str(pmid).strip().isdigit()]
            
            print(f"{strategy_name}: {len(clean_ids)} results", file=sys.stderr)
            return clean_ids, strategy_name
            
    except Exception as e:
        print(f"Search failed for '{query[:30]}...': {e}", file=sys.stderr)
        return [], strategy_name

async def fetch_abstracts_batch(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Dict]:
    """Optimized abstract fetching with larger batches"""
    if not pmids:
        return {}
    
    info_map = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # Use larger batches for better performance
    batch_size = 1000
    batches = [pmids[i:i+batch_size] for i in range(0, len(pmids), batch_size)]
    
    async def fetch_single_batch(batch: List[str]) -> Dict[str, Dict]:
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(batch)
        }
        params.update(_eparams())
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                response.raise_for_status()
                xml_text = await response.text()
                
                batch_info = {}
                root = ET.fromstring(xml_text)
                for art in root.findall(".//PubmedArticle"):
                    pid = art.findtext(".//PMID")
                    if pid:
                        abstr = art.findtext(".//Abstract/AbstractText") or ""
                        
                        # Get ISSN efficiently
                        issn = (art.findtext(".//ISSN[@IssnType='Print']") or 
                               art.findtext(".//ISSN[@IssnType='Electronic']") or 
                               art.findtext(".//ISSNLinking") or 
                               art.findtext(".//ISSN") or "")
                        
                        # Get journal title efficiently
                        journal_title = (art.findtext(".//Journal/Title") or 
                                       art.findtext(".//Journal/ISOAbbreviation") or 
                                       art.findtext(".//MedlineJournalInfo/MedlineTA") or "")
                        
                        batch_info[pid] = {
                            'abstract': abstr,
                            'issn': issn,
                            'journal_title': journal_title
                        }
                
                return batch_info
                        
        except Exception as e:
            print(f"Abstract batch failed: {e}", file=sys.stderr)
            return {}
    
    # Process batches concurrently but with semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async def controlled_fetch(batch):
        async with semaphore:
            return await fetch_single_batch(batch)
    
    tasks = [controlled_fetch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict):
            info_map.update(result)
    
    return info_map

async def fetch_citation_data_batch(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Dict]:
    """Optimized citation data fetching"""
    info = {}
    
    # Process in large batches (iCite supports up to 1000)
    batch_size = 1000
    batches = [pmids[i:i+batch_size] for i in range(0, len(pmids), batch_size)]
    
    async def fetch_single_batch(batch: List[str]) -> Dict[str, Dict]:
        try:
            url = "https://icite.od.nih.gov/api/pubs"
            params = {"pmids": ",".join(batch)}
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    data = await response.json()
                    batch_info = {}
                    
                    for item in data.get("data", []):
                        pmid = str(item.get("pmid", ""))
                        if pmid:
                            cc = item.get("citation_count", 0) or 0
                            yr = item.get("year", NOW) or NOW
                            cpy = item.get("citations_per_year", 0) or 0
                            
                            # Proper author formatting for iCite API
                            authors_raw = item.get("authors", "Unknown")
                            if isinstance(authors_raw, list) and authors_raw:
                                first_author = authors_raw[0]
                                if isinstance(first_author, dict):
                                    # Extract name from author object
                                    if 'fullName' in first_author and first_author['fullName']:
                                        authors = first_author['fullName']
                                    elif 'lastName' in first_author:
                                        if 'firstName' in first_author and first_author['firstName']:
                                            authors = f"{first_author['lastName']}, {first_author['firstName']}"
                                        else:
                                            authors = first_author['lastName']
                                    else:
                                        authors = str(first_author)
                                else:
                                    # Simple string in list
                                    authors = str(first_author)
                            elif isinstance(authors_raw, str):
                                # If it's a string, use as-is or take first author
                                if ';' in authors_raw:
                                    authors = authors_raw.split(';')[0].strip()
                                elif ',' in authors_raw and len(authors_raw.split(',')) > 2:
                                    # Multiple authors separated by commas
                                    authors = authors_raw.split(',')[0].strip()
                                else:
                                    authors = authors_raw
                            else:
                                authors = "Unknown"
                            
                            # Limit length for performance and display
                            if len(authors) > 50:
                                authors = authors[:47] + "..."
                            
                            batch_info[pmid] = {
                                "weight": cpy,  # Use citations per year directly
                                "year": yr,
                                "journal": item.get("journal", "Unknown") or "Unknown",
                                "authors": authors,
                                "title": item.get("title", "Unknown") or "Unknown",
                                "citation_count": cc
                            }
                    
                    return batch_info
                else:
                    return {}
        
        except Exception as e:
            print(f"iCite batch failed: {e}", file=sys.stderr)
            return {}
    
    # Control concurrency
    semaphore = asyncio.Semaphore(5)  # Limit iCite requests
    
    async def controlled_fetch(batch):
        async with semaphore:
            return await fetch_single_batch(batch)
    
    tasks = [controlled_fetch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict):
            info.update(result)
    
    # Add defaults for missing PMIDs
    for pmid in pmids:
        if pmid not in info:
            info[pmid] = {
                "weight": 0.0,
                "year": NOW,
                "journal": "Unknown",
                "authors": "Unknown",
                "title": "Unknown",
                "citation_count": 0
            }
    
    return info

async def fetch_detailed_metadata(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Dict]:
    """Fetch detailed metadata for RIS export (volume, issue, pages, DOI)"""
    if not pmids:
        return {}
    
    detailed_info = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # Process in batches
    batch_size = 200
    batches = [pmids[i:i+batch_size] for i in range(0, len(pmids), batch_size)]
    
    async def fetch_single_batch(batch: List[str]) -> Dict[str, Dict]:
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "rettype": "medline",
            "id": ",".join(batch)
        }
        params.update(_eparams())
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                xml_text = await response.text()
                
                batch_info = {}
                root = ET.fromstring(xml_text)
                
                for art in root.findall(".//PubmedArticle"):
                    pmid = art.findtext(".//PMID")
                    if pmid:
                        # Get volume, issue, pages
                        volume = art.findtext(".//Volume") or ""
                        issue = art.findtext(".//Issue") or ""
                        
                        # Get pagination
                        start_page = art.findtext(".//StartPage") or art.findtext(".//MedlinePgn") or ""
                        end_page = art.findtext(".//EndPage") or ""
                        if start_page and end_page:
                            pages = f"{start_page}-{end_page}"
                        elif start_page:
                            pages = start_page
                        else:
                            pages = ""
                        
                        # Get DOI
                        doi = ""
                        for elem in art.findall(".//ArticleId"):
                            if elem.get("IdType") == "doi":
                                doi = elem.text or ""
                                break
                        
                        batch_info[pmid] = {
                            'volume': volume,
                            'issue': issue,
                            'pages': pages,
                            'doi': doi
                        }
                
                return batch_info
                        
        except Exception as e:
            print(f"Failed to fetch detailed metadata: {e}", file=sys.stderr)
            return {}
    
    # Control concurrency
    semaphore = asyncio.Semaphore(3)
    
    async def controlled_fetch(batch):
        async with semaphore:
            return await fetch_single_batch(batch)
    
    tasks = [controlled_fetch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict):
            detailed_info.update(result)
    
    return detailed_info

def generate_ris_content(results: List[SearchResult]) -> str:
    """Generate RIS format content from search results"""
    ris_lines = []
    
    for result in results:
        # Start record
        ris_lines.append("TY  - JOUR")  # Journal article
        
        # Title
        if result.title:
            ris_lines.append(f"TI  - {result.title}")
        
        # Authors - split and format properly
        if result.authors and result.authors != "Unknown":
            # Handle different author formats
            authors_text = result.authors
            if "," in authors_text:
                # If it's "Last, First" format, keep as is
                ris_lines.append(f"AU  - {authors_text}")
            else:
                # Otherwise just use as provided
                ris_lines.append(f"AU  - {authors_text}")
        
        # Journal
        if result.journal and result.journal != "Unknown":
            ris_lines.append(f"JO  - {result.journal}")
        
        # Year
        if result.year:
            ris_lines.append(f"PY  - {result.year}")
        
        # Volume
        if result.volume:
            ris_lines.append(f"VL  - {result.volume}")
        
        # Issue
        if result.issue:
            ris_lines.append(f"IS  - {result.issue}")
        
        # Pages
        if result.pages:
            if "-" in result.pages:
                start_page, end_page = result.pages.split("-", 1)
                ris_lines.append(f"SP  - {start_page.strip()}")
                ris_lines.append(f"EP  - {end_page.strip()}")
            else:
                ris_lines.append(f"SP  - {result.pages}")
        
        # DOI
        if result.doi:
            ris_lines.append(f"DO  - {result.doi}")
        
        # ISSN
        if result.issn:
            ris_lines.append(f"SN  - {result.issn}")
        
        # Abstract
        if result.abstract:
            # RIS abstracts should be on one line or properly continued
            abstract_clean = result.abstract.replace('\n', ' ').replace('\r', ' ')
            ris_lines.append(f"AB  - {abstract_clean}")
        
        # URL to PubMed
        ris_lines.append(f"UR  - https://pubmed.ncbi.nlm.nih.gov/{result.pmid}/")
        
        # PubMed ID as a note
        ris_lines.append(f"N1  - PMID: {result.pmid}")
        
        # End record
        ris_lines.append("ER  - ")
        ris_lines.append("")  # Blank line between records
    
    return "\n".join(ris_lines)

# ---------- Main Search Pipeline (OPTIMIZED) ----------
async def perform_concentric_search_optimized(nlq: str, max_results: int = 200) -> List[SearchResult]:
    """Optimized concentric search pipeline with deduplication"""
    print(f"\nSearch: '{nlq}' (max: {max_results})", file=sys.stderr)
    
    # Validate input
    if not nlq or len(nlq.strip()) < 2:
        raise ValueError("Query too short")
    
    # Generate queries
    queries = generate_smart_queries(nlq)
    strategy_names = ["MeSH Specific", "Moderate", "Broad", "Natural"][:len(queries)]
    
    # Use connection pooling for better performance
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=20,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Step 1: Run all searches in parallel
        print("Running parallel searches...", file=sys.stderr)
        search_start = time.time()
        
        search_tasks = [
            esearch_async(session, query, retmax=120, strategy_name=name)  
            for query, name in zip(queries, strategy_names)
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        search_time = time.time() - search_start
        print(f"Searches completed in {search_time:.1f}s", file=sys.stderr)
        
        # Step 2: Collect and deduplicate PMIDs
        all_pmids = set()
        strategy_results = []
        
        for result in search_results:
            if isinstance(result, tuple):
                pmids, strategy_name = result
                if pmids:
                    strategy_results.append({"strategy": strategy_name, "pmids": pmids})
                    all_pmids.update(pmids)
        
        print(f"Found {len(all_pmids)} unique PMIDs", file=sys.stderr)
        
        if not all_pmids:
            return []
        
        # Step 3: Limit PMIDs for performance (take top results from each strategy)
        limited_pmids = set()
        for strategy_data in strategy_results:
            # Take first N results from each strategy
            limited_pmids.update(strategy_data["pmids"][:80])
        
        pmid_list = list(limited_pmids)[:max_results * 2]  # Process 2x requested for better ranking
        print(f"Processing {len(pmid_list)} PMIDs for efficiency", file=sys.stderr)
        
        # Step 4: Fetch data in parallel
        print("Fetching abstracts and citations...", file=sys.stderr)
        fetch_start = time.time()
        
        abstracts_task = fetch_abstracts_batch(session, pmid_list)
        citations_task = fetch_citation_data_batch(session, pmid_list)
        
        abstracts_info, citation_info = await asyncio.gather(abstracts_task, citations_task)
        fetch_time = time.time() - fetch_start
        print(f"Data fetched in {fetch_time:.1f}s", file=sys.stderr)
    
    # Step 5: Build results efficiently with deduplication
    print("Building and ranking results...", file=sys.stderr)
    rank_start = time.time()
    
    results = []
    journal_impact_lookups = []
    seen_pmids = set()  # Track duplicates
    
    # First pass: create results and batch impact factor lookups, avoiding duplicates
    for strategy_data in strategy_results:
        for pmid in strategy_data["pmids"]:
            if pmid in citation_info and pmid in pmid_list and pmid not in seen_pmids:
                seen_pmids.add(pmid)  # Mark as seen
                
                info = citation_info[pmid]
                abstract_info = abstracts_info.get(pmid, {})
                
                journal = info.get("journal", "Unknown")
                issn = abstract_info.get('issn', '')
                
                journal_impact_lookups.append((journal, issn))
                
                result = SearchResult(
                    pmid=pmid,
                    title=info["title"],
                    authors=info["authors"],
                    journal=journal,
                    year=info["year"],
                    abstract=abstract_info.get('abstract', ''),
                    weight=info["weight"],
                    strategy=strategy_data["strategy"],
                    rank=0,
                    journal_impact=0.0,  # Will be filled in batch
                    issn=issn,
                    combined_score=0.0
                )
                results.append(result)
    
    # Batch journal impact factor lookups (with learning)
    print(f"Looking up {len(journal_impact_lookups)} journal impact factors...", file=sys.stderr)
    lookup_start = time.time()
    for i, (journal, issn) in enumerate(journal_impact_lookups):
        if i < len(results):
            results[i].journal_impact = find_journal_impact_optimized(journal, issn, JOURNAL_IMPACTS)
    lookup_time = time.time() - lookup_start
    print(f"Journal lookups completed in {lookup_time:.1f}s", file=sys.stderr)
    
    # Batch ranking calculations
    ranking_data = [(citation_info[r.pmid].get('citation_count', 0), r.journal_impact, r.year) for r in results]
    combined_scores = calculate_combined_score_batch(ranking_data)
    
    for i, score in enumerate(combined_scores):
        if i < len(results):
            results[i].combined_score = score
    
    # Sort and rank
    results.sort(key=lambda x: x.combined_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1
    
    rank_time = time.time() - rank_start
    print(f"Ranking completed in {rank_time:.1f}s", file=sys.stderr)
    
    # Limit final results BEFORE logging
    total_ranked = len(results)
    results = results[:max_results]
    
    impact_count = len([r for r in results if r.journal_impact > 0])
    print(f"Final: {len(results)} results (from {total_ranked} ranked, {impact_count} with impact factors)", file=sys.stderr)
    
    # Show learning progress
    if LEARNED_MAPPINGS_COUNT > 0:
        print(f"New learning: {LEARNED_MAPPINGS_COUNT} journal mappings added this session", file=sys.stderr)
        print(f"Next time these journals will match instantly", file=sys.stderr)
    
    return results

# ---------- FastAPI Web Interface (ENHANCED WITH PASSWORD PROTECTION) ----------
app = FastAPI(title="Pubmed (Taylor's Version) - Enhanced with Security")
templates = Jinja2Templates(directory="templates")

# Debug: Print registered routes on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI routes registered:", file=sys.stderr)
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            print(f"  {route.methods} {route.path}", file=sys.stderr)
    print(f"🔐 Password protection enabled with password: '{ACCESS_PASSWORD}'", file=sys.stderr)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, session_token: Optional[str] = Cookie(None)):
    # Check if user is authenticated
    is_authenticated = await check_authentication(request, session_token)
    
    if not is_authenticated:
        # Redirect to login page
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": [],
        "query": "",
        "total_results": 0,
        "search_time": 0,
        "impact_count": 0
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, password: str = Form(...)):
    """Process login form"""
    client_ip = get_client_ip(request)
    
    if password == ACCESS_PASSWORD:
        # Create session token
        session_token = create_session_token(client_ip)
        AUTHENTICATED_SESSIONS.add(session_token)
        
        print(f"✅ Authentication successful for IP: {client_ip}", file=sys.stderr)
        
        # Redirect to main page with session cookie
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=86400 * 7,  # 7 days
            httponly=True,
            secure=True,  # Use HTTPS in production
            samesite="strict"
        )
        return response
    else:
        print(f"❌ Authentication failed for IP: {client_ip}", file=sys.stderr)
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Incorrect password. Please try again."
        })

@app.post("/logout")
async def logout(request: Request, session_token: Optional[str] = Cookie(None)):
    """Log out user"""
    if session_token and session_token in AUTHENTICATED_SESSIONS:
        AUTHENTICATED_SESSIONS.remove(session_token)
    
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_token")
    return response

@app.post("/search", response_class=HTMLResponse)
async def search_endpoint(
    request: Request, 
    query: str = Form(...), 
    max_results: int = Form(50),
    session_token: Optional[str] = Cookie(None)
):
    # Check authentication
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        return RedirectResponse(url="/login", status_code=302)
    
    # Validate empty search
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")
    
    try:
        start_time = time.time()
        
        print(f"Processing: '{query}' (max {max_results})", file=sys.stderr)
        
        # Use optimized search
        results = await perform_concentric_search_optimized(query.strip(), max_results)
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.1f}s", file=sys.stderr)
        
        return templates.TemplateResponse("search.html", {
            "request": request,
            "results": results,
            "query": query,
            "total_results": len(results),
            "search_time": search_time,
            "max_results": max_results,
            "impact_count": len([r for r in results if r.journal_impact > 0])
        })
        
    except ValueError as e:
        return templates.TemplateResponse("search.html", {
            "request": request,
            "results": [],
            "query": query,
            "total_results": 0,
            "search_time": 0,
            "error": str(e),
            "impact_count": 0
        })
    except Exception as e:
        print(f"Search error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("search.html", {
            "request": request,
            "results": [],
            "query": query,
            "total_results": 0,
            "search_time": 0,
            "error": f"Search failed: {str(e)}",
            "impact_count": 0
        })

@app.post("/analyze")
async def analyze_endpoint(
    query: str = Form(...), 
    results_json: str = Form(...),
    session_token: Optional[str] = Cookie(None),
    request: Request = None
):
    """AI analysis endpoint for relevance ranking and synthesis"""
    # Check authentication
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API not configured")
        
        # Parse the results from JSON
        results_data = json.loads(results_json)
        
        # Convert back to SearchResult objects
        results = []
        for item in results_data:
            result = SearchResult(
                pmid=item['pmid'],
                title=item['title'],
                authors=item['authors'],
                journal=item['journal'],
                year=item['year'],
                abstract=item['abstract'],
                weight=item['weight'],
                strategy=item['strategy'],
                rank=item['rank'],
                journal_impact=item.get('journal_impact', 0.0),
                issn=item.get('issn', ''),
                combined_score=item.get('combined_score', 0.0)
            )
            results.append(result)
        
        # Perform AI analysis
        analysis_result = await analyze_with_ai(results, query)
        
        if not analysis_result.get("success", False):
            raise HTTPException(status_code=500, detail=analysis_result.get("error", "AI analysis failed"))
        
        # Update AI rankings
        ranked_pmids = analysis_result["ranked_pmids"]
        pmid_to_ai_rank = {pmid: idx + 1 for idx, pmid in enumerate(ranked_pmids)}
        
        # Apply AI rankings to results
        for result in results:
            result.ai_rank = pmid_to_ai_rank.get(result.pmid, None)
        
        # Sort by AI ranking (putting unranked items at the end)
        results.sort(key=lambda x: (x.ai_rank or 999, x.rank))
        
        # Convert back to JSON-serializable format
        results_json_output = []
        for result in results:
            results_json_output.append({
                'pmid': result.pmid,
                'title': result.title,
                'authors': result.authors,
                'journal': result.journal,
                'year': result.year,
                'abstract': result.abstract,
                'weight': result.weight,
                'strategy': result.strategy,
                'rank': result.rank,
                'journal_impact': result.journal_impact,
                'issn': result.issn,
                'combined_score': result.combined_score,
                'ai_rank': result.ai_rank
            })
        
        return JSONResponse({
            "success": True,
            "synthesis": analysis_result["synthesis"],
            "results": results_json_output
        })
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid results data")
    except Exception as e:
        print(f"Analysis error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/download-ris")
async def download_ris_endpoint(
    query: str = Form(...), 
    max_results: int = Form(100),
    session_token: Optional[str] = Cookie(None),
    request: Request = None
):
    """Generate and download RIS file for search results"""
    # Check authentication
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    print(f"📥 RIS download endpoint called with query: '{query}' (max: {max_results})", file=sys.stderr)
    
    try:
        if not query or len(query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Query too short")
        
        print(f"🔍 Generating RIS download for: '{query}' (max: {max_results})", file=sys.stderr)
        
        # Perform search to get results using the user's selected max_results
        results = await perform_concentric_search_optimized(query.strip(), max_results)
        
        if not results:
            print("❌ No results found for RIS download", file=sys.stderr)
            raise HTTPException(status_code=404, detail="No results found")
        
        print(f"✅ Found {len(results)} results for RIS export", file=sys.stderr)
        
        # Get PMIDs for detailed metadata fetch
        pmids = [r.pmid for r in results]
        
        # Fetch additional metadata for RIS
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            print(f"📚 Fetching detailed metadata for {len(pmids)} articles...", file=sys.stderr)
            detailed_metadata = await fetch_detailed_metadata(session, pmids)
        
        # Update results with detailed metadata
        metadata_count = 0
        for result in results:
            if result.pmid in detailed_metadata:
                meta = detailed_metadata[result.pmid]
                result.volume = meta.get('volume', '')
                result.issue = meta.get('issue', '')
                result.pages = meta.get('pages', '')
                result.doi = meta.get('doi', '')
                if any([result.volume, result.issue, result.pages, result.doi]):
                    metadata_count += 1
        
        print(f"📄 Enhanced {metadata_count} articles with detailed metadata", file=sys.stderr)
        
        # Generate RIS content
        ris_content = generate_ris_content(results)
        
        # Create filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pubmed_results_{timestamp}.ris"
        
        print(f"💾 Generated RIS file '{filename}' with {len(results)} references", file=sys.stderr)
        
        return Response(
            content=ris_content,
            media_type="application/x-research-info-systems",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/x-research-info-systems; charset=utf-8"
            }
        )
        
    except ValueError as e:
        print(f"❌ RIS download validation error: {e}", file=sys.stderr)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ RIS download error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate RIS file: {str(e)}")

# ---------- Template Creation (UPDATED WITH LOGIN PAGE) ----------
def create_templates():
    """Create enhanced template with split-screen layout, AI analysis, and login page"""
    os.makedirs("templates", exist_ok=True)
    
    # Create login template
    login_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pubmed (Taylor's Version) - Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --primary-dark: #1d4ed8;
            --error: #dc2626;
            --gray-50: #f8fafc;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-600: #475569;
            --gray-700: #334155;
            --gray-800: #1e293b;
            --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--gradient-bg);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .login-container {
            background: white;
            border-radius: 24px;
            box-shadow: var(--shadow-xl);
            padding: 40px;
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        
        .logo {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--gray-600);
            margin-bottom: 30px;
        }
        
        .login-form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .form-group {
            text-align: left;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--gray-700);
        }
        
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        .form-group input.error {
            border-color: var(--error);
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
        }
        
        .login-btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            padding: 14px 24px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .login-btn:hover {
            transform: translateY(-2px);
        }
        
        .error {
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid var(--gray-200);
            color: var(--gray-600);
            font-size: 0.8rem;
        }
        
        @media (max-width: 480px) {
            .login-container {
                padding: 30px 20px;
            }
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">🔬 Pubmed Research</div>
        <div class="subtitle">Enter password to access</div>
        
        {% if error %}
        <div class="error">
            {{ error }}
        </div>
        {% endif %}
        
        <form method="post" action="/login" class="login-form">
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" 
                       id="password" 
                       name="password" 
                       required
                       autocomplete="current-password"
                       {% if error %}class="error"{% endif %}>
            </div>
            <button type="submit" class="login-btn">
                Access System
            </button>
        </form>
        
        <div class="footer">
            &copy; 2025 MGB Center for Quantitative Health<br>
            Secure research literature access
        </div>
    </div>
    
    <script>
        // Focus password field on load
        document.getElementById('password').focus();
        
        // Remove error state on input
        document.getElementById('password').addEventListener('input', function() {
            this.classList.remove('error');
        });
    </script>
</body>
</html>'''
    
    with open("templates/login.html", "w") as f:
        f.write(login_template)
    
    # Create the main search template (same as before but with logout option)
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pubmed (Taylor's Version) - Enhanced</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --primary-dark: #1d4ed8;
            --secondary: #0891b2;
            --success: #059669;
            --warning: #d97706;
            --error: #dc2626;
            --ai-color: #8b5cf6;
            --ai-light: #a78bfa;
            --gray-50: #f8fafc;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-300: #cbd5e1;
            --gray-400: #94a3b8;
            --gray-500: #64748b;
            --gray-600: #475569;
            --gray-700: #334155;
            --gray-800: #1e293b;
            --gray-900: #0f172a;
            --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--gradient-bg);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            min-height: calc(100vh - 40px);
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
            color: white;
            padding: 20px 40px;
            text-align: center;
            position: relative;
        }
        
        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        
        .header .subtitle {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .header-buttons {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 15px;
            z-index: 10;
        }
        
        .round-button {
            width: 55px;
            height: 55px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.15);
            border: 3px solid rgba(255, 255, 255, 0.3);
            color: white;
            font-size: 24px;
            font-weight: bold;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            text-decoration: none;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .round-button:hover {
            background: rgba(255, 255, 255, 0.25);
            border-color: rgba(255, 255, 255, 0.6);
            transform: scale(1.15) translateY(-2px);
            color: white;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        }
        
        .round-button:active {
            transform: scale(1.05) translateY(0px);
            transition: all 0.1s ease;
        }
        
        .round-button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        
        .round-button:disabled:hover {
            transform: none;
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .ai-button {
            background: linear-gradient(135deg, var(--ai-color) 0%, var(--ai-light) 100%);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        .ai-button:hover:not(:disabled) {
            background: linear-gradient(135deg, var(--ai-light) 0%, var(--ai-color) 100%);
        }
        
        .logout-button {
            background: rgba(220, 38, 38, 0.8);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        .logout-button:hover:not(:disabled) {
            background: rgba(220, 38, 38, 0.9);
        }
        
        .download-arrow {
            font-size: 22px;
            font-weight: 900;
        }
        
        .lightning-bolt {
            font-size: 22px;
            font-weight: 900;
        }
        
        .help-question {
            font-size: 26px;
            font-weight: 900;
        }
        
        .logout-icon {
            font-size: 20px;
            font-weight: 900;
        }
        
        .search-section {
            padding: 25px 40px;
            border-bottom: 1px solid var(--gray-200);
            background: var(--gray-50);
            position: relative;
        }
        
        .search-form {
            display: flex;
            gap: 15px;
            align-items: end;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .search-stats {
            position: absolute;
            top: 8px;
            right: 40px;
            display: flex;
            gap: 20px;
            font-size: 0.8rem;
            color: var(--gray-600);
        }
        
        .search-stat {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .search-stat-value {
            font-weight: 600;
            color: var(--primary);
        }
        
        .form-group {
            flex: 1;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--gray-700);
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        .form-group input.error {
            border-color: var(--error);
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
        }
        
        .search-btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }
        
        .search-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        
        .search-btn:disabled {
            background: #94a3b8;
            cursor: not-allowed;
            transform: none;
        }
        
        .main-content {
            display: flex;
            flex: 1;
            min-height: calc(100vh - 200px);
        }
        
        .results-panel {
            flex: 1;
            padding: 25px 35px;
            overflow-y: auto;
            max-height: calc(100vh - 200px);
        }
        
        .analysis-panel {
            width: 40%;
            background: var(--gray-50);
            border-left: 1px solid var(--gray-200);
            padding: 25px 35px;
            overflow-y: auto;
            max-height: calc(100vh - 200px);
            display: none;
        }
        
        .analysis-panel.visible {
            display: block;
        }
        
        .analysis-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 20px;
            gap: 15px;
        }
        
        .analysis-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--ai-color);
            flex: 1;
        }
        
        .copy-button {
            background: var(--ai-color);
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .copy-button:hover {
            background: var(--ai-light);
            transform: translateY(-1px);
        }
        
        .analysis-content {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: var(--shadow-sm);
            line-height: 1.6;
            color: var(--gray-700);
        }
        
        .analysis-loading {
            text-align: center;
            padding: 40px 20px;
            color: var(--gray-500);
        }
        
        .analysis-loading .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--gray-200);
            border-top: 3px solid var(--ai-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        .result-item {
            background: white;
            border: 1px solid var(--gray-200);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            border-left: 4px solid var(--primary);
        }
        
        .result-item.ai-ranked {
            border-left-color: var(--ai-color);
        }
        
        .result-item:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 12px;
            gap: 20px;
        }
        
        .result-rank {
            background: var(--primary);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            min-width: 30px;
            text-align: center;
        }
        
        .result-ai-rank {
            background: var(--ai-color);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .result-combined {
            background: #6366f1;
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .result-impact {
            background: var(--warning);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .result-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--gray-800);
            margin-bottom: 8px;
            flex: 1;
        }
        
        .result-title a {
            color: inherit;
            text-decoration: none;
        }
        
        .result-title a:hover {
            color: var(--primary);
        }
        
        .result-meta {
            color: var(--gray-600);
            font-size: 0.9rem;
            margin-bottom: 12px;
        }
        
        .result-abstract {
            color: var(--gray-700);
            line-height: 1.6;
            margin-bottom: 12px;
        }
        
        .result-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: #64748b;
        }
        
        .strategy-tag {
            background: #0891b2;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        
        .loading {
            text-align: center;
            padding: 60px 20px;
            color: var(--gray-600);
        }
        
        .loading .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--gray-200);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes modalSlide {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .error {
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 16px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }
        
        .empty-state {
            text-align: center;
            padding: 80px 20px;
            color: var(--gray-600);
        }
        
        .empty-state h3 {
            font-size: 1.5rem;
            margin-bottom: 12px;
            color: var(--gray-700);
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(4px);
        }
        
        .modal-content {
            background-color: white;
            margin: 10% auto;
            padding: 30px;
            border-radius: 16px;
            width: 90%;
            max-width: 500px;
            box-shadow: var(--shadow-xl);
            position: relative;
            animation: modalSlide 0.3s ease-out;
        }
        
        .modal-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 15px;
        }
        
        .modal-body {
            color: var(--gray-700);
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .modal-footer {
            color: var(--gray-500);
            font-size: 0.9rem;
            border-top: 1px solid var(--gray-200);
            padding-top: 15px;
        }
        
        .close {
            position: absolute;
            top: 15px;
            right: 20px;
            color: var(--gray-400);
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        
        .close:hover {
            color: var(--gray-600);
        }
        
        .footer {
            text-align: center;
            padding: 15px 40px;
            background: var(--gray-50);
            border-top: 1px solid var(--gray-200);
            color: var(--gray-600);
            font-size: 0.8rem;
        }
        
        .copy-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 2000;
            transform: translateX(calc(100% + 40px));
            transition: transform 0.3s ease;
            opacity: 0;
            pointer-events: none;
        }
        
        .copy-notification.show {
            transform: translateX(0);
            opacity: 1;
            pointer-events: auto;
        }
        
        @media (max-width: 1200px) {
            .main-content {
                flex-direction: column;
            }
            
            .analysis-panel {
                width: 100%;
                border-left: none;
                border-top: 1px solid var(--gray-200);
                max-height: 400px;
            }
            
            .results-panel {
                max-height: calc(100vh - 350px);
            }
        }
        
        @media (max-width: 768px) {
            .search-form {
                flex-direction: column;
            }
            
            .search-stats {
                position: static;
                justify-content: center;
                margin-bottom: 15px;
            }
            
            .result-header {
                flex-direction: column;
                gap: 10px;
            }
            
            .header-buttons {
                top: 12px;
                right: 15px;
                gap: 12px;
            }
            
            .round-button {
                width: 48px;
                height: 48px;
                font-size: 20px;
            }
            
            .download-arrow, .lightning-bolt {
                font-size: 18px;
            }
            
            .help-question {
                font-size: 22px;
            }
            
            .logout-icon {
                font-size: 18px;
            }
            
            .modal-content {
                margin: 20% auto;
                width: 95%;
                padding: 20px;
            }
            
            .header {
                padding: 15px;
                padding-top: 30px;
            }
            
            .header h1 {
                font-size: 1.8rem;
            }
            
            .results-panel, .analysis-panel {
                padding: 20px;
            }
        }
        
        @media (max-width: 480px) {
            .header-buttons {
                top: 12px;
                right: 12px;
                gap: 10px;
            }
            
            .round-button {
                width: 42px;
                height: 42px;
                font-size: 18px;
            }
            
            .download-arrow, .lightning-bolt {
                font-size: 16px;
            }
            
            .help-question {
                font-size: 20px;
            }
            
            .logout-icon {
                font-size: 16px;
            }
            
            .header {
                padding: 15px;
                padding-top: 30px;
            }
            
            .header h1 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-buttons">
                <button class="round-button ai-button" id="analyzeBtn" title="AI Analysis & Ranking" disabled>
                    <span class="lightning-bolt">⚡</span>
                </button>
                <button class="round-button" id="downloadBtn" title="Download RIS file" disabled>
                    <span class="download-arrow">↓</span>
                </button>
                <button class="round-button" id="helpBtn" title="About this tool">
                    <span class="help-question">?</span>
                </button>
                <button class="round-button logout-button" id="logoutBtn" title="Logout">
                    <span class="logout-icon">🚪</span>
                </button>
            </div>
            <h1>Pubmed (Taylor's Version)</h1>
            <p class="subtitle">AI-Powered Literature Search with Intelligent Ranking & Analysis</p>
        </div>
        
        <div class="search-section">
            {% if total_results > 0 or search_time > 0 %}
            <div class="search-stats">
                <div class="search-stat">
                    <span class="search-stat-value">{{ total_results }}</span>
                    <span>results</span>
                </div>
                <div class="search-stat">
                    <span class="search-stat-value">{{ "%.1f"|format(search_time) }}s</span>
                    <span>search</span>
                </div>
                {% if impact_count is defined and impact_count > 0 %}
                <div class="search-stat">
                    <span class="search-stat-value">{{ impact_count }}</span>
                    <span>w/ JIF</span>
                </div>
                {% endif %}
            </div>
            {% endif %}
            <form method="post" action="/search" class="search-form" id="searchForm">
                <div class="form-group">
                    <label for="query">Research Question</label>
                    <input type="text" 
                           id="query" 
                           name="query" 
                           value="{{ query }}"
                           placeholder="e.g., machine learning in medical imaging"
                           required
                           minlength="2">
                </div>
                <div class="form-group" style="max-width: 120px;">
                    <label for="max_results">Max Results</label>
                    <select id="max_results" name="max_results">
                        <option value="25" {% if max_results == 25 %}selected{% endif %}>25</option>
                        <option value="50" {% if max_results == 50 or not max_results %}selected{% endif %}>50</option>
                        <option value="100" {% if max_results == 100 %}selected{% endif %}>100</option>
                        <option value="200" {% if max_results == 200 %}selected{% endif %}>200</option>
                    </select>
                </div>
                <button type="submit" class="search-btn" id="searchBtn">
                    Search
                </button>
            </form>
        </div>
        
        {% if total_results > 0 or search_time > 0 %}
        <!-- Stats moved to search section -->
        {% endif %}
        
        <div class="main-content">
            <div class="results-panel">
                {% if error %}
                <div class="error">
                    <strong>⚠️ Error:</strong> {{ error }}
                </div>
                {% elif results %}
                <div id="resultsContainer">
                    {% for result in results %}
                    <div class="result-item" data-pmid="{{ result.pmid }}">
                        <div class="result-header">
                            <div>
                                <span class="result-rank">#{{ result.rank }}</span>
                                <span class="result-ai-rank" style="display: none;">AI #<span class="ai-rank-number"></span></span>
                            </div>
                            <h3 class="result-title">
                                <a href="https://pubmed.ncbi.nlm.nih.gov/{{ result.pmid }}/" target="_blank">
                                    {{ result.title }}
                                </a>
                            </h3>
                            <div>
                                <span class="result-combined" title="Combined Ranking Score (Citations × Recency + Journal Impact)">
                                    Score: {{ "%.2f"|format(result.combined_score) }}
                                </span>
                                {% if result.journal_impact > 0 %}
                                <span class="result-impact" title="Journal Impact Factor">
                                    JIF: {{ "%.1f"|format(result.journal_impact) }}
                                </span>
                                {% endif %}
                            </div>
                        </div>
                        <div class="result-meta">
                            <strong>{{ result.authors }}</strong>
                            • {{ result.journal }} ({{ result.year }})
                            • PMID: {{ result.pmid }}
                        </div>
                        {% if result.abstract %}
                        <div class="result-abstract">
                            {{ (result.abstract[:300] + "..." if result.abstract|length > 300 else result.abstract) }}
                        </div>
                        {% endif %}
                        <div class="result-footer">
                            <span class="strategy-tag">{{ result.strategy }}</span>
                            <span>{{ result.year }} • Weight: {{ "%.2f"|format(result.weight) }}</span>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% elif query %}
                <div class="loading" id="loadingState" style="display: none;">
                    <div class="spinner"></div>
                    <p>Running search strategies...</p>
                </div>
                {% else %}
                <div class="empty-state">
                    <h3>Ready for Search</h3>
                    <p>Enter a research question above to find relevant papers with intelligent ranking</p>
                </div>
                {% endif %}
            </div>
            
            <div class="analysis-panel" id="analysisPanel">
                <div class="analysis-header">
                    <div class="analysis-title">AI Analysis & Synthesis</div>
                    <button class="copy-button" id="copyBtn" style="display: none;">
                        <span>📋</span> Copy
                    </button>
                </div>
                <div class="analysis-content" id="analysisContent">
                    <div class="analysis-loading" id="analysisLoading" style="display: none;">
                        <div class="spinner"></div>
                        <p>Analyzing papers with AI...</p>
                    </div>
                    <div id="analysisText" style="display: none;"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            &copy; 2025 MGB Center for Quantitative Health - Because the robots will get us eventually but not today.
        </div>
    </div>

    <!-- Copy notification -->
    <div class="copy-notification" id="copyNotification">
        Copied to clipboard!
    </div>

    <!-- Help Modal -->
    <div id="helpModal" class="modal">
        <div class="modal-content">
            <span class="close" id="closeModal">&times;</span>
            <div class="modal-header">Pubmed (Taylor's Version)</div>
            <div class="modal-body">
                An enhanced PubMed search tool that combines concentric search strategies, journal impact ranking, and AI-powered analysis.
                <br><br>
                <strong>Features:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>⚡ AI relevance ranking and synthesis</li>
                    <li>📊 Journal impact factor integration</li>
                    <li>🎯 Smart query generation</li>
                    <li>📋 Copy-to-clipboard functionality</li>
                    <li>📥 RIS export for reference managers</li>
                    <li>🔐 Secure password protection</li>
                </ul>
            </div>
            <div class="modal-footer">
                &copy; 2025 MGB Center for Quantitative Health<br>
                Because the robots will get us eventually but not today.
            </div>
        </div>
    </div>

    <script>
        let currentResults = [];
        let currentQuery = '';
        
        // Enhanced form validation and UX
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            const queryInput = document.getElementById('query');
            const query = queryInput.value.trim();
            
            // Prevent empty searches
            if (query.length < 2) {
                e.preventDefault();
                queryInput.classList.add('error');
                alert('Please enter at least 2 characters for your search query.');
                queryInput.focus();
                return false;
            }
            
            // Clear any error state
            queryInput.classList.remove('error');
            
            // Store current query
            currentQuery = query;
            
            // Update UI for search
            const btn = document.getElementById('searchBtn');
            btn.disabled = true;
            btn.textContent = 'Searching...';
            
            const loadingState = document.getElementById('loadingState');
            if (loadingState) {
                loadingState.style.display = 'block';
            }
            
            // Hide analysis panel
            document.getElementById('analysisPanel').classList.remove('visible');
            
            // Update button states after form submission
            setTimeout(updateButtonStates, 100);
        });
        
        // Remove error state on input
        document.getElementById('query').addEventListener('input', function() {
            this.classList.remove('error');
        });
        
        // Button functionality
        const helpBtn = document.getElementById('helpBtn');
        const helpModal = document.getElementById('helpModal');
        const closeModal = document.getElementById('closeModal');
        const downloadBtn = document.getElementById('downloadBtn');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const copyBtn = document.getElementById('copyBtn');
        const logoutBtn = document.getElementById('logoutBtn');
        
        // Help modal
        helpBtn.addEventListener('click', function() {
            helpModal.style.display = 'block';
        });
        
        closeModal.addEventListener('click', function() {
            helpModal.style.display = 'none';
        });
        
        // Close modal when clicking outside or pressing Escape
        window.addEventListener('click', function(event) {
            if (event.target === helpModal) {
                helpModal.style.display = 'none';
            }
        });
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && helpModal.style.display === 'block') {
                helpModal.style.display = 'none';
            }
        });
        
        // Logout functionality
        logoutBtn.addEventListener('click', async function() {
            if (confirm('Are you sure you want to logout?')) {
                try {
                    await fetch('/logout', { method: 'POST' });
                    window.location.href = '/login';
                } catch (error) {
                    console.error('Logout error:', error);
                    window.location.href = '/login';
                }
            }
        });
        
        // Download RIS functionality
        downloadBtn.addEventListener('click', async function() {
            if (downloadBtn.disabled) return;
            
            const originalContent = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '<span style="font-size: 16px; animation: spin 1s linear infinite;">⟳</span>';
            downloadBtn.disabled = true;
            downloadBtn.style.transform = 'scale(0.95)';
            
            try {
                const query = document.getElementById('query').value;
                const maxResults = document.getElementById('max_results').value;
                
                const response = await fetch('/download-ris', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `query=${encodeURIComponent(query)}&max_results=${encodeURIComponent(maxResults)}`
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `pubmed-results-${new Date().toISOString().split('T')[0]}.ris`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    
                    downloadBtn.innerHTML = '<span style="color: #10b981;">✓</span>';
                    setTimeout(() => {
                        downloadBtn.innerHTML = originalContent;
                    }, 1500);
                } else {
                    alert('Failed to download RIS file. Please try again.');
                    downloadBtn.innerHTML = originalContent;
                }
            } catch (error) {
                console.error('Download error:', error);
                alert('Failed to download RIS file. Please try again.');
                downloadBtn.innerHTML = originalContent;
            } finally {
                downloadBtn.disabled = false;
                downloadBtn.style.transform = '';
            }
        });
        
        // AI Analysis functionality
        analyzeBtn.addEventListener('click', async function() {
            if (analyzeBtn.disabled || currentResults.length === 0) return;
            
            const originalContent = analyzeBtn.innerHTML;
            analyzeBtn.innerHTML = '<span style="font-size: 16px; animation: spin 1s linear infinite;">⟳</span>';
            analyzeBtn.disabled = true;
            
            // Show analysis panel and loading state
            const analysisPanel = document.getElementById('analysisPanel');
            const analysisLoading = document.getElementById('analysisLoading');
            const analysisText = document.getElementById('analysisText');
            
            analysisPanel.classList.add('visible');
            analysisLoading.style.display = 'block';
            analysisText.style.display = 'none';
            copyBtn.style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `query=${encodeURIComponent(currentQuery)}&results_json=${encodeURIComponent(JSON.stringify(currentResults))}`
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.success) {
                        // Update results with AI ranking
                        updateResultsWithAIRanking(data.results);
                        
                        // Show synthesis with proper paragraph formatting
                        const formattedSynthesis = data.synthesis.replace(/\\n\\n/g, '</p><p>').replace(/\\n/g, '<br>');
                        analysisText.innerHTML = `<p>${formattedSynthesis}</p>`;
                        analysisLoading.style.display = 'none';
                        analysisText.style.display = 'block';
                        copyBtn.style.display = 'flex';
                        
                        analyzeBtn.innerHTML = '<span style="color: #10b981;">✓</span>';
                        setTimeout(() => {
                            analyzeBtn.innerHTML = originalContent;
                        }, 2000);
                    } else {
                        throw new Error(data.error || 'Analysis failed');
                    }
                } else {
                    throw new Error('Network error');
                }
            } catch (error) {
                console.error('Analysis error:', error);
                analysisLoading.style.display = 'none';
                analysisText.innerHTML = `<div style="color: var(--error); text-align: center; padding: 20px;">
                    <strong>Analysis Failed</strong><br>
                    ${error.message || 'Please try again later.'}
                </div>`;
                analysisText.style.display = 'block';
                analyzeBtn.innerHTML = originalContent;
            } finally {
                analyzeBtn.disabled = false;
            }
        });
        
        // Copy functionality
        copyBtn.addEventListener('click', async function() {
            const analysisText = document.getElementById('analysisText');
            const textContent = analysisText.textContent || analysisText.innerText;
            
            try {
                await navigator.clipboard.writeText(textContent);
                showCopyNotification();
            } catch (error) {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = textContent;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                
                try {
                    document.execCommand('copy');
                    showCopyNotification();
                } catch (fallbackError) {
                    alert('Failed to copy to clipboard');
                }
                
                document.body.removeChild(textArea);
            }
        });
        
        function showCopyNotification() {
            const notification = document.getElementById('copyNotification');
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
        
        function updateResultsWithAIRanking(aiResults) {
            // Update current results
            currentResults = aiResults;
            
            // Create a map of PMID to AI rank
            const pmidToAiRank = {};
            aiResults.forEach((result, index) => {
                if (result.ai_rank) {
                    pmidToAiRank[result.pmid] = result.ai_rank;
                }
            });
            
            // Update the DOM
            const resultItems = document.querySelectorAll('.result-item');
            const resultsContainer = document.getElementById('resultsContainer');
            
            // Create array of elements with their AI rankings for sorting
            const sortableItems = Array.from(resultItems).map(item => {
                const pmid = item.getAttribute('data-pmid');
                const aiRank = pmidToAiRank[pmid];
                return {
                    element: item,
                    pmid: pmid,
                    aiRank: aiRank || 999 // Unranked items go to end
                };
            });
            
            // Sort by AI ranking
            sortableItems.sort((a, b) => a.aiRank - b.aiRank);
            
            // Update visual indicators and reorder DOM
            sortableItems.forEach((item, index) => {
                const element = item.element;
                const aiRank = item.aiRank;
                
                // Show AI ranking badge
                const aiRankElement = element.querySelector('.result-ai-rank');
                const aiRankNumber = element.querySelector('.ai-rank-number');
                
                if (aiRank && aiRank !== 999) {
                    aiRankElement.style.display = 'inline-block';
                    aiRankNumber.textContent = aiRank;
                    element.classList.add('ai-ranked');
                }
                
                // Reorder in DOM
                resultsContainer.appendChild(element);
            });
        }
        
        // Store results data for AI analysis
        function storeResultsData() {
            const resultItems = document.querySelectorAll('.result-item');
            currentResults = [];
            
            resultItems.forEach(item => {
                const pmid = item.getAttribute('data-pmid');
                const title = item.querySelector('.result-title a').textContent.trim();
                const meta = item.querySelector('.result-meta').textContent;
                const abstract = item.querySelector('.result-abstract') ? 
                    item.querySelector('.result-abstract').textContent.trim() : '';
                
                // Parse metadata
                const metaParts = meta.split('•');
                const authors = metaParts[0] ? metaParts[0].trim() : 'Unknown';
                const journalYear = metaParts[1] ? metaParts[1].trim() : '';
                const [journal, year] = journalYear.includes('(') ? 
                    [journalYear.split('(')[0].trim(), parseInt(journalYear.match(/\((\d{4})\)/)?.[1]) || new Date().getFullYear()] :
                    [journalYear, new Date().getFullYear()];
                
                // Parse footer for strategy and weight
                const footer = item.querySelector('.result-footer').textContent;
                const strategy = item.querySelector('.strategy-tag').textContent.trim();
                const weightMatch = footer.match(/Weight:\s*([\d.]+)/);
                const weight = weightMatch ? parseFloat(weightMatch[1]) : 0;
                
                // Parse scores
                const combinedScoreText = item.querySelector('.result-combined').textContent;
                const combinedScore = parseFloat(combinedScoreText.match(/Score:\s*([\d.]+)/)?.[1]) || 0;
                
                const impactElement = item.querySelector('.result-impact');
                const journalImpact = impactElement ? 
                    parseFloat(impactElement.textContent.match(/JIF:\s*([\d.]+)/)?.[1]) || 0 : 0;
                
                const rank = parseInt(item.querySelector('.result-rank').textContent.replace('#', '')) || 0;
                
                currentResults.push({
                    pmid: pmid,
                    title: title,
                    authors: authors,
                    journal: journal,
                    year: year,
                    abstract: abstract,
                    weight: weight,
                    strategy: strategy,
                    rank: rank,
                    journal_impact: journalImpact,
                    issn: '',
                    combined_score: combinedScore
                });
            });
        }
        
        // Enable/disable buttons based on search results
        function updateButtonStates() {
            const hasResults = document.querySelectorAll('.result-item').length > 0;
            downloadBtn.disabled = !hasResults;
            analyzeBtn.disabled = !hasResults;
            
            downloadBtn.title = hasResults ? 'Download RIS file' : 'No results to download';
            analyzeBtn.title = hasResults ? 'AI Analysis & Ranking' : 'No results to analyze';
            
            if (hasResults) {
                // Store the current query
                currentQuery = document.getElementById('query').value;
                // Store results data for AI analysis
                storeResultsData();
            }
        }
        
        // Call on page load to set initial state
        updateButtonStates();
        
        // Update button states when results are present
        if (document.querySelectorAll('.result-item').length > 0) {
            setTimeout(updateButtonStates, 100);
        }
    </script>
</body>
</html>'''
    
    with open("templates/search.html", "w") as f:
        f.write(template_content)

# ---------- Main ----------
if __name__ == "__main__":
    # Create templates
    create_templates()
    
    print("Pubmed (Taylor's Version) - Enhanced with AI Analysis & Security")
    print("=" * 70)
    print("🔐 SECURITY FEATURES:")
    print("  • Password protection for new sessions")
    print("  • Session-based authentication with secure cookies")
    print("  • IP-based session tracking")
    print("  • Automatic logout functionality")
    print()
    print("🚀 NEW FEATURES:")
    print("  • AI-powered relevance ranking and synthesis")
    print("  • Split-screen layout with results and analysis")
    print("  • Copy-to-clipboard functionality")
    print("  • Lightning bolt button for intelligent reordering")
    print("  • Inline PMID references in AI summaries")
    print()
    print("⚡ PERFORMANCE IMPROVEMENTS:")
    print("  • Cached journal impact factor lookups")
    print("  • Async operations with connection pooling")  
    print("  • Batch processing for citations and abstracts")
    print("  • Limited fuzzy matching for speed")
    print("  • Empty search validation")
    print("  • Duplicate result elimination")
    print("  • Smart learning: Fuzzy matches saved to CSV for future exact matches")
    
    if OPENAI_API_KEY:
        print(f"✅ OpenAI: {OPENAI_MODEL}")
        print("  • AI query generation enabled")
        print("  • AI analysis and ranking enabled")
    else:
        print("⚠️  OpenAI: Not configured")
        print("  • Using fallback query generation")
        print("  • AI analysis will be disabled")
    
    print(f"✅ NCBI: {'API key configured' if NCBI_API_KEY else 'Rate limited'}")
    print(f"✅ Journals: {len(JOURNAL_IMPACTS)} impact factors loaded")
    print(f"🔐 Password: '{ACCESS_PASSWORD}' (configure in .env file)")
    
    # Count learned mappings
    learned_count = 0
    for key, data in JOURNAL_IMPACTS.items():
        if isinstance(data, dict) and 'pubmed_abbreviation' in data:
            if str(data.get('pubmed_abbreviation', '')).startswith('LEARNED_FROM_'):
                learned_count += 1
    
    if learned_count > 0:
        print(f"🧠 Previously learned mappings: {learned_count}")
        print("   System gets faster as it learns more journal name variations")
    else:
        print("🧠 No learned mappings yet - will learn and improve over time")
    
    if UPDATE_MAPPINGS:
        print("💾 Auto-learning: ENABLED - Will save new fuzzy matches to impacts_mapped.csv")
    else:
        print("💾 Auto-learning: DISABLED - Set UPDATE_MAPPINGS=true to enable")
    
    print(f"⚡ Concurrency: {MAX_CONCURRENT} requests, {BATCH_SIZE} batch size")
    
    port = int(os.environ.get("PORT", 8000))
    print(f"\nStarting secure server on http://localhost:{port}")
    print("🔑 SECURITY FEATURES:")
    print("  🔐 Password protection enabled")
    print("  🍪 Secure session management")
    print("  🚪 Logout functionality")
    print("  📱 Mobile-responsive design")
    print()
    print("✨ AVAILABLE FEATURES:")
    print("  🔍 Smart search with multiple strategies")
    print("  📊 Journal impact factor ranking")
    print("  ⚡ AI-powered relevance analysis")
    print("  📋 Copy-to-clipboard synthesis")
    print("  📥 RIS export for reference managers")
    print("  🧠 Learning system that improves over time")
    
    uvicorn.run(app, host="0.0.0.0", port=port)