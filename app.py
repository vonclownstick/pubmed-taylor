#!/usr/bin/env python3
"""
Pubmed (Taylor's Version) - Enhanced with AI Analysis and Password Protection
Smart concentric search with LLM-generated queries, citation weighting, and AI-powered relevance ranking
OPTIMIZED VERSION - 3-5x Performance Improvement
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
from typing import Dict, List, Optional, Tuple, Set
from functools import lru_cache
import json
import hashlib
import secrets
import zipfile
import io

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException, Cookie
from fastapi.responses import HTMLResponse, Response, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files BEFORE defining routes
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Failed to mount static files: {e}")

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
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "Cassowary")
DEBUG_JOURNALS = os.getenv("DEBUG_JOURNALS", "false").lower() in ["true", "1", "yes"]
UPDATE_MAPPINGS = os.getenv("UPDATE_MAPPINGS", "true").lower() in ["true", "1", "yes"]

NOW = dt.datetime.now().year
NCBI_TIMEOUT = 15
BATCH_SIZE = 800
MAX_CONCURRENT = 10
RATE_LIMIT_SEC = 0.02 if NCBI_API_KEY else 0.1

# ---------- OPTIMIZATION ADDITIONS ----------
# Pre-compiled regex patterns for better performance
TITLE_CLEANUP_REGEX = re.compile(r'[^\w\s]')
WHITESPACE_REGEX = re.compile(r'\s+')

# Enhanced caching
TFIDF_CACHE = {}
SIMILARITY_CACHE = {}
LLM_QUERY_CACHE = {}

# Configuration for optimized processing
OPTIMIZED_CONFIG = {
    "TITLE_SEMAPHORE": 15,        # Up from 6
    "TITLE_BATCH_SIZE": 200,      # Down from 300 for better parallelism  
    "ABSTRACT_BATCH_SIZE": 600,   # Up from 400
    "CITATION_BATCH_SIZE": 600,   # Up from 400
    "CONNECTION_LIMIT": 200,      # Up from 100
    "CONNECTION_PER_HOST": 50,    # Up from default
    "MINIMAL_DELAY": 0.005,       # Reduced from 0.2-0.3
    "TWO_STAGE_FACTOR": 8,        # Fetch details for top 8x results before final filtering
}

# Global cache for fuzzy matching results
FUZZY_MATCH_CACHE = {}
LEARNED_MAPPINGS_COUNT = 0

# Session management for password protection
AUTHENTICATED_SESSIONS = set()
SESSION_SECRET = secrets.token_hex(32)

# Production security settings
if os.getenv("ENVIRONMENT") == "production":
    if not ACCESS_PASSWORD or ACCESS_PASSWORD in ["Cassowary", "change_this_secure_password"]:
        raise ValueError("Must set a secure ACCESS_PASSWORD for production")

# Load environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

def get_client_ip(request: Request) -> str:
    """Get client IP address, handling proxies"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    if request.client:
        return request.client.host
    
    return "unknown"

def create_session_token(ip: str) -> str:
    """Create a session token for an authenticated IP"""
    timestamp = str(int(time.time()))
    data = f"{ip}:{timestamp}:{SESSION_SECRET}"
    return hashlib.sha256(data.encode()).hexdigest()

def verify_session_token(token: str, ip: str) -> bool:
    """Verify a session token is valid and belongs to the IP"""
    if not token:
        return False
    return token in AUTHENTICATED_SESSIONS

async def check_authentication(
    request: Request, 
    session_token: Optional[str] = Cookie(None)
) -> bool:
    """Check if the request is authenticated"""
    client_ip = get_client_ip(request)
    
    if session_token and verify_session_token(session_token, client_ip):
        return True
    
    return False

# ---------- Journal Impact Factor Data ----------
@lru_cache(maxsize=50000)  # Increased cache size
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
    """Load journal impact factor data with better indexing"""
    impacts = {}
    
    if not os.path.exists(csv_file):
        return impacts
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                journal_title = row.get('journal_title', '').strip()
                pubmed_abbrev = row.get('pubmed_abbreviation', '').strip()
                issn = row.get('issn', '').strip()
                
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
                    
                    normalized_title = normalize_journal_name_cached(journal_title)
                    normalized_abbrev = normalize_journal_name_cached(pubmed_abbrev) if pubmed_abbrev else ''
                    
                    if normalized_title:
                        impacts[normalized_title] = impact_data
                    
                    if normalized_abbrev and normalized_abbrev != normalized_title:
                        impacts[normalized_abbrev] = impact_data
                    
                    # Index by ISSN
                    if issn:
                        impacts[issn] = impact_data
                        clean_issn = issn.replace('-', '').replace(' ', '')
                        if clean_issn != issn:
                            impacts[clean_issn] = impact_data
            
    except Exception as e:
        print(f"Error loading {csv_file}: {e}", file=sys.stderr)
    
    return impacts

def lookup_journal_impact_optimized(impacts_db: Dict[str, Dict], journal_name: str, issn: str) -> float:
    """Optimized journal impact factor lookup with caching and limited fuzzy matching"""
    if not impacts_db:
        return 0.0
    
    cache_key = f"{journal_name}|{issn}"
    if cache_key in FUZZY_MATCH_CACHE:
        return FUZZY_MATCH_CACHE[cache_key]
    
    # Try exact ISSN match first
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
    
    # Try exact normalized journal name match
    if journal_name:
        normalized = normalize_journal_name_cached(journal_name)
        
        if normalized in impacts_db:
            result = impacts_db[normalized]['jif_sans_self']
            FUZZY_MATCH_CACHE[cache_key] = result
            return result
        
        # Limited fuzzy matching for performance
        if len(journal_name) > 5 and len(journal_name) < 100:
            best_score = 0
            best_match = None
            candidates_checked = 0
            max_candidates = 500
            
            for stored_key, data in impacts_db.items():
                if candidates_checked >= max_candidates:
                    break
                    
                if not isinstance(data, dict) or stored_key.isdigit():
                    continue
                
                candidates_checked += 1
                
                if len(stored_key) > 3:
                    common_chars = set(normalized.lower()) & set(stored_key.lower())
                    char_overlap = len(common_chars) / min(len(normalized), len(stored_key))
                    
                    if char_overlap > 0.4:
                        try:
                            from difflib import SequenceMatcher
                            similarity = SequenceMatcher(None, normalized, stored_key).ratio()
                            
                            if similarity > 0.75 and similarity > best_score:
                                best_score = similarity
                                best_match = data
                        except ImportError:
                            pass
            
            if best_match and best_score >= 0.75:
                result = best_match['jif_sans_self']
                FUZZY_MATCH_CACHE[cache_key] = result
                
                if best_score >= 0.80:
                    save_learned_mapping(journal_name, issn, best_match, best_score)
                
                return result
    
    FUZZY_MATCH_CACHE[cache_key] = 0.0
    return 0.0

def save_learned_mapping(pubmed_name: str, issn: str, matched_data: Dict, confidence_score: float, csv_file: str = "impacts_mapped.csv") -> None:
    """Save a successful fuzzy match to CSV file for future exact matching"""
    global LEARNED_MAPPINGS_COUNT
    
    if not UPDATE_MAPPINGS:
        return
        
    try:
        normalized_pubmed = normalize_journal_name_cached(pubmed_name)
        
        if normalized_pubmed in JOURNAL_IMPACTS:
            return
        
        JOURNAL_IMPACTS[normalized_pubmed] = matched_data.copy()
        
        if pubmed_name.strip() != normalized_pubmed:
            JOURNAL_IMPACTS[pubmed_name.strip()] = matched_data.copy()
        
        if issn:
            clean_issn = issn.replace('-', '').replace(' ', '')
            if issn not in JOURNAL_IMPACTS:
                JOURNAL_IMPACTS[issn] = matched_data.copy()
            if clean_issn != issn and clean_issn not in JOURNAL_IMPACTS:
                JOURNAL_IMPACTS[clean_issn] = matched_data.copy()
        
        if os.path.exists(csv_file):
            backup_file = csv_file + ".backup"
            if not os.path.exists(backup_file):
                import shutil
                shutil.copy2(csv_file, backup_file)
            
            with open(csv_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                new_row = [
                    pubmed_name,
                    issn or matched_data.get('issn', ''),
                    '', '', '', '',
                    str(matched_data['jif_sans_self']),
                    '', '', '',
                    f"LEARNED_FROM_{matched_data.get('journal_title', 'FUZZY_MATCH')}"
                ]
                
                writer.writerow(new_row)
            
            LEARNED_MAPPINGS_COUNT += 1
        
    except Exception as e:
        print(f"Failed to save learned mapping: {e}", file=sys.stderr)

# Load journal impacts at startup
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
    ai_rank: Optional[int] = None
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    
    def to_dict(self) -> dict:
        """Convert SearchResult to dictionary for JSON serialization"""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'authors': self.authors,
            'journal': self.journal,
            'year': self.year,
            'abstract': self.abstract,
            'weight': self.weight,
            'strategy': self.strategy,
            'rank': self.rank,
            'journal_impact': self.journal_impact,
            'issn': self.issn,
            'combined_score': self.combined_score,
            'ai_rank': self.ai_rank,
            'volume': self.volume,
            'issue': self.issue,
            'pages': self.pages,
            'doi': self.doi
        }

# ---------- Enhanced Ranking Functions ----------
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

def calculate_total_weight(year: int, citations_per_year: float, journal_impact: float, current_year: int = NOW) -> float:
    """Calculate a combined total weight score incorporating recency, citations, and journal impact"""
    
    # Recency component (0-1 scale, more recent is better)
    years_ago = current_year - year
    if years_ago <= 1:
        recency_score = 1.0
    elif years_ago <= 3:
        recency_score = 0.8
    elif years_ago <= 5:
        recency_score = 0.6
    elif years_ago <= 10:
        recency_score = 0.4
    else:
        recency_score = max(0.1, 1.0 - (years_ago - 10) * 0.05)
    
    # Citations component (normalized)
    citations_score = min(1.0, citations_per_year / 10.0)  # Cap at 10 citations/year for normalization
    
    # Journal impact component (normalized using log scale)
    if journal_impact > 0:
        impact_score = min(1.0, math.log10(journal_impact + 1) / math.log10(50))  # Cap at IF=50 for normalization
    else:
        impact_score = 0.0
    
    # Weighted combination (adjust weights as needed)
    total_weight = (recency_score * 0.3 + citations_score * 0.4 + impact_score * 0.3) * 100
    
    return round(total_weight, 2)

# ---------- AI Analysis Functions ----------
async def analyze_with_ai(results: List[SearchResult], query: str) -> Dict:
    """Send results to AI for relevance ranking and summary generation"""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured")
    
    if not results:
        raise ValueError("No results to analyze")
    
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
    
    prompt = f"""You are analyzing scientific papers for relevance to the research question: "{query}"

Please analyze these {len(papers_data)} papers and provide:

1. A ranked list of PMIDs from most relevant to least relevant based on title and abstract content
2. A comprehensive synthesis organized as exactly 3 distinct paragraphs

SYNTHESIS REQUIREMENTS:
- Write exactly 3 paragraphs separated by blank lines
- Paragraph 1: Overview of the research landscape and main themes (avoid a vague initial sentence, just jump into content!)
- Paragraph 2: Key findings and methodological approaches  
- Paragraph 3: Implications, gaps, and future directions identified in abstracts
- Only reference papers provided using format (PMID: 12345678)
- Include multiple relevant PMIDs throughout each paragraph
- Focus on findings most relevant to: "{query}"
- Use scientific writing style: 

STYLE
- Vary sentence lengths and shapes. Target ~25–35% short (<10 words), 45–60% medium (10–20), 10–25% long (20+).
- Avoid hedging statements and stock disclaimers ("It is important to note that…", "Of note…"). 

LEXICON (SOFT AVOID LIST)
Use plain, specific words over buzzwords. DO NOT USE these words: moreover, furthermore, additionally, thus, therefore, consequently, accordingly, indeed, robust, comprehensive, nuanced, holistic, landscape, realm, framework, granular, cohesive, leverage/leveraging, empower/empowering, unlock, harness, transformative, synergy/synergistic, unprecedented, pivotal, imperative, cornerstone, mitigate, underscore, elucidate, articulate, navigate, pertaining, noteworthy, tapestry, amidst, akin, delve, foster, foray, vital, vibrant, undeniably/undoubtedly.
→ Replace with concrete, domain-specific nouns/verbs and precise adjectives.

CONTENT
- Never fabricate citations or quotes. Do not include "As an AI…" self-references.

OUTPUT
- Produce continuous professional scientific avoiding bullets, steps, or a table.

INTERNAL SELF-CHECK (DO NOT PRINT)
Before returning the final text, silently verify:
1) Sentence starts using transition starters ≤15%.
2) Hedging phrases ≤1 per ~200 words and replaced with precise qualifiers where possible.
3) "Soft avoid" lexicon frequency near zero; replaced with concrete alternatives.
4) Paragraph lengths vary (no uniform 3–4 sentences each), and no templated intro/body/conclusion pattern.
5) Details included when appropriate.
Return only the final text.

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
                {"role": "system", "content": "You are a scientific writer. Return only valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        raw_content = result["choices"][0]["message"]["content"]
        
        try:
            parsed_result = json.loads(raw_content)
            
            if "ranked_pmids" not in parsed_result or "synthesis" not in parsed_result:
                raise ValueError("AI response missing required fields")
            
            ranked_pmids = parsed_result["ranked_pmids"]
            synthesis = parsed_result["synthesis"]
            
            original_pmids = {r.pmid for r in results}
            valid_ranked_pmids = [pmid for pmid in ranked_pmids if pmid in original_pmids]
            
            if not valid_ranked_pmids:
                raise ValueError("No valid PMIDs in AI ranking")
            
            return {
                "ranked_pmids": valid_ranked_pmids,
                "synthesis": synthesis,
                "success": True
            }
            
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse AI response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------- Advanced Search Filter Functions ----------
def apply_search_filters(base_query: str, clinical_trials_only: bool, exclude_reviews: bool, year_filter: str) -> str:
    """Apply advanced search filters to the base query"""
    filtered_query = base_query.strip()
    
    # Apply clinical trials filter
    if clinical_trials_only:
        filtered_query += " AND (clinical trial[pt] OR randomized controlled trial[pt] OR controlled clinical trial[pt])"
    
    # Apply exclude reviews filter
    if exclude_reviews:
        filtered_query += " AND NOT (review[pt] OR systematic review[pt] OR meta-analysis[pt])"
    
    # Apply year filter - be more careful with the syntax
    if year_filter != "all" and year_filter.strip():
        try:
            years_back = int(year_filter)
            if years_back > 0:  # Ensure positive number
                current_year = dt.datetime.now().year
                start_year = current_year - years_back
                # Use a more robust date range format
                filtered_query += f" AND ({start_year}[pdat]:{current_year}[pdat])"
                print(f"DEBUG FILTER: Applied year filter: {start_year}-{current_year}")
        except ValueError:
            print(f"DEBUG FILTER: Invalid year filter value: {year_filter}")
            pass  # Invalid year filter, ignore
    
    print(f"DEBUG FILTER: Final query: {filtered_query}")
    return filtered_query

# ---------- LLM Query Generation ----------
def generate_smart_queries_cached(nlq: str, model: str = OPENAI_MODEL) -> List[str]:
    """Cached version of smart query generation"""
    cache_key = hash(f"{nlq}:{model}")
    
    if cache_key in LLM_QUERY_CACHE:
        return LLM_QUERY_CACHE[cache_key]
    
    result = generate_smart_queries(nlq, model)
    LLM_QUERY_CACHE[cache_key] = result
    return result

def generate_smart_queries(nlq: str, model: str = OPENAI_MODEL) -> List[str]:
    """Generate smart queries using OpenAI or fallback"""
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
            "max_tokens": 400,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        raw = result["choices"][0]["message"]["content"]
        
        parsed = json.loads(raw)
        queries = parsed.get("queries", [])
        
        clean_queries = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 5]
        
        if len(clean_queries) >= 3:
            return clean_queries[:4]
        else:
            return generate_fallback_queries(nlq)
                
    except Exception:
        return generate_fallback_queries(nlq)

def generate_smart_queries_with_filters(nlq: str, clinical_trials_only: bool, exclude_reviews: bool, year_filter: str, model: str = OPENAI_MODEL) -> List[str]:
    """Generate smart queries with advanced filters applied"""
    
    # Generate base queries first
    base_queries = generate_smart_queries_cached(nlq, model)
    
    # Apply filters to each query
    filtered_queries = []
    for query in base_queries:
        filtered_query = apply_search_filters(query, clinical_trials_only, exclude_reviews, year_filter)
        filtered_queries.append(filtered_query)
    
    return filtered_queries

def generate_fallback_queries(nlq: str) -> List[str]:
    """Generate fallback queries when OpenAI is not available"""
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

def generate_fallback_queries_with_filters(nlq: str, clinical_trials_only: bool, exclude_reviews: bool, year_filter: str) -> List[str]:
    """Generate fallback queries with filters when OpenAI is not available"""
    base_queries = generate_fallback_queries(nlq)
    
    # Apply filters to each query
    filtered_queries = []
    for query in base_queries:
        filtered_query = apply_search_filters(query, clinical_trials_only, exclude_reviews, year_filter)
        filtered_queries.append(filtered_query)
    
    return filtered_queries

# ---------- NCBI API helpers ----------
def _eparams(extra: dict = None) -> dict:
    """Add NCBI email and API key to parameters"""
    p = {} if extra is None else dict(extra)
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p

async def esearch_count_async(session: aiohttp.ClientSession, query: str) -> int:
    """Return total count for a query via ESearch"""
    if not query or len(query.strip()) < 2:
        return 0
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query.strip(), "retmode": "json", "retmax": 0}
    params.update(_eparams())
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=12)) as response:
            response.raise_for_status()
            text = await response.text()
            if "down_bethesda.html" in text or "<title>302 Found</title>" in text:
                raise RuntimeError("NCBI E-utilities maintenance redirect detected")
            data = await response.json()
            return int(data.get("esearchresult", {}).get("count", 0))
    except Exception:
        return 0

async def esearch_ids_paged_async(session: aiohttp.ClientSession, query: str, limit: int, page_size: int = 10000, sort: str = "relevance") -> List[str]:
    """Retrieve PMIDs for a query by paging retstart"""
    if limit <= 0:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    pmids: List[str] = []
    retstart = 0
    while len(pmids) < limit:
        fetch = min(page_size, limit - len(pmids))
        params = {
            "db": "pubmed",
            "term": query.strip(),
            "retmode": "json",
            "retmax": fetch,
            "retstart": retstart,
            "sort": sort
        }
        params.update(_eparams())
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                response.raise_for_status()
                data = await response.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                ids = [str(x).strip() for x in ids if str(x).strip().isdigit()]
                if not ids:
                    break
                pmids.extend(ids)
                retstart += len(ids)
                await asyncio.sleep(RATE_LIMIT_SEC)
        except Exception:
            break
    return pmids

# ---------- OPTIMIZED FUNCTIONS ----------

@lru_cache(maxsize=1000)
def preprocess_text_cached(text: str) -> str:
    """Cached text preprocessing for TF-IDF"""
    if not text:
        return ""
    # Use pre-compiled regex
    cleaned = TITLE_CLEANUP_REGEX.sub(' ', text.lower())
    normalized = WHITESPACE_REGEX.sub(' ', cleaned).strip()
    return normalized

async def esummary_titles_async_optimized(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, str]:
    """Heavily optimized title fetching with maximum parallelization"""
    titles: Dict[str, str] = {}
    if not pmids:
        return titles
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    # Smaller batches for better parallelization
    BATCH = OPTIMIZED_CONFIG["TITLE_BATCH_SIZE"]
    batches = [pmids[i:i+BATCH] for i in range(0, len(pmids), BATCH)]
    
    async def fetch_single_batch(batch: List[str]) -> Dict[str, str]:
        params = {"db": "pubmed", "retmode": "json", "id": ",".join(batch)}
        params.update(_eparams())
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=12)) as response:
                response.raise_for_status()
                data = await response.json()
                result = data.get("result", {})
                batch_titles = {}
                for uid in result.get("uids", []):
                    title = result.get(uid, {}).get("title") or ""
                    batch_titles[uid] = title
                return batch_titles
        except Exception:
            return {}
    
    # MAXIMUM PARALLELISM: Much higher semaphore limit
    semaphore = asyncio.Semaphore(OPTIMIZED_CONFIG["TITLE_SEMAPHORE"])
    
    async def controlled_fetch(batch):
        async with semaphore:
            result = await fetch_single_batch(batch)
            await asyncio.sleep(OPTIMIZED_CONFIG["MINIMAL_DELAY"])  # Minimal delay
            return result
    
    # Process ALL batches in parallel
    tasks = [controlled_fetch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict):
            titles.update(result)
    
    return titles

def tfidf_rank_titles_optimized(user_query: str, titles_by_pmid: Dict[str, str]) -> List[Tuple[str, float]]:
    """Optimized TF-IDF ranking with caching and better vectorizer settings"""
    
    # Create cache key
    query_hash = hash(user_query)
    titles_hash = hash(tuple(sorted(titles_by_pmid.items())))
    cache_key = (query_hash, titles_hash)
    
    # Check cache first
    if cache_key in TFIDF_CACHE:
        return TFIDF_CACHE[cache_key]
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        # Fallback: optimized keyword overlap
        q_terms = set(preprocess_text_cached(user_query).split())
        scored = []
        for pmid, title in titles_by_pmid.items():
            t_terms = set(preprocess_text_cached(title).split())
            inter = len(q_terms & t_terms)
            score = inter / (len(q_terms) + 1e-9)
            scored.append((pmid, float(score)))
        result = sorted(scored, key=lambda x: x[1], reverse=True)
        TFIDF_CACHE[cache_key] = result
        return result

    pmids = list(titles_by_pmid.keys())
    titles = [titles_by_pmid[p] for p in pmids]
    
    # Preprocess all text
    processed_query = preprocess_text_cached(user_query)
    processed_titles = [preprocess_text_cached(title) for title in titles]
    corpus = [processed_query] + processed_titles
    
    # Optimized vectorizer settings
    vec = TfidfVectorizer(
        lowercase=False,  # Already lowercased
        ngram_range=(1, 2), 
        max_df=0.8,
        max_features=10000,  # Increased
        stop_words='english',  # Add stop words
        min_df=1,
        sublinear_tf=True  # Better for short documents
    )
    
    try:
        mat = vec.fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        result = sorted([(p, float(s)) for p, s in zip(pmids, sims)], key=lambda x: x[1], reverse=True)
        
        # Cache the result
        TFIDF_CACHE[cache_key] = result
        return result
    except Exception:
        result = [(pmid, 0.0) for pmid in pmids]
        TFIDF_CACHE[cache_key] = result
        return result

async def elink_expand_async(session: aiohttp.ClientSession, seed_pmids: List[str], mode: str = "similar", max_add: int = 500) -> Set[str]:
    """Expand a set of PMIDs using ELink"""
    if not seed_pmids:
        return set()
    linkmap = {
        "similar": "pubmed_pubmed",
        "citedin": "pubmed_pubmed_citedin",
        "references": "pubmed_pubmed_refs",
    }
    linkname = linkmap.get(mode, "pubmed_pubmed")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    out: Set[str] = set()
    BATCH = 200
    for i in range(0, len(seed_pmids), BATCH):
        if len(out) >= max_add: 
            break
        batch = seed_pmids[i:i+BATCH]
        params = {"dbfrom": "pubmed", "linkname": linkname, "retmode": "json", "id": ",".join(batch)}
        params.update(_eparams())
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                response.raise_for_status()
                data = await response.json()
                linksets = data.get("linksets", [])
                for ls in linksets:
                    for ld in ls.get("linksetdbs", []):
                        if ld.get("linkname") == linkname:
                            for rec in ld.get("links", []):
                                pmid = str(rec).strip()
                                if pmid.isdigit():
                                    out.add(pmid)
                await asyncio.sleep(RATE_LIMIT_SEC)
        except Exception:
            pass
    return set(list(out)[:max_add])

def process_authors_fast(authors_raw) -> str:
    """Fast author processing"""
    if isinstance(authors_raw, list) and authors_raw:
        first_author = authors_raw[0]
        if isinstance(first_author, dict):
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
            authors = str(first_author)
    elif isinstance(authors_raw, str):
        if ';' in authors_raw:
            authors = authors_raw.split(';')[0].strip()
        elif ',' in authors_raw and len(authors_raw.split(',')) > 2:
            authors = authors_raw.split(',')[0].strip()
        else:
            authors = authors_raw
    else:
        authors = "Unknown"
    
    if len(authors) > 50:
        authors = authors[:47] + "..."
    
    return authors

async def fetch_abstracts_batch_optimized(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Dict]:
    """Optimized abstract fetching with larger batches and better parallelism"""
    if not pmids:
        return {}
    
    info_map = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # Larger batch sizes
    batch_size = OPTIMIZED_CONFIG["ABSTRACT_BATCH_SIZE"]
    batches = [pmids[i:i+batch_size] for i in range(0, len(pmids), batch_size)]
    
    async def fetch_single_batch(batch: List[str]) -> Dict[str, Dict]:
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(batch)
        }
        params.update(_eparams())
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=25)) as response:
                response.raise_for_status()
                xml_text = await response.text()
                
                batch_info = {}
                root = ET.fromstring(xml_text)
                for art in root.findall(".//PubmedArticle"):
                    pid = art.findtext(".//PMID")
                    if pid:
                        abstr = art.findtext(".//Abstract/AbstractText") or ""
                        
                        issn = (art.findtext(".//ISSN[@IssnType='Print']") or 
                               art.findtext(".//ISSN[@IssnType='Electronic']") or 
                               art.findtext(".//ISSNLinking") or 
                               art.findtext(".//ISSN") or "")
                        
                        journal_title = (art.findtext(".//Journal/Title") or 
                                       art.findtext(".//Journal/ISOAbbreviation") or 
                                       art.findtext(".//MedlineJournalInfo/MedlineTA") or "")
                        
                        batch_info[pid] = {
                            'abstract': abstr,
                            'issn': issn,
                            'journal_title': journal_title
                        }
                
                return batch_info
                        
        except Exception:
            return {}
    
    # Higher parallelism for abstracts
    semaphore = asyncio.Semaphore(10)
    
    async def controlled_fetch(batch):
        async with semaphore:
            result = await fetch_single_batch(batch)
            await asyncio.sleep(OPTIMIZED_CONFIG["MINIMAL_DELAY"])
            return result

    tasks = [controlled_fetch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict):
            info_map.update(result)
    
    return info_map

# QUICK FIX: Replace your fetch_citation_data_batch_optimized function with this version

async def fetch_citation_data_batch_optimized(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Dict]:
    """Fixed version with iCite fallback when API fails"""
    info = {}
    
    # Try iCite first (smaller batches to avoid errors)
    batch_size = 50  # Much smaller than 600
    batches = [pmids[i:i+batch_size] for i in range(0, len(pmids), batch_size)]
    
    icite_success = False
    
    async def fetch_icite_batch(batch: List[str]) -> Dict[str, Dict]:
        try:
            url = "https://icite.od.nih.gov/api/pubs"
            params = {"pmids": ",".join(batch)}
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    batch_info = {}
                    
                    for item in data.get("data", []):
                        pmid = str(item.get("pmid", "")).strip()
                        if pmid:
                            authors_raw = item.get("authors", [])
                            authors = process_authors_fast(authors_raw)
                            
                            batch_info[pmid] = {
                                "weight": float(item.get("citations_per_year", 0) or 0),
                                "year": int(item.get("year", 2025) or 2025),
                                "journal": str(item.get("journal", "") or "Unknown"),
                                "authors": authors,
                                "title": str(item.get("title", "") or "Unknown"),
                                "citation_count": int(item.get("citation_count", 0) or 0)
                            }
                    
                    return batch_info
                else:
                    print(f"iCite API error {response.status}: {await response.text()}")
                    return {}
        except Exception as e:
            print(f"iCite batch failed: {e}")
            return {}
    
    # Try iCite with first few batches
    print(f"Attempting iCite for {len(pmids)} PMIDs...")
    for i, batch in enumerate(batches[:3]):  # Only try first 3 batches
        batch_result = await fetch_icite_batch(batch)
        if batch_result:
            info.update(batch_result)
            icite_success = True
            print(f"iCite batch {i+1} successful: {len(batch_result)} records")
        else:
            print(f"iCite batch {i+1} failed")
            break  # Stop trying if one fails
        await asyncio.sleep(0.3)
    
    # FALLBACK: Get missing data from NCBI ESummary when iCite fails
    missing_pmids = [pmid for pmid in pmids if pmid not in info]
    
    if missing_pmids:
        print(f"Using NCBI ESummary fallback for {len(missing_pmids)} PMIDs...")
        
        # Get data from ESummary
        esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi" 
        esummary_batches = [missing_pmids[i:i+100] for i in range(0, len(missing_pmids), 100)]
        
        for batch in esummary_batches:
            try:
                params = {"db": "pubmed", "retmode": "json", "id": ",".join(batch)}
                params.update(_eparams())
                
                async with session.get(esummary_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("result", {})
                        
                        for uid in result.get("uids", []):
                            item = result.get(uid, {})
                            
                            # Extract authors from ESummary format
                            authors_list = item.get("authors", [])
                            authors = "Unknown"
                            if authors_list and isinstance(authors_list, list):
                                first_author = authors_list[0]
                                if isinstance(first_author, dict):
                                    author_name = first_author.get("name", "").strip()
                                    if author_name:
                                        authors = author_name
                                        if len(authors_list) > 1:
                                            authors += " et al."
                            
                            # Extract year from pubdate 
                            year = 2025
                            pubdate = item.get("pubdate", "")
                            if pubdate:
                                import re
                                year_match = re.search(r'(\d{4})', pubdate)
                                if year_match:
                                    year = int(year_match.group(1))
                            
                            info[uid] = {
                                "weight": 0.0,  # No citation data available
                                "year": year,
                                "journal": item.get("source", "Unknown"),
                                "authors": authors,
                                "title": item.get("title", "Unknown"),
                                "citation_count": 0
                            }
                        
                        print(f"ESummary fallback successful: {len(result.get('uids', []))} records")
                
                await asyncio.sleep(0.2)
                
            except Exception as e:
                print(f"ESummary fallback failed: {e}")
    
    # Fill in any remaining missing PMIDs with defaults
    for pmid in pmids:
        if pmid not in info:
            info[pmid] = {
                "weight": 0.0,
                "year": 2025,
                "journal": "Unknown",
                "authors": "Unknown",
                "title": "Unknown",
                "citation_count": 0
            }
    
    success_source = "iCite" if icite_success else "ESummary fallback"
    print(f"Citation data complete: {len(info)} records from {success_source}")
    
    return info

async def fetch_detailed_metadata(session: aiohttp.ClientSession, pmids: List[str]) -> Dict[str, Dict]:
    """Fetch detailed metadata for RIS export"""
    if not pmids:
        return {}
    
    detailed_info = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
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
                        volume = art.findtext(".//Volume") or ""
                        issue = art.findtext(".//Issue") or ""
                        
                        start_page = art.findtext(".//StartPage") or art.findtext(".//MedlinePgn") or ""
                        end_page = art.findtext(".//EndPage") or ""
                        if start_page and end_page:
                            pages = f"{start_page}-{end_page}"
                        elif start_page:
                            pages = start_page
                        else:
                            pages = ""
                        
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
                        
        except Exception:
            return {}
    
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

def create_optimized_session() -> aiohttp.ClientSession:
    """Create aiohttp session with optimized settings"""
    timeout = aiohttp.ClientTimeout(total=45, connect=10)
    connector = aiohttp.TCPConnector(
        limit=OPTIMIZED_CONFIG["CONNECTION_LIMIT"],
        limit_per_host=OPTIMIZED_CONFIG["CONNECTION_PER_HOST"],
        ttl_dns_cache=300,      # DNS caching
        use_dns_cache=True,
        enable_cleanup_closed=True,
        keepalive_timeout=30,   # Keep connections alive longer
        force_close=False       # Reuse connections
    )
    return aiohttp.ClientSession(timeout=timeout, connector=connector)

# ---------- MAIN SEARCH PIPELINE WITH FILTERS ----------
async def perform_concentric_search_ultra_optimized_with_filters(
    nlq: str, 
    max_results: int = 200,
    clinical_trials_only: bool = False,
    exclude_reviews: bool = False,
    year_filter: str = "all"
) -> List[SearchResult]:
    """Ultra-optimized hybrid retrieval pipeline with advanced search filters"""
    
    if not nlq or len(nlq.strip()) < 2:
        raise ValueError("Query too short")

    print(f"DEBUG FILTER: Starting search with filters - CT: {clinical_trials_only}, ER: {exclude_reviews}, Year: {year_filter}, Max: {max_results}")

    # Keep existing pool sizes for quality
    TITLE_POOL_SIZE = int(os.getenv("TITLE_POOL_SIZE", "2000"))
    PER_QUERY_MIN = int(os.getenv("PER_QUERY_MIN", "300"))
    ENABLE_ELINK = os.getenv("ENABLE_ELINK", "1") == "1"
    ELINK_MODES = os.getenv("ELINK_MODES", "similar,citedin").split(",")

    # Two-stage factor: fetch expensive metadata for top N results
    second_stage_size = min(max_results * OPTIMIZED_CONFIG["TWO_STAGE_FACTOR"], 600)

    # Generate queries with filters applied
    queries = generate_smart_queries_with_filters(nlq, clinical_trials_only, exclude_reviews, year_filter)
    print(f"DEBUG FILTER: Generated {len(queries)} filtered queries: {queries}")
    
    strategy_names = ["MeSH/Specific", "Moderate", "Broad", "Natural"][:len(queries)]
    variants = list(zip(strategy_names, queries))

    async with create_optimized_session() as session:
        # STAGE 1: Parallel variant processing
        async def process_single_variant(name_query_pair):
            name, query = name_query_pair
            try:
                count = await esearch_count_async(session, query)
                print(f"DEBUG FILTER: Query '{name}' returned {count} total results")
                
                target_pool = max(TITLE_POOL_SIZE, max_results * 5)
                per_share = max(PER_QUERY_MIN, target_pool // max(1, len(variants)))
                to_get = min(per_share, count) if count > 0 else per_share
                
                pmids = await esearch_ids_paged_async(session, query, to_get, page_size=1000, sort="relevance")
                print(f"DEBUG FILTER: Query '{name}' fetched {len(pmids)} PMIDs")
                await asyncio.sleep(OPTIMIZED_CONFIG["MINIMAL_DELAY"])
                
                return name, query, count, pmids
                
            except Exception as e:
                print(f"DEBUG FILTER: Error processing variant {name}: {e}")
                return name, query, 0, []

        # Execute all variants in parallel
        variant_results = await asyncio.gather(
            *[process_single_variant(variant) for variant in variants],
            return_exceptions=True
        )
        
        # Collect PMIDs
        all_pmids: Set[str] = set()
        pmid_to_strategy = {}
        
        for result in variant_results:
            if isinstance(result, tuple) and len(result) == 4:
                name, query, count, pmids = result
                for pmid in pmids:
                    if pmid not in pmid_to_strategy:
                        pmid_to_strategy[pmid] = name
                all_pmids.update(pmids)

        print(f"DEBUG FILTER: Total unique PMIDs collected: {len(all_pmids)}")

        # Optional ELink expansion (parallel with other operations when possible)
        if ENABLE_ELINK and all_pmids:
            seed = list(all_pmids)[:200]
            elink_tasks = []
            for mode in [m.strip() for m in ELINK_MODES if m.strip()]:
                elink_tasks.append(elink_expand_async(session, seed, mode=mode, max_add=500))
            
            if elink_tasks:
                elink_results = await asyncio.gather(*elink_tasks, return_exceptions=True)
                for extra in elink_results:
                    if isinstance(extra, set):
                        for pmid in extra:
                            if pmid not in pmid_to_strategy:
                                pmid_to_strategy[pmid] = "Similar"
                        all_pmids.update(extra)

        pmid_list = list(all_pmids)
        print(f"DEBUG FILTER: Final PMID list length: {len(pmid_list)}")
        
        if not pmid_list:
            print("DEBUG FILTER: No PMIDs found, returning empty results")
            return []

        # STAGE 2: OPTIMIZED TITLE RANKING - Keep large pool for quality
        titles_by_pmid = await esummary_titles_async_optimized(session, pmid_list)
        print(f"DEBUG FILTER: Fetched titles for {len(titles_by_pmid)} PMIDs")
        
        ranked = tfidf_rank_titles_optimized(nlq, titles_by_pmid)
        print(f"DEBUG FILTER: Ranked {len(ranked)} results by TF-IDF")

        # STAGE 3: TWO-STAGE PROCESSING - Only fetch expensive data for top candidates
        top_pmids = [pmid for pmid, _ in ranked[:second_stage_size]]
        print(f"DEBUG FILTER: Selected top {len(top_pmids)} PMIDs for detailed processing")

        # Fetch expensive metadata in parallel for top candidates only
        print("DEBUG FILTER: Fetching abstracts and citation data...")
        abstract_info_task = fetch_abstracts_batch_optimized(session, top_pmids)
        citation_info_task = fetch_citation_data_batch_optimized(session, top_pmids)
        
        abstract_info, citation_info = await asyncio.gather(
            abstract_info_task, 
            citation_info_task
        )

        print(f"DEBUG FILTER: Retrieved abstract info for {len(abstract_info)} papers")
        print(f"DEBUG FILTER: Retrieved citation info for {len(citation_info)} papers")

        # STAGE 4: Create SearchResult objects and apply additional filtering
        results: List[SearchResult] = []
        skipped_count = 0
        
        for idx, pmid in enumerate(top_pmids, start=1):
            abs_info = abstract_info.get(pmid, {})
            cite_info = citation_info.get(pmid, {})
            
            title = cite_info.get("title", titles_by_pmid.get(pmid, "Unknown"))
            authors = cite_info.get("authors", "Unknown")
            journal = cite_info.get("journal", abs_info.get("journal_title", "Unknown"))
            year = cite_info.get("year", NOW)
            weight = cite_info.get("weight", 0.0)
            abstract = abs_info.get("abstract", "")
            issn = abs_info.get("issn", "")

            # REMOVE THE POST-PROCESSING YEAR FILTER - it's already applied in the query
            # The year filter should be handled by the PubMed query, not post-processing
            # if year_filter != "all":
            #     try:
            #         years_back = int(year_filter)
            #         current_year = dt.datetime.now().year
            #         if year < (current_year - years_back):
            #             skipped_count += 1
            #             continue  # Skip this result
            #     except ValueError:
            #         pass

            result = SearchResult(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                year=year,
                abstract=abstract,
                weight=weight,
                strategy=pmid_to_strategy.get(pmid, "Related"),
                rank=idx,
                journal_impact=0.0,
                issn=issn
            )
            results.append(result)

        print(f"DEBUG FILTER: Created {len(results)} SearchResult objects (skipped {skipped_count})")

        # Fill journal impact (optimized lookup)
        if results and JOURNAL_IMPACTS:
            for r in results:
                r.journal_impact = lookup_journal_impact_optimized(JOURNAL_IMPACTS, r.journal, r.issn)

        # Calculate combined scores
        for result in results:
            result.combined_score = calculate_total_weight(
                result.year, 
                result.weight, 
                result.journal_impact
            )

        final_results = results[:max_results]
        print(f"DEBUG FILTER: Returning {len(final_results)} final results")
        
        return final_results

# ---------- ORIGINAL SEARCH FUNCTION (for backwards compatibility) ----------
async def perform_concentric_search_ultra_optimized(nlq: str, max_results: int = 200) -> List[SearchResult]:
    """Original ultra-optimized search function without filters"""
    return await perform_concentric_search_ultra_optimized_with_filters(
        nlq, max_results, False, False, "all"
    )

def generate_ris_content(results: List[SearchResult]) -> str:
    """Generate RIS format content from search results"""
    ris_lines = []
    
    for result in results:
        ris_lines.append("TY  - JOUR")
        
        if result.title:
            ris_lines.append(f"TI  - {result.title}")
        
        if result.authors and result.authors != "Unknown":
            authors_text = result.authors
            if "," in authors_text:
                ris_lines.append(f"AU  - {authors_text}")
            else:
                ris_lines.append(f"AU  - {authors_text}")
        
        if result.journal and result.journal != "Unknown":
            ris_lines.append(f"JO  - {result.journal}")
        
        if result.year:
            ris_lines.append(f"PY  - {result.year}")
        
        if result.volume:
            ris_lines.append(f"VL  - {result.volume}")
        
        if result.issue:
            ris_lines.append(f"IS  - {result.issue}")
        
        if result.pages:
            if "-" in result.pages:
                start_page, end_page = result.pages.split("-", 1)
                ris_lines.append(f"SP  - {start_page.strip()}")
                ris_lines.append(f"EP  - {end_page.strip()}")
            else:
                ris_lines.append(f"SP  - {result.pages}")
        
        if result.doi:
            ris_lines.append(f"DO  - {result.doi}")
        
        if result.issn:
            ris_lines.append(f"SN  - {result.issn}")
        
        if result.abstract:
            abstract_clean = result.abstract.replace('\n', ' ').replace('\r', ' ')
            ris_lines.append(f"AB  - {abstract_clean}")
        
        ris_lines.append(f"UR  - https://pubmed.ncbi.nlm.nih.gov/{result.pmid}/")
        ris_lines.append(f"N1  - PMID: {result.pmid}")
        ris_lines.append("ER  - ")
        ris_lines.append("")
    
    return "\n".join(ris_lines)

# ---------- Cache Management ----------
def clear_caches():
    """Clear all caches to free memory"""
    global TFIDF_CACHE, SIMILARITY_CACHE, LLM_QUERY_CACHE
    TFIDF_CACHE.clear()
    SIMILARITY_CACHE.clear()
    LLM_QUERY_CACHE.clear()

def get_cache_stats():
    """Get cache statistics"""
    return {
        "tfidf_cache_size": len(TFIDF_CACHE),
        "similarity_cache_size": len(SIMILARITY_CACHE), 
        "llm_query_cache_size": len(LLM_QUERY_CACHE),
        "fuzzy_match_cache_size": len(FUZZY_MATCH_CACHE)
    }


@app.get("/favicon.ico")
async def favicon():
    """Serve a simple favicon to avoid 404s"""
    # Simple book icon for literature search
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect x="20" y="25" width="60" height="50" fill="#2563eb" rx="5"/>
        <rect x="25" y="30" width="50" height="40" fill="white" rx="2"/>
        <line x1="30" y1="40" x2="65" y2="40" stroke="#2563eb" stroke-width="2"/>
        <line x1="30" y1="50" x2="65" y2="50" stroke="#2563eb" stroke-width="2"/>
        <line x1="30" y1="60" x2="55" y2="60" stroke="#2563eb" stroke-width="2"/>
    </svg>"""
    return Response(content=svg_content, media_type="image/svg+xml")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools request to avoid 404s in logs"""
    return {"extensions": []}

@app.get("/debug")
async def debug_files():
    """Debug endpoint to check if files exist"""
    css_content = ""
    css_size = 0
    template_content = ""
    
    try:
        if os.path.exists("static/style.css"):
            with open("static/style.css", "r") as f:
                css_content = f.read()
            css_size = len(css_content)
        
        if os.path.exists("templates/login.html"):
            with open("templates/login.html", "r") as f:
                template_content = f.read()[:200] + "..."
    except Exception as e:
        css_content = f"Error reading: {e}"
    
    files_status = {
        "templates/login.html": os.path.exists("templates/login.html"),
        "templates/search.html": os.path.exists("templates/search.html"), 
        "static/style.css": os.path.exists("static/style.css"),
        "static/app.js": os.path.exists("static/app.js"),
        "css_size_bytes": css_size,
        "css_first_100_chars": css_content[:100] if css_content else "EMPTY",
        "template_preview": template_content,
        "working_directory": os.getcwd()
    }
    return files_status

@app.get("/auth-debug")
async def auth_debug(request: Request, session_token: Optional[str] = Cookie(None)):
    """Debug authentication status"""
    client_ip = get_client_ip(request)
    is_authenticated = await check_authentication(request, session_token)
    
    return {
        "client_ip": client_ip,
        "session_token": session_token[:16] + "..." if session_token else None,
        "is_authenticated": is_authenticated,
        "active_sessions_count": len(AUTHENTICATED_SESSIONS),
        "active_sessions": [token[:16] + "..." for token in AUTHENTICATED_SESSIONS],
        "access_password": ACCESS_PASSWORD
    }

# ---------- CACHE MANAGEMENT ENDPOINTS ----------
@app.get("/cache-stats")
async def cache_stats_endpoint(request: Request, session_token: Optional[str] = Cookie(None)):
    """Get cache statistics"""
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    stats = get_cache_stats()
    return JSONResponse(stats)

@app.post("/clear-caches") 
async def clear_caches_endpoint(request: Request, session_token: Optional[str] = Cookie(None)):
    """Clear all caches"""
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    clear_caches()
    return JSONResponse({"status": "caches cleared"})

# ---------- Helper Functions ----------
def get_active_filters_summary(clinical_trials_only: bool, exclude_reviews: bool, year_filter: str) -> str:
    """Generate a human-readable summary of active filters"""
    filters = []
    
    if clinical_trials_only:
        filters.append("Clinical trials only")
    
    if exclude_reviews:
        filters.append("Reviews excluded")
    
    if year_filter != "all":
        try:
            years = int(year_filter)
            filters.append(f"Past {years} year{'s' if years > 1 else ''}")
        except ValueError:
            pass
    
    if not filters:
        return "No filters applied"
    
    return "Filters: " + ", ".join(filters)

async def require_authentication(request: Request, session_token: Optional[str] = Cookie(None)):
    """Check authentication and redirect to login if not authenticated"""
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        return RedirectResponse(url="/login", status_code=302)
    return None  # Continue with request

# ---------- FastAPI Routes ----------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, session_token: Optional[str] = Cookie(None)):
    is_authenticated = await check_authentication(request, session_token)
    
    if not is_authenticated:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": [],
        "query": "",
        "total_results": 0,
        "search_time": 0,
        "max_results": 50,
        "clinical_trials_only": "false",
        "exclude_reviews": "false",
        "year_filter": "all",
        "mean_score": 0
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, password: str = Form(...)):
    client_ip = get_client_ip(request)
    
    if password == ACCESS_PASSWORD:
        session_token = create_session_token(client_ip)
        AUTHENTICATED_SESSIONS.add(session_token)
        
        response = RedirectResponse(url="/", status_code=302)
        
        # Use secure settings for production
        is_production = ENVIRONMENT == "production"
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=86400 * 7,
            httponly=True,
            secure=is_production,  # Only HTTPS in production
            samesite="strict" if is_production else "lax"
        )
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Incorrect password. Please try again."
        })

@app.post("/logout")
async def logout(request: Request, session_token: Optional[str] = Cookie(None)):
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
    clinical_trials_only: str = Form("false"),
    exclude_reviews: str = Form("false"),
    year_filter: str = Form("all"),
    session_token: Optional[str] = Cookie(None)
):
    is_authenticated = await check_authentication(request, session_token)
    if not is_authenticated:
        return RedirectResponse(url="/login", status_code=302)
    
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")
    
    try:
        start_time = time.time()
        
        # Convert string parameters to boolean
        clinical_trials_filter = clinical_trials_only.lower() == "true"
        exclude_reviews_filter = exclude_reviews.lower() == "true"
        
        # Check if any filters are actually applied
        has_filters = (
            clinical_trials_filter or 
            exclude_reviews_filter or 
            (year_filter != "all" and year_filter.strip())
        )
        
        print(f"DEBUG: Filters - CT: {clinical_trials_filter}, ER: {exclude_reviews_filter}, Year: {year_filter}, Has filters: {has_filters}")
        
        # Use appropriate search function
        if has_filters:
            print("DEBUG: Using filtered search function")
            results = await perform_concentric_search_ultra_optimized_with_filters(
                query.strip(), 
                max_results,
                clinical_trials_filter,
                exclude_reviews_filter,
                year_filter
            )
        else:
            print("DEBUG: Using original search function")
            # Use the original working function when no filters are applied
            results = await perform_concentric_search_ultra_optimized(query.strip(), max_results)
        
        search_time = time.time() - start_time
        print(f"DEBUG: Found {len(results)} results in {search_time:.2f}s")
        
        # Sort results by total score (highest first) by default
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Convert SearchResult objects to dictionaries using the to_dict() method
        results_dict = [result.to_dict() for result in results]
        
        return templates.TemplateResponse("search.html", {
            "request": request,
            "results": results_dict,
            "query": query,
            "total_results": len(results),
            "search_time": search_time,
            "max_results": max_results,
            "clinical_trials_only": clinical_trials_only,
            "exclude_reviews": exclude_reviews,
            "year_filter": year_filter,
            "mean_score": round(sum(r.combined_score for r in results) / len(results)) if results else 0
        })
        
    except ValueError as e:
        print(f"DEBUG: ValueError: {e}")
        return templates.TemplateResponse("search.html", {
            "request": request,
            "results": [],
            "query": query,
            "total_results": 0,
            "search_time": 0,
            "max_results": max_results,
            "clinical_trials_only": clinical_trials_only,
            "exclude_reviews": exclude_reviews,
            "year_filter": year_filter,
            "error": str(e),
            "mean_score": 0
        })

    except Exception as e:
        print(f"DEBUG: Exception: {e}")
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("search.html", {
            "request": request,
            "results": [],
            "query": query,
            "total_results": 0,
            "search_time": 0,
            "max_results": max_results,
            "clinical_trials_only": clinical_trials_only,
            "exclude_reviews": exclude_reviews,
            "year_filter": year_filter,
            "error": f"Search failed: {str(e)}",
            "mean_score": 0
        })

@app.post("/analyze")
async def analyze_endpoint(
    request: Request,
    query: str = Form(...), 
    results_json: str = Form(...),
    session_token: Optional[str] = Cookie(None)
):
    # Check auth and redirect if needed
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API not configured")
        
        results_data = json.loads(results_json)
        
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
        
        analysis_result = await analyze_with_ai(results, query)
        
        if not analysis_result.get("success", False):
            raise HTTPException(status_code=500, detail=analysis_result.get("error", "AI analysis failed"))
        
        ranked_pmids = analysis_result["ranked_pmids"]
        pmid_to_ai_rank = {pmid: idx + 1 for idx, pmid in enumerate(ranked_pmids)}
        
        for result in results:
            result.ai_rank = pmid_to_ai_rank.get(result.pmid, None)
        
        results.sort(key=lambda x: (x.ai_rank or 999, x.rank))
        
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
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/download-ris")
async def download_ris_endpoint(
    request: Request,
    query: str = Form(...), 
    max_results: int = Form(100),
    clinical_trials_only: str = Form("false"),
    exclude_reviews: str = Form("false"), 
    year_filter: str = Form("all"),
    session_token: Optional[str] = Cookie(None)
):
    # Check auth and redirect if needed
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    try:
        if not query or len(query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Query too short")
        
        # Convert string parameters to boolean
        clinical_trials_filter = clinical_trials_only.lower() == "true"
        exclude_reviews_filter = exclude_reviews.lower() == "true"
        
        # Check if any filters are actually applied
        has_filters = (
            clinical_trials_filter or 
            exclude_reviews_filter or 
            (year_filter != "all" and year_filter.strip())
        )
        
        # Use appropriate search function
        if has_filters:
            results = await perform_concentric_search_ultra_optimized_with_filters(
                query.strip(), 
                max_results,
                clinical_trials_filter,
                exclude_reviews_filter,
                year_filter
            )
        else:
            # Use the original working function when no filters are applied
            results = await perform_concentric_search_ultra_optimized(query.strip(), max_results)
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        
        pmids = [r.pmid for r in results]
        
        async with create_optimized_session() as session:
            detailed_metadata = await fetch_detailed_metadata(session, pmids)
        
        for result in results:
            if result.pmid in detailed_metadata:
                meta = detailed_metadata[result.pmid]
                result.volume = meta.get('volume', '')
                result.issue = meta.get('issue', '')
                result.pages = meta.get('pages', '')
                result.doi = meta.get('doi', '')
        
        ris_content = generate_ris_content(results)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include filter info in filename only if filters are applied
        if has_filters:
            filter_info = []
            if clinical_trials_filter:
                filter_info.append("clinical-trials")
            if exclude_reviews_filter:
                filter_info.append("no-reviews")
            if year_filter != "all":
                filter_info.append(f"past-{year_filter}y")
            
            filter_suffix = f"_{'_'.join(filter_info)}" if filter_info else ""
            filename = f"pubmed_results{filter_suffix}_{timestamp}.ris"
        else:
            filename = f"pubmed_results_{timestamp}.ris"
        
        return Response(
            content=ris_content,
            media_type="application/x-research-info-systems",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/x-research-info-systems; charset=utf-8"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate RIS file: {str(e)}")

@app.post("/download-synthesis-text")
async def download_synthesis_text_endpoint(
    request: Request,
    query: str = Form(...),
    synthesis_text: str = Form(...),
    session_token: Optional[str] = Cookie(None)
):
    """Download just the synthesis text file"""
    # Check auth and redirect if needed
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    text_content = f"""AI Literature Synthesis
Query: {query}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SYNTHESIS:
{synthesis_text}

Generated by PubMed (Taylor's Version)
"""
    
    return Response(
        content=text_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=synthesis_{timestamp}.txt",
            "Content-Type": "text/plain; charset=utf-8"
        }
    )

@app.post("/download-synthesis-ris")
async def download_synthesis_ris_endpoint(
    request: Request,
    cited_pmids: str = Form(...),
    session_token: Optional[str] = Cookie(None)
):
    """Download RIS file for cited papers"""
    # Check auth and redirect if needed
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    try:
        pmids_list = json.loads(cited_pmids)
        
        if not pmids_list:
            raise HTTPException(status_code=400, detail="No cited papers found")
        
        async with create_optimized_session() as session:
            citation_info = await fetch_citation_data_batch_optimized(session, pmids_list)
            abstract_info = await fetch_abstracts_batch_optimized(session, pmids_list)
            detailed_metadata = await fetch_detailed_metadata(session, pmids_list)
        
        # Create SearchResult objects and generate RIS
        cited_results = []
        for idx, pmid in enumerate(pmids_list, start=1):
            cite_info = citation_info.get(pmid, {})
            abs_info = abstract_info.get(pmid, {})
            meta_info = detailed_metadata.get(pmid, {})
            
            result = SearchResult(
                pmid=pmid,
                title=cite_info.get("title", "Unknown"),
                authors=cite_info.get("authors", "Unknown"),
                journal=cite_info.get("journal", abs_info.get("journal_title", "Unknown")),
                year=cite_info.get("year", NOW),
                abstract=abs_info.get("abstract", ""),
                weight=cite_info.get("weight", 0.0),
                strategy="Cited in Synthesis",
                rank=idx,
                journal_impact=0.0,
                issn=abs_info.get("issn", ""),
                volume=meta_info.get("volume", ""),
                issue=meta_info.get("issue", ""),
                pages=meta_info.get("pages", ""),
                doi=meta_info.get("doi", "")
            )
            
            if JOURNAL_IMPACTS:
                result.journal_impact = lookup_journal_impact_optimized(
                    JOURNAL_IMPACTS, result.journal, result.issn
                )
            
            cited_results.append(result)
        
        ris_content = generate_ris_content(cited_results)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return Response(
            content=ris_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=cited_references_{timestamp}.ris",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid cited PMIDs data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate RIS file: {str(e)}")

# Also update the cache management endpoints:
@app.get("/cache-stats")
async def cache_stats_endpoint(request: Request, session_token: Optional[str] = Cookie(None)):
    """Get cache statistics"""
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    stats = get_cache_stats()
    return JSONResponse(stats)

@app.post("/clear-caches") 
async def clear_caches_endpoint(request: Request, session_token: Optional[str] = Cookie(None)):
    """Clear all caches"""
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    clear_caches()
    return JSONResponse({"status": "caches cleared"})

# Optional: Add a catch-all for any GET requests to protected endpoints
@app.get("/search")
async def search_get_redirect(request: Request, session_token: Optional[str] = Cookie(None)):
    """Redirect GET requests to /search to the main page"""
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    return RedirectResponse(url="/", status_code=302)

# Catch-all route for undefined endpoints
@app.get("/{path:path}")
async def catch_all(request: Request, path: str, session_token: Optional[str] = Cookie(None)):
    """Catch-all route for undefined endpoints"""
    # Check if user is authenticated
    auth_redirect = await require_authentication(request, session_token)
    if auth_redirect:
        return auth_redirect
    
    # If authenticated, redirect to home page
    return RedirectResponse(url="/", status_code=302)

# ---------- Main ----------
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)