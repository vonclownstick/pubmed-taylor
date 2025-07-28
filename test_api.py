#!/usr/bin/env python3
"""
Quick PubMed API Status Checker
"""
import requests
import json
import time

def check_pubmed_api():
    """Test if PubMed API is responding"""
    
    print("🔍 Checking PubMed E-utilities API status...")
    print("=" * 50)
    
    # Test 1: Simple ESearch
    print("1. Testing ESearch (basic search)...")
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": "cancer",
            "retmode": "json",
            "retmax": 5
        }
        
        start_time = time.time()
        response = requests.get(url, params=params, timeout=10)
        response_time = time.time() - start_time
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            count = data.get("esearchresult", {}).get("count", 0)
            ids = data.get("esearchresult", {}).get("idlist", [])
            print(f"   ✅ Success! Found {count} results")
            print(f"   Sample PMIDs: {ids[:3]}")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 2: ESummary
    print("2. Testing ESummary (get article details)...")
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": "38295446",  # A recent PMID
            "retmode": "json"
        }
        
        start_time = time.time()
        response = requests.get(url, params=params, timeout=10)
        response_time = time.time() - start_time
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {})
            if "38295446" in result:
                title = result["38295446"].get("title", "No title")
                print(f"   ✅ Success! Retrieved article")
                print(f"   Title: {title[:80]}...")
            else:
                print(f"   ⚠️  Unexpected response format")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 3: EFetch
    print("3. Testing EFetch (get full records)...")
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": "38295446",
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        start_time = time.time()
        response = requests.get(url, params=params, timeout=15)
        response_time = time.time() - start_time
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.2f} seconds")
        
        if response.status_code == 200:
            xml_content = response.text
            if "PubmedArticle" in xml_content and len(xml_content) > 500:
                print(f"   ✅ Success! Retrieved XML ({len(xml_content)} chars)")
            else:
                print(f"   ⚠️  Unexpected XML content length: {len(xml_content)}")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 4: iCite API (for citation data)
    print("4. Testing iCite API (citation data)...")
    try:
        url = "https://icite.od.nih.gov/api/pubs"
        params = {"pmids": "38295446"}
        
        start_time = time.time()
        response = requests.get(url, params=params, timeout=10)
        response_time = time.time() - start_time
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                citations = data["data"][0].get("citation_count", 0)
                print(f"   ✅ Success! Citations: {citations}")
            else:
                print(f"   ⚠️  No data returned")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("=" * 50)
    print("🏁 API Check Complete!")

if __name__ == "__main__":
    check_pubmed_api()