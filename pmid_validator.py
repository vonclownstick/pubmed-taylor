#!/usr/bin/env python3
"""
PMID Validator for AI-generated summaries
Extracted from refcheck1.py and adapted for the PubMed search app
Enhanced with detailed error handling and reporting
"""

import re
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Set
import requests
import asyncio

# Handle aiohttp import gracefully
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# PMID extraction regex
PMID_SERIES_RE = re.compile(r'\bPMID\s*:?\s*([0-9]{1,8}(?:\s*,\s*[0-9]{1,8})*)', re.IGNORECASE)

# Configuration
MAX_CONTEXT_CHARS = 1200
MAX_TITLE_CHARS = 500
MAX_ABSTRACT_CHARS = 2500
LLM_BATCH_SIZE = 20
MAX_CONCURRENT_REQUESTS = 5

# Validation settings
STRICT_VALIDATION = False  # Set to True for stricter validation

class ValidationError(Exception):
    """Custom exception for validation errors with detailed context"""
    def __init__(self, message: str, error_type: str, pmid: str = None, details: Dict = None, exception: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.pmid = pmid
        self.details = details or {}
        self.exception = exception

class SummaryValidator:
    """Validates PMID citations in AI-generated summaries"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", custom_requirements: str = ""):
        self.openai_api_key = openai_api_key
        
        # Validate model name
        valid_models = ["gpt-4o-mini"]
        if model not in valid_models:
            print(f"[VALIDATOR WARNING] Unknown model '{model}', using 'gpt-4o-mini'")
            model = "gpt-4o-mini"
        
        self.model = model
        self._llm_cache = {}
        self.debug = False
        self.custom_requirements = custom_requirements 
        
    def extract_pmids_from_text(self, text: str) -> List[str]:
        """Extract all PMIDs from text and return as list of strings"""
        pmids = []
        for match in PMID_SERIES_RE.finditer(text):
            group = match.group(1)
            # Split on commas and clean each PMID
            nums = [re.sub(r'\D', '', n.strip()) for n in group.split(",")]
            pmids.extend([n for n in nums if n and n.isdigit()])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pmids = []
        for pmid in pmids:
            if pmid not in seen:
                seen.add(pmid)
                unique_pmids.append(pmid)
        
        return unique_pmids

    def normalize_pmid_formatting(self, text: str) -> str:
        """Convert semicolon-separated PMIDs to comma-separated"""
        def fix_pmid_format(match):
            full_match = match.group(0)
            pmid_part = match.group(1)
            # Replace semicolons with commas in the PMID series
            fixed_pmids = re.sub(r'\s*;\s*', ', ', pmid_part)
            return full_match.replace(pmid_part, fixed_pmids)
        
        # Apply the fix
        normalized = PMID_SERIES_RE.sub(fix_pmid_format, text)
        return normalized
    
    def check_pmids_exist_in_results(self, pmids: List[str], search_results: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Check which PMIDs exist in our search results
        Returns: (existing_pmids, missing_pmids)
        """
        result_pmids = {str(result['pmid']) for result in search_results}
        
        existing = []
        missing = []
        
        for pmid in pmids:
            if pmid in result_pmids:
                existing.append(pmid)
            else:
                missing.append(pmid)
        
        return existing, missing
    
    def get_pmid_data(self, pmid: str, search_results: List[Dict]) -> Optional[Dict]:
        """Get title and abstract for a PMID from search results"""
        for result in search_results:
            if str(result['pmid']) == pmid:
                return {
                    'title': result.get('title', ''),
                    'abstract': result.get('abstract', ''),
                    'pmid': pmid
                }
        return None
    
    def extract_citation_contexts(self, text: str) -> List[Dict]:
        """
        Extract context around each PMID citation
        Returns list of {'pmid': str, 'context': str, 'sentence': str}
        """
        contexts = []
        seen_pmids = set()  # Track which PMIDs we've already processed
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent_idx, sentence in enumerate(sentences):
            # Find PMIDs in this sentence
            for match in PMID_SERIES_RE.finditer(sentence):
                group = match.group(1)
                nums = [re.sub(r'\D', '', n.strip()) for n in re.split(r'[,;]', group)]
                
                for pmid in nums:
                    if pmid and pmid.isdigit() and pmid not in seen_pmids:
                        seen_pmids.add(pmid)  # Mark as processed
                        
                        # Create larger context window (current sentence + 2 before + 2 after)
                        context_sentences = sentences[max(0, sent_idx-2):min(len(sentences), sent_idx+3)]
                        context = ' '.join(context_sentences).strip()
                        
                        if len(context) > MAX_CONTEXT_CHARS:
                            context = context[:MAX_CONTEXT_CHARS] + "..."
                        
                        contexts.append({
                            'pmid': pmid,
                            'context': context,
                            'sentence': sentence.strip()
                        })
        
        return contexts
    
    def _create_context_hash(self, context: str, title: str, abstract: str) -> str:
        """Create hash for caching LLM responses"""
        content = f"{context[:500]}|{title[:200]}|{abstract[:1000]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_error_result(self, pmid: str, error_type: str, error_details: str, exception: Exception = None) -> Dict:
        """Create a standardized error result with detailed information"""
        base_reason = f"Validation failed ({error_type}): {error_details}"
        
        if exception:
            exception_info = f" | Exception: {type(exception).__name__}: {str(exception)}"
            base_reason += exception_info
        
        if hasattr(self, 'debug') and self.debug:
            print(f"[VALIDATOR DEBUG] Creating error result for PMID {pmid}: {base_reason}")
        
        return {
            'pmid': pmid,
            'valid': True,  # Default to valid on errors to avoid removing valid citations
            'reason': base_reason,
            'error_type': error_type
        }
    
    def _validate_input_data(self, validation_data: List[Dict]) -> List[str]:
        """Validate input data and return list of error messages"""
        errors = []
        
        for i, data in enumerate(validation_data):
            if not data.get('pmid'):
                errors.append(f"Item {i}: Missing PMID")
            elif not str(data['pmid']).isdigit():
                errors.append(f"Item {i}: Invalid PMID format '{data['pmid']}'")
            
            if not data.get('context', '').strip():
                errors.append(f"Item {i} (PMID {data.get('pmid', 'unknown')}): Empty context")
            
            if not data.get('title', '').strip():
                errors.append(f"Item {i} (PMID {data.get('pmid', 'unknown')}): Empty title")
            
            # Abstract can be empty, but if present should not be just whitespace
            abstract = data.get('abstract', '')
            if abstract and not abstract.strip():
                errors.append(f"Item {i} (PMID {data.get('pmid', 'unknown')}): Abstract is whitespace only")
        
        return errors
    
    async def validate_pmid_contexts_batch(self, validation_data: List[Dict]) -> List[Dict]:
        """
        Validate multiple PMID contexts using LLM
        validation_data: List of{'pmid': str, 'context': str, 'title': str, 'abstract': str}
        Returns: List of {'pmid': str, 'valid': bool, 'reason': str}
        """
        if not self.openai_api_key:
            return [self._create_error_result(
                data['pmid'], 
                'missing_api_key', 
                'No OpenAI API key provided'
            ) for data in validation_data]
        
        # Validate input data
        input_errors = self._validate_input_data(validation_data)
        if input_errors:
            print(f"[VALIDATOR ERROR] Input validation failed: {'; '.join(input_errors)}")
            return [self._create_error_result(
                data.get('pmid', 'unknown'),
                'invalid_input',
                f"Input validation failed: {'; '.join(input_errors)}"
            ) for data in validation_data]
        
        # Create a mapping to track all PMIDs and their first occurrence
        unique_pmids = []
        seen_pmids = set()
        
        for data in validation_data:
            pmid = data['pmid']
            
            # Only process first occurrence of each PMID
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_pmids.append(pmid)
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"[VALIDATOR DEBUG] Added unique PMID {pmid} for processing")
            elif hasattr(self, 'debug') and self.debug:
                print(f"[VALIDATOR DEBUG] Skipping duplicate PMID {pmid}")
        
        print(f"[VALIDATOR] Processing {len(unique_pmids)} unique PMIDs from {len(validation_data)} total entries")
        
        # Process unique PMIDs
        final_results = []
        
        # Check cache and prepare for LLM processing
        cached_results = {}
        pmids_to_process = []
        
        for pmid in unique_pmids:
            # Find the data for this PMID (use first occurrence)
            data = None
            for item in validation_data:
                if item['pmid'] == pmid:
                    data = item
                    break
            
            if not data:
                final_results.append(self._create_error_result(
                    pmid, 'missing_data', 'Could not find data for PMID'
                ))
                continue
            
            try:
                context_hash = self._create_context_hash(
                    data['context'], data['title'], data['abstract']
                )
                
                if context_hash in self._llm_cache:
                    cached_result = self._llm_cache[context_hash]
                    final_results.append(cached_result)
                    if hasattr(self, 'debug') and self.debug:
                        print(f"[VALIDATOR DEBUG] Cache hit for PMID {pmid}")
                else:
                    pmids_to_process.append({
                        'pmid': pmid,
                        'context': data['context'],
                        'title': data['title'], 
                        'abstract': data['abstract'],
                        'hash': context_hash
                    })
            except Exception as e:
                final_results.append(self._create_error_result(
                    pmid, 'hash_generation', 'Failed to generate cache hash', e
                ))
        
        print(f"[VALIDATOR] LLM cache: {len(final_results)} hits, {len(pmids_to_process)} misses")
        
        # Process uncached PMIDs in batches
        if pmids_to_process:
            try:
                batches = [pmids_to_process[i:i+LLM_BATCH_SIZE] for i in range(0, len(pmids_to_process), LLM_BATCH_SIZE)]
                
                for batch_idx, batch in enumerate(batches):
                    try:
                        print(f"[VALIDATOR] Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} items")
                        batch_results = await self._process_validation_batch(batch)
                        
                        # Add batch results to final results
                        for pmid_data, result in zip(batch, batch_results):
                            final_results.append(result)
                            # Cache the result
                            self._llm_cache[pmid_data['hash']] = result
                                    
                    except ValidationError as ve:
                        print(f"[VALIDATOR ERROR] Validation error in batch {batch_idx + 1}: {ve}")
                        # Mark all in this batch with specific validation error
                        for pmid_data in batch:
                            final_results.append(self._create_error_result(
                                pmid_data['pmid'], ve.error_type, str(ve), ve
                            ))
                    except Exception as e:
                        print(f"[VALIDATOR ERROR] Unexpected error in batch {batch_idx + 1}: {e}")
                        # Mark all in this batch with generic error
                        for pmid_data in batch:
                            final_results.append(self._create_error_result(
                                pmid_data['pmid'], 'batch_processing_error', 
                                f"Unexpected error in batch processing", e
                            ))
                        
            except Exception as e:
                print(f"[VALIDATOR ERROR] Error in batch management: {e}")
                # Fallback: mark all as error
                for pmid_data in pmids_to_process:
                    final_results.append(self._create_error_result(
                        pmid_data['pmid'], 'batch_management_error',
                        f'Batch management failed', e
                    ))
        
        # Debug output for all results
        if hasattr(self, 'debug') and self.debug:
            for result in final_results:
                error_indicator = f" [{result.get('error_type', '')}]" if 'error_type' in result else ""
                print(f"[VALIDATOR DEBUG] PMID {result['pmid']}: {'VALID' if result['valid'] else 'INVALID'} - {result['reason']}{error_indicator}")
        
        return final_results
    
    async def _process_validation_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of validations using OpenAI API
        Returns: List of result dictionaries, one per batch item
        """
        
        # Debug: Print batch structure
        if hasattr(self, 'debug') and self.debug:
            print(f"[VALIDATOR DEBUG] Processing batch with {len(batch)} items")
            for i, item in enumerate(batch):
                print(f"  Item {i}: PMID {item.get('pmid', 'unknown')}")
        
        def sanitize_text(text: str) -> str:
            """Clean text to prevent API issues"""
            if not text:
                return ""
            # Remove control characters and excessive whitespace
            import re
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control chars
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            return text.strip()
        
        # Validate batch data
        if not batch:
            raise ValidationError("Empty batch provided", "empty_batch")
        
        for data in batch:
            if not data.get('pmid'):
                raise ValidationError(f"Missing PMID in batch data", "missing_pmid")
        
        system_msg = (
            "You are a scientific fact-checker reviewing citations in a literature synthesis. "
            "For each citation, you'll receive the surrounding context, article title, and abstract.\n\n"
            "Your task: Determine if the citation is appropriate and accurate.\n\n"
            "VALIDATION CRITERIA:\n"
            "- Mark as VALID if the cited paper is relevant to the topic discussed\n"
            "- Mark as VALID if the paper supports or relates to the general theme, even if not perfectly specific\n"
            "- Mark as INVALID only if the paper is clearly unrelated or contradicts the claim\n"
            "- When in doubt, lean toward VALID rather than rejecting appropriate citations\n"
            "- Consider that literature syntheses often cite papers for general background or related concepts\n\n"
            "Respond with exactly this JSON format:\n"
            '{"citations": [{"id": 1, "valid": true, "reason": "brief explanation"}, ...]}'
        )
        
        user_parts = []
        total_chars = 0
        processed_batch = []  # Track which items we include in the prompt
        
        for idx, data in enumerate(batch, 1):
            try:
                # Sanitize and truncate inputs
                context = sanitize_text(data.get("context", ""))[:MAX_CONTEXT_CHARS]
                title = sanitize_text(data.get("title", ""))[:MAX_TITLE_CHARS] 
                abstract = sanitize_text(data.get("abstract", ""))[:MAX_ABSTRACT_CHARS]
                pmid = str(data.get("pmid", "")).strip()
                
                # Validate that we have content
                if not context:
                    raise ValidationError(f"Empty context for PMID {pmid}", "empty_context", pmid)
                if not title:
                    raise ValidationError(f"Empty title for PMID {pmid}", "empty_title", pmid)

                user_part = f"""Citation {idx} (PMID: {pmid}):

Context from synthesis:
{context}

Cited article title:
{title}

Cited article abstract:
{abstract}

"""
                user_parts.append(user_part)
                total_chars += len(user_part)
                processed_batch.append(data)
                
                # Check if we're approaching token limits (rough estimate: 1 token ≈ 4 chars)
                if total_chars > 12000:  # Stay well under typical limits
                    print(f"[VALIDATOR] Truncating batch at {idx} items due to size limits")
                    break
                    
            except Exception as e:
                raise ValidationError(
                    f"Error processing batch item {idx} (PMID {data.get('pmid', 'unknown')}): {str(e)}", 
                    "batch_item_processing", 
                    data.get('pmid')
                )
        
        user_msg = "\n".join(user_parts)
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": min(1500, 150 * len(processed_batch)),  # Scale tokens appropriately
        }
        
        # Add response_format only if the model supports it
        if self.model in ["gpt-5-mini", "gpt-4o-mini", "gpt-4-turbo"]:
            payload["response_format"] = {"type": "json_object"}
        
        batch_results = []
        
        try:
            # Log the request for debugging
            if hasattr(self, 'debug') and self.debug:
                print(f"[VALIDATOR DEBUG] Sending batch of {len(processed_batch)} validations")
                print(f"[VALIDATOR DEBUG] Total prompt length: {len(user_msg)} chars")
                print(f"[VALIDATOR DEBUG] Model: {self.model}")
            
            if AIOHTTP_AVAILABLE:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=120)
                        ) as resp:
                            response_text = await resp.text()
                            
                            if resp.status == 401:
                                raise ValidationError("Invalid OpenAI API key", "invalid_api_key")
                            elif resp.status == 429:
                                raise ValidationError("OpenAI API rate limit exceeded", "rate_limit_exceeded", 
                                                    details={"status_code": resp.status, "response": response_text})
                            elif resp.status == 400:
                                raise ValidationError("Bad request to OpenAI API", "bad_request",
                                                    details={"status_code": resp.status, "response": response_text})
                            elif resp.status >= 500:
                                raise ValidationError("OpenAI API server error", "server_error",
                                                    details={"status_code": resp.status, "response": response_text})
                            elif resp.status != 200:
                                raise ValidationError(f"OpenAI API returned status {resp.status}", "api_error",
                                                    details={"status_code": resp.status, "response": response_text})
                            
                            try:
                                response_data = json.loads(response_text)
                            except json.JSONDecodeError as je:
                                raise ValidationError("Failed to parse OpenAI API response as JSON", "json_parse_error",
                                                    details={"response": response_text[:500], "json_error": str(je)})
                            
                            if 'choices' not in response_data or not response_data['choices']:
                                raise ValidationError("No choices in OpenAI API response", "missing_choices",
                                                    details={"response": response_data})
                                
                            content = response_data["choices"][0]["message"]["content"].strip()
                            
                except asyncio.TimeoutError:
                    raise ValidationError("OpenAI API request timed out", "timeout")
                except aiohttp.ClientError as ce:
                    raise ValidationError(f"HTTP client error: {str(ce)}", "http_client_error", exception=ce)
            else:
                # Fallback to synchronous requests
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=120
                    )
                    
                    if response.status_code == 401:
                        raise ValidationError("Invalid OpenAI API key", "invalid_api_key")
                    elif response.status_code == 429:
                        raise ValidationError("OpenAI API rate limit exceeded", "rate_limit_exceeded",
                                            details={"status_code": response.status_code, "response": response.text})
                    elif response.status_code == 400:
                        raise ValidationError("Bad request to OpenAI API", "bad_request",
                                            details={"status_code": response.status_code, "response": response.text})
                    elif response.status_code >= 500:
                        raise ValidationError("OpenAI API server error", "server_error",
                                            details={"status_code": response.status_code, "response": response.text})
                    elif response.status_code != 200:
                        raise ValidationError(f"OpenAI API returned status {response.status_code}", "api_error",
                                            details={"status_code": response.status_code, "response": response.text})
                    
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError as je:
                        raise ValidationError("Failed to parse OpenAI API response as JSON", "json_parse_error",
                                            details={"response": response.text[:500], "json_error": str(je)})
                    
                    if 'choices' not in response_data or not response_data['choices']:
                        raise ValidationError("No choices in OpenAI API response", "missing_choices",
                                            details={"response": response_data})
                    
                    content = response_data["choices"][0]["message"]["content"].strip()
                    
                except requests.exceptions.Timeout:
                    raise ValidationError("OpenAI API request timed out", "timeout")
                except requests.exceptions.RequestException as re:
                    raise ValidationError(f"HTTP request error: {str(re)}", "http_request_error", exception=re)
            
            # Parse the LLM response
            try:
                llm_data = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        llm_data = json.loads(json_match.group(0))
                    except json.JSONDecodeError as je:
                        raise ValidationError("Could not parse JSON from LLM response", "llm_json_parse_error",
                                            details={"content": content[:500], "json_error": str(je)})
                else:
                    raise ValidationError("No JSON found in LLM response", "no_json_in_response",
                                        details={"content": content[:500]})
            
            citations = llm_data.get("citations", [])
            
            if not citations:
                raise ValidationError("No citations found in LLM response", "no_citations_in_response",
                                    details={"llm_response": llm_data})
            
            if len(citations) != len(processed_batch):
                print(f"[VALIDATOR WARNING] Expected {len(processed_batch)} citations, got {len(citations)}")
            
            # Process each citation result and create results
            for idx, citation_result in enumerate(citations):
                try:
                    if idx >= len(processed_batch):
                        print(f"[VALIDATOR WARNING] Citation {idx + 1} exceeds batch size {len(processed_batch)}, skipping")
                        continue
                        
                    data_item = processed_batch[idx]
                    
                    # Validate citation result structure
                    if not isinstance(citation_result, dict):
                        raise ValidationError(f"Citation {idx + 1} is not a dictionary", "invalid_citation_format",
                                            details={"citation": citation_result})
                    
                    valid = citation_result.get("valid")
                    reason = citation_result.get("reason", "").strip()
                    
                    if valid is None:
                        print(f"[VALIDATOR WARNING] Missing 'valid' field for citation {idx + 1}, defaulting to True")
                        valid = True
                    
                    if not reason:
                        reason = "Valid citation" if valid else "Invalid citation"
                    
                    result = {
                        'pmid': data_item['pmid'],
                        'valid': bool(valid),
                        'reason': reason or ("Valid citation" if valid else "Invalid citation")
                    }
                    
                    batch_results.append(result)
                    
                    if hasattr(self, 'debug') and self.debug:
                        print(f"[VALIDATOR DEBUG] Batch item {idx + 1}: PMID {data_item['pmid']}: {'VALID' if valid else 'INVALID'} - {reason}")
                        
                except KeyError as ke:
                    print(f"[VALIDATOR ERROR] KeyError processing citation {idx + 1}: {ke}")
                    print(f"[VALIDATOR ERROR] Available keys in data_item: {list(processed_batch[idx].keys()) if idx < len(processed_batch) else 'N/A'}")
                    # Create error result for this specific citation
                    if idx < len(processed_batch):
                        batch_results.append({
                            'pmid': processed_batch[idx].get('pmid', f'unknown_{idx}'),
                            'valid': True,
                            'reason': f"KeyError accessing '{ke}': {str(ke)}"
                        })
                except Exception as e:
                    print(f"[VALIDATOR ERROR] Error processing citation {idx + 1}: {e}")
                    print(f"[VALIDATOR ERROR] Error type: {type(e).__name__}")
                    # Create error result for this specific citation
                    if idx < len(processed_batch):
                        batch_results.append({
                            'pmid': processed_batch[idx].get('pmid', f'unknown_{idx}'),
                            'valid': True,
                            'reason': f"Citation processing error: {str(e)}"
                        })
        
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Convert unexpected errors to validation errors
            print(f"[VALIDATOR ERROR] Raw exception in LLM processing: {type(e).__name__}: {e}")
            import traceback
            print(f"[VALIDATOR ERROR] Traceback: {traceback.format_exc()}")
            raise ValidationError(f"Unexpected error during LLM validation: {str(e)}", "unexpected_llm_error", exception=e)
        
        # Ensure we have results for all processed items
        while len(batch_results) < len(processed_batch):
            idx = len(batch_results)
            batch_results.append({
                'pmid': processed_batch[idx]['pmid'],
                'valid': True,
                'reason': "No LLM response received for this item - defaulting to valid"
            })
        
        return batch_results
    
    async def validate_summary(self, summary_text: str, search_results: List[Dict]) -> Dict:
        """
        Validate all PMIDs in a summary text
        """
        print(f"[VALIDATOR] Validating summary with {len(search_results)} search results")
        
        # First, normalize semicolons to commas
        normalized_text = self.normalize_pmid_formatting(summary_text)
        if normalized_text != summary_text:
            print("[VALIDATOR] Fixed semicolon-separated PMIDs to comma-separated")
        
        # Extract PMIDs from normalized summary
        cited_pmids = self.extract_pmids_from_text(normalized_text)
        print(f"[VALIDATOR] Found {len(cited_pmids)} unique cited PMIDs: {cited_pmids}")
        
        if not cited_pmids:
            return {
                'valid': True,
                'missing_pmids': [],
                'invalid_pmids': [],
                'validation_details': [],
                'normalized_text': normalized_text
            }
        
        # Check which PMIDs exist in our search results
        existing_pmids, missing_pmids = self.check_pmids_exist_in_results(cited_pmids, search_results)
        print(f"[VALIDATOR] Existing: {len(existing_pmids)}, Missing: {len(missing_pmids)}")
        
        if missing_pmids:
            print(f"[VALIDATOR] Missing PMIDs: {missing_pmids}")
        
        # For existing PMIDs, validate context
        invalid_pmids = []
        validation_details = []
        
        if existing_pmids:
            # Extract contexts for each citation from normalized text
            contexts = self.extract_citation_contexts(normalized_text)
            print(f"[VALIDATOR] Extracted {len(contexts)} unique contexts for validation")
            
            # Build validation data - only for existing PMIDs
            validation_data = []
            for context_info in contexts:
                if context_info['pmid'] in existing_pmids:
                    pmid_data = self.get_pmid_data(context_info['pmid'], search_results)
                    if pmid_data:
                        validation_data.append({
                            'pmid': context_info['pmid'],
                            'context': context_info['context'],
                            'title': pmid_data['title'],
                            'abstract': pmid_data['abstract']
                        })
            
            print(f"[VALIDATOR] Validating {len(validation_data)} contexts with LLM")
            
            if validation_data:
                validation_results = await self.validate_pmid_contexts_batch(validation_data)
                
                for result in validation_results:
                    validation_details.append(result)
                    
                    # Only mark as invalid if strict validation is on, or if clearly problematic
                    if not result['valid']:
                        if STRICT_VALIDATION:
                            invalid_pmids.append(result['pmid'])
                        else:
                            # In non-strict mode, only remove if reason indicates clear problems
                            reason_lower = result['reason'].lower()
                            clear_problems = ['unrelated', 'contradicts', 'wrong', 'incorrect', 'not relevant', 'different topic']
                            if any(problem in reason_lower for problem in clear_problems):
                                invalid_pmids.append(result['pmid'])
                                if hasattr(self, 'debug') and self.debug:
                                    print(f"[VALIDATOR DEBUG] Removing PMID {result['pmid']} - clearly problematic")
                            else:
                                if hasattr(self, 'debug') and self.debug:
                                    print(f"[VALIDATOR DEBUG] Keeping questionable PMID {result['pmid']} - not clearly wrong")
                
                print(f"[VALIDATOR] Invalid contexts: {len(invalid_pmids)}")
        
        # Remove duplicates
        invalid_pmids = list(set(invalid_pmids))
        
        is_valid = len(missing_pmids) == 0 and len(invalid_pmids) == 0
        
        return {
            'valid': is_valid,
            'missing_pmids': missing_pmids,
            'invalid_pmids': invalid_pmids,
            'validation_details': validation_details,
            'normalized_text': normalized_text
        }
    
    async def clean_summary_iteratively(self, initial_summary: str, search_results: List[Dict], 
                                      max_iterations: int = 3) -> Tuple[str, List[Dict]]:
        """
        Iteratively clean a summary by removing invalid PMIDs and regenerating
        Returns: (cleaned_summary, validation_history)
        """
        current_summary = initial_summary
        validation_history = []
        
        for iteration in range(max_iterations):
            print(f"[VALIDATOR] Iteration {iteration + 1}/{max_iterations}")
            
            # Validate current summary
            validation_result = await self.validate_summary(current_summary, search_results)
            validation_history.append({
                'iteration': iteration + 1,
                'summary': current_summary,
                'validation': validation_result
            })
            
            if validation_result['valid']:
                print(f"[VALIDATOR] Summary is clean after {iteration + 1} iteration(s)")
                break
            
            # Identify PMIDs to remove
            pmids_to_remove = validation_result['missing_pmids'] + validation_result['invalid_pmids']
            
            if not pmids_to_remove:
                print("[VALIDATOR] No specific PMIDs to remove, stopping")
                break
            
            print(f"[VALIDATOR] Removing {len(pmids_to_remove)} problematic PMIDs: {pmids_to_remove}")
            
            # Generate new summary without problematic PMIDs
            try:
                current_summary = await self._regenerate_summary_without_pmids(
                    current_summary, pmids_to_remove, search_results
                )
            except Exception as e:
                print(f"[VALIDATOR] Error regenerating summary: {e}")
                break
        
        else:
            print(f"[VALIDATOR] Reached maximum iterations ({max_iterations}) without full validation")
        
        return current_summary, validation_history
    
    async def _regenerate_summary_without_pmids(self, original_summary: str, 
                                      pmids_to_remove: List[str], search_results: List[Dict]) -> str:
        """Regenerate summary excluding specific PMIDs using custom requirements if available"""
        
        # Get list of valid PMIDs
        all_pmids = {str(result['pmid']) for result in search_results}
        valid_pmids = [pmid for pmid in all_pmids if pmid not in pmids_to_remove]
        
        # Use custom requirements if provided, otherwise use default approach
        if self.custom_requirements and self.custom_requirements.strip():
            system_msg = (
                "You are a scientific writer. You will be provided with an existing literature synthesis "
                "and a list of PMIDs to exclude from citations. Your task is to rewrite the synthesis "
                "following the specific requirements provided, while excluding the problematic PMIDs.\n\n"
                "SYNTHESIS REQUIREMENTS:\n"
                f"{self.custom_requirements}\n\n"
                "CITATION RULES:\n"
                "- Only reference papers using the format (PMID: 12345678)\n"
                "- Only use PMIDs from the valid list provided below\n"
                "- Do not make up or hallucinate any PMIDs\n"
                "- Maintain the structure and style specified in the requirements above"
            )
        else:
            # Default system message (original behavior)
            system_msg = (
                "You are a scientific writer. You will be provided with an existing literature synthesis "
                "and a list of PMIDs to exclude from citations. Your task is to rewrite the synthesis "
                "to remove any references to the excluded PMIDs while maintaining the scientific accuracy "
                "and flow of the text. You may only cite PMIDs from the provided valid list.\n\n"
                "IMPORTANT: Only reference papers using the format (PMID: 12345678) and only use PMIDs "
                "from the valid list provided. Do not make up or hallucinate any PMIDs."
            )
        
        user_msg = f"""Original synthesis:
    {original_summary}

    PMIDs to EXCLUDE (do not cite these):
    {', '.join(pmids_to_remove)}

    Valid PMIDs you may cite (only use these):
    {', '.join(sorted(valid_pmids))}

    Please rewrite the synthesis excluding the problematic PMIDs while maintaining scientific accuracy."""

        # Rest of the method remains the same...
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        try:
            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise ValidationError(f"OpenAI API returned status {resp.status}", "api_error",
                                                details={"status_code": resp.status, "response": error_text})
                        
                        response_data = await resp.json()
                        new_summary = response_data["choices"][0]["message"]["content"].strip()
            else:
                # Fallback to synchronous requests
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise ValidationError(f"OpenAI API returned status {response.status_code}", "api_error",
                                        details={"status_code": response.status_code, "response": response.text})
                
                new_summary = response.json()["choices"][0]["message"]["content"].strip()
            
            return new_summary
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            raise ValidationError(f"Failed to regenerate summary: {str(e)}", "summary_regeneration_error", exception=e)
        