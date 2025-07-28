let currentResults = [];
let currentQuery = '';

// Enhanced form validation and UX
document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
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
            const analysisPanel = document.getElementById('analysisPanel');
            if (analysisPanel) {
                analysisPanel.classList.remove('visible');
            }
            
            // Update button states after form submission
            setTimeout(updateButtonStates, 100);
        });
    }

    // Remove error state on input
    const queryInput = document.getElementById('query');
    if (queryInput) {
        queryInput.addEventListener('input', function() {
            this.classList.remove('error');
        });
    }

    // Initialize button event listeners
    initializeButtons();
    
    // Update button states on page load
    updateButtonStates();
    
    // Store results data if present
    if (document.querySelectorAll('.result-item').length > 0) {
        setTimeout(updateButtonStates, 100);
    }
});

function initializeButtons() {
    // Button references
    const helpBtn = document.getElementById('helpBtn');
    const helpModal = document.getElementById('helpModal');
    const closeModal = document.getElementById('closeModal');
    const downloadBtn = document.getElementById('downloadBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const copyBtn = document.getElementById('copyBtn');
    const logoutBtn = document.getElementById('logoutBtn');

    // Help modal
    if (helpBtn && helpModal && closeModal) {
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
    }

    // Logout functionality
    if (logoutBtn) {
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
    }

    // Download RIS functionality
    if (downloadBtn) {
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
    }

    // AI Analysis functionality
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async function() {
            if (analyzeBtn.disabled || currentResults.length === 0) return;
            
            const originalContent = analyzeBtn.innerHTML;
            analyzeBtn.innerHTML = '<span style="font-size: 16px; animation: spin 1s linear infinite;">⟳</span>';
            analyzeBtn.disabled = true;
            
            // Show analysis panel and loading state
            const analysisPanel = document.getElementById('analysisPanel');
            const analysisLoading = document.getElementById('analysisLoading');
            const analysisText = document.getElementById('analysisText');
            const copyButton = document.getElementById('copyBtn');
            
            if (analysisPanel) analysisPanel.classList.add('visible');
            if (analysisLoading) analysisLoading.style.display = 'block';
            if (analysisText) analysisText.style.display = 'none';
            if (copyButton) copyButton.style.display = 'none';
            
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
                        if (analysisText) {
                            analysisText.innerHTML = `<p>${formattedSynthesis}</p>`;
                            analysisText.style.display = 'block';
                        }
                        if (analysisLoading) analysisLoading.style.display = 'none';
                        if (copyButton) copyButton.style.display = 'flex';
                        
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
                if (analysisLoading) analysisLoading.style.display = 'none';
                if (analysisText) {
                    analysisText.innerHTML = `<div style="color: var(--error); text-align: center; padding: 20px;">
                        <strong>Analysis Failed</strong><br>
                        ${error.message || 'Please try again later.'}
                    </div>`;
                    analysisText.style.display = 'block';
                }
                analyzeBtn.innerHTML = originalContent;
            } finally {
                analyzeBtn.disabled = false;
            }
        });
    }

    // Copy functionality
    if (copyBtn) {
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
    }
}

function showCopyNotification() {
    const notification = document.getElementById('copyNotification');
    if (notification) {
        notification.classList.add('show');
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
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
    
    if (!resultsContainer) return;
    
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
        
        if (aiRank && aiRank !== 999 && aiRankElement && aiRankNumber) {
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
        const titleElement = item.querySelector('.result-title a');
        const title = titleElement ? titleElement.textContent.trim() : '';
        const metaElement = item.querySelector('.result-meta');
        const meta = metaElement ? metaElement.textContent : '';
        const abstractElement = item.querySelector('.result-abstract');
        const abstract = abstractElement ? abstractElement.textContent.trim() : '';
        
        // Parse metadata
        const metaParts = meta.split('•');
        const authors = metaParts[0] ? metaParts[0].trim() : 'Unknown';
        const journalYear = metaParts[1] ? metaParts[1].trim() : '';
        const [journal, year] = journalYear.includes('(') ? 
            [journalYear.split('(')[0].trim(), parseInt(journalYear.match(/\((\d{4})\)/)?.[1]) || new Date().getFullYear()] :
            [journalYear, new Date().getFullYear()];
        
        // Parse footer for strategy and weight
        const footerElement = item.querySelector('.result-footer');
        const footer = footerElement ? footerElement.textContent : '';
        const strategyElement = item.querySelector('.strategy-tag');
        const strategy = strategyElement ? strategyElement.textContent.trim() : '';
        const weightMatch = footer.match(/Weight:\s*([\d.]+)/);
        const weight = weightMatch ? parseFloat(weightMatch[1]) : 0;
        
        // Parse scores
        const combinedScoreElement = item.querySelector('.result-combined');
        const combinedScoreText = combinedScoreElement ? combinedScoreElement.textContent : '';
        const combinedScore = parseFloat(combinedScoreText.match(/Score:\s*([\d.]+)/)?.[1]) || 0;
        
        const impactElement = item.querySelector('.result-impact');
        const journalImpact = impactElement ? 
            parseFloat(impactElement.textContent.match(/JIF:\s*([\d.]+)/)?.[1]) || 0 : 0;
        
        const rankElement = item.querySelector('.result-rank');
        const rank = rankElement ? parseInt(rankElement.textContent.replace('#', '')) || 0 : 0;
        
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
    const downloadBtn = document.getElementById('downloadBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const hasResults = document.querySelectorAll('.result-item').length > 0;
    
    if (downloadBtn) {
        downloadBtn.disabled = !hasResults;
        downloadBtn.title = hasResults ? 'Download RIS file' : 'No results to download';
    }
    
    if (analyzeBtn) {
        analyzeBtn.disabled = !hasResults;
        analyzeBtn.title = hasResults ? 'AI Analysis & Ranking' : 'No results to analyze';
    }
    
    if (hasResults) {
        // Store the current query
        const queryInput = document.getElementById('query');
        if (queryInput) {
            currentQuery = queryInput.value;
        }
        // Store results data for AI analysis
        storeResultsData();
    }
}