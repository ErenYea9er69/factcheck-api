document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const verifyBtn = document.getElementById('verify-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const resultsSection = document.getElementById('results-section');
    const claimsList = document.getElementById('claims-list');
    const statsContainer = document.getElementById('stats');

    verifyBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please paste some text first.');
            return;
        }

        // Set Loading State
        verifyBtn.classList.add('loading');
        verifyBtn.disabled = true;
        statusIndicator.textContent = 'Verifying claims via Longcat & Tavily...';
        statusIndicator.className = 'status-indicator loading';
        
        // Hide previous results
        resultsSection.style.display = 'none';
        claimsList.innerHTML = '';
        statsContainer.innerHTML = '';

        try {
            const response = await fetch('/api/check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Verification failed');
            }

            const results = await response.json();
            
            // Set Success State
            verifyBtn.classList.remove('loading');
            verifyBtn.disabled = false;
            statusIndicator.textContent = 'Verification complete';
            statusIndicator.className = 'status-indicator success';

            renderResults(results);

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
            verifyBtn.classList.remove('loading');
            verifyBtn.disabled = false;
            statusIndicator.textContent = 'Ready to check';
            statusIndicator.className = 'status-indicator';
        }
    });

    function renderResults(claims) {
        if (!claims || claims.length === 0) {
            claimsList.innerHTML = '<p style="text-align:center; color:var(--text-secondary);">No factual claims were found in the provided text.</p>';
            resultsSection.style.display = 'block';
            return;
        }

        // Calculate stats
        const stats = { confirmed: 0, contradicted: 0, unverifiable: 0, outdated: 0 };
        claims.forEach(c => {
            if (stats[c.verdict] !== undefined) {
                stats[c.verdict]++;
            }
        });

        statsContainer.innerHTML = `
            <span class="stat-badge" style="color:var(--verdict-confirmed); border-color:var(--verdict-confirmed)">${stats.confirmed} Confirmed</span>
            <span class="stat-badge" style="color:var(--verdict-contradicted); border-color:var(--verdict-contradicted)">${stats.contradicted} Contradicted</span>
        `;

        claims.forEach((claim, index) => {
            const card = document.createElement('div');
            card.className = 'claim-card';
            card.style.animationDelay = `${index * 0.1}s`;

            const colorVar = `var(--verdict-${claim.verdict})`;
            
            let sourceHtml = '';
            if (claim.sourceLink) {
                sourceHtml = `<a href="${claim.sourceLink}" target="_blank" class="source-link">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                    Source
                </a>`;
            } else {
                sourceHtml = `<span style="color:var(--text-secondary)">No source</span>`;
            }

            card.innerHTML = `
                <div class="claim-header">
                    <div class="claim-text">"${claim.claim}"</div>
                    <div class="verdict-badge verdict-${claim.verdict}">${claim.verdict}</div>
                </div>
                <div class="claim-reasoning">${claim.reasoning}</div>
                <div class="claim-footer">
                    <div class="confidence-bar-container">
                        <span>Confidence</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${claim.confidenceScore}%; background: ${colorVar}"></div>
                        </div>
                        <span>${claim.confidenceScore}%</span>
                    </div>
                    ${sourceHtml}
                </div>
            `;
            claimsList.appendChild(card);
        });

        resultsSection.style.display = 'block';
    }
});
