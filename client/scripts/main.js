import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
import { initGraph } from './graph.js';

mermaid.initialize({
    startOnLoad: false,
    theme: 'base',
    themeVariables: {
        fontFamily: "'Source Sans 3', system-ui, sans-serif",
        fontSize: '13px',
        primaryColor: '#d8f3dc',
        primaryTextColor: '#2a2420',
        primaryBorderColor: '#b7e4c7',
        lineColor: '#8a7d74',
        secondaryColor: '#f5f0eb',
        tertiaryColor: '#faf8f5',
    }
});

let readmeLoaded = false;
let graphLoaded = false;

// ── Tab switching ─────────────────────────────────────────────────────────────
window.showTab = function (name, el) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    el.classList.add('active');

    document.body.className = document.body.className
        .replace(/\btab-\S+/g, '').trim();
    document.body.classList.add(`tab-${name}`);

    window.scrollTo(0, 0);
    if (name === 'docs' && !readmeLoaded) loadReadme();
    if (name === 'graph' && !graphLoaded) {
        graphLoaded = true;
        initGraph('graph-container');
    }
    // Sync drawer tab highlights
    document.querySelectorAll('.nav-drawer-tab').forEach(t => {
        t.classList.toggle('active', t.getAttribute('href').includes('tab=' + name));
    });
    const url = new URL(window.location);
    url.searchParams.set('tab', name);
    window.history.pushState({}, '', url);
};

// On load, restore tab from URL
window.addEventListener('DOMContentLoaded', function () {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get('tab');
    if (tab === 'docs' || tab === 'graph') {
        const tabEl = document.querySelector(`.nav-tab[href*="tab=${tab}"]`);
        if (tabEl) window.showTab(tab, tabEl);
    }
});

// ── README loader ─────────────────────────────────────────────────────────────
async function loadReadme() {
    const loadingEl = document.getElementById('readme-loading');
    const contentEl = document.getElementById('readme-content');
    try {
        let res = await fetch('../README.md');
        if (!res.ok) {
            res = await fetch('./README.md');
        }
        if (!res.ok) {
            res = await fetch('https://raw.githubusercontent.com/jaxendutta/emma/main/README.md');
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}: Not found`);
        const text = await res.text();

        let html = '';
        if (typeof marked.use === 'function') {
            marked.use({
                renderer: {
                    code(args) {
                        const codeText = typeof args === 'object' ? args.text : args;
                        const lang = typeof args === 'object' ? args.lang : arguments[1];
                        if (lang === 'mermaid') return `<div class="mermaid">${codeText}</div>\n`;
                        return `<pre><code>${codeText}</code></pre>\n`;
                    }
                }
            });
            html = marked.parse(text);
        } else {
            const renderer = new marked.Renderer();
            renderer.code = (code, lang) => {
                const codeText = typeof code === 'object' ? code.text : code;
                const codeLang = typeof code === 'object' ? code.lang : lang;
                if (codeLang === 'mermaid') return `<div class="mermaid">${codeText}</div>`;
                return `<pre><code>${codeText}</code></pre>`;
            };
            html = marked.parse(text, { renderer });
        }

        loadingEl.style.display = 'none';
        contentEl.innerHTML = html;

        // Render KaTeX LaTeX math expressions ($\kappa$, etc.)
        if (window.renderMathInElement) {
            window.renderMathInElement(contentEl, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false
            });
        }

        // Wrap tables in scroll + fade containers
        contentEl.querySelectorAll('table').forEach(table => {
            if (!table.parentElement.classList.contains('table-scroll')) {
                const scroll = document.createElement('div');
                scroll.className = 'table-scroll';
                const outer = document.createElement('div');
                outer.className = 'table-fade-wrap';
                table.parentNode.insertBefore(outer, table);
                scroll.appendChild(table);
                outer.appendChild(scroll);
            }
        });

        contentEl.querySelectorAll('.table-fade-wrap').forEach(outer => {
            const scroll = outer.querySelector('.table-scroll');
            const update = () => {
                const atEnd = scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - 4;
                const atStart = scroll.scrollLeft <= 4;
                const hasOverflow = scroll.scrollWidth > scroll.clientWidth;
                outer.classList.toggle('fade-right', hasOverflow && !atEnd);
                outer.classList.toggle('fade-left', hasOverflow && !atStart);
            };
            update();
            scroll.addEventListener('scroll', update, { passive: true });
            window.addEventListener('resize', update, { passive: true });
        });

        await mermaid.run({ nodes: document.querySelectorAll('#readme-content .mermaid') });

        // Mermaid zoom/pan toolbars
        contentEl.querySelectorAll('.mermaid').forEach(diagram => {
            if (diagram.parentElement.classList.contains('mermaid-zoomable')) return;

            const wrapper = document.createElement('div');
            wrapper.className = 'mermaid-zoomable';
            diagram.parentNode.insertBefore(wrapper, diagram);
            wrapper.appendChild(diagram);

            const toolbar = document.createElement('div');
            toolbar.className = 'mermaid-toolbar';
            toolbar.innerHTML = `
                <button title="Zoom in">+</button>
                <button title="Zoom out">&ndash;</button>
                <button title="Reset zoom">&#x27F3;</button>
                <button title="Open in new tab">&#x26F6;</button>
            `;
            wrapper.insertBefore(toolbar, diagram);

            let scale = 1, minScale = 0.5, maxScale = 3;
            let panX = 0, panY = 0, isPanning = false, startX = 0, startY = 0;
            const svg = diagram.querySelector('svg');
            if (!svg) return;
            svg.style.transformOrigin = '0 0';

            function updateTransform() {
                svg.style.transform = `translate(${panX}px,${panY}px) scale(${scale})`;
            }
            function reset() { scale = 1; panX = 0; panY = 0; updateTransform(); }

            toolbar.children[0].onclick = () => { scale = Math.min(maxScale, scale * 1.2); updateTransform(); };
            toolbar.children[1].onclick = () => { scale = Math.max(minScale, scale / 1.2); updateTransform(); };
            toolbar.children[2].onclick = () => reset();
            toolbar.children[3].onclick = () => {
                const svgData = new XMLSerializer().serializeToString(svg);
                const blob = new Blob([svgData], { type: 'image/svg+xml' });
                const url = URL.createObjectURL(blob);
                window.open(url, '_blank');
            };

            svg.addEventListener('mousedown', e => {
                if (scale === 1) return;
                isPanning = true;
                startX = e.clientX - panX;
                startY = e.clientY - panY;
                wrapper.style.cursor = 'grabbing';
            });
            window.addEventListener('mousemove', e => {
                if (!isPanning) return;
                panX = e.clientX - startX;
                panY = e.clientY - startY;
                updateTransform();
            });
            window.addEventListener('mouseup', () => {
                isPanning = false;
                wrapper.style.cursor = 'grab';
            });
            svg.addEventListener('touchstart', e => {
                if (scale === 1) return;
                isPanning = true;
                const t = e.touches[0];
                startX = t.clientX - panX;
                startY = t.clientY - panY;
            });
            window.addEventListener('touchmove', e => {
                if (!isPanning) return;
                const t = e.touches[0];
                panX = t.clientX - startX;
                panY = t.clientY - startY;
                updateTransform();
            });
            window.addEventListener('touchend', () => { isPanning = false; });
        });

        readmeLoaded = true;
    } catch (e) {
        console.error('Failed to load README.md:', e);
        if (loadingEl) {
            loadingEl.style.display = 'block';
            loadingEl.textContent = `Could not load README.md (${e.message}). Make sure index.html and README.md are in the same folder.`;
        }
    }
}

// ── Hamburger menu ────────────────────────────────────────────────────────────
const hamburger = document.getElementById('nav-hamburger');
const drawer = document.getElementById('nav-drawer');
const overlay = document.getElementById('nav-overlay');
const nav = document.querySelector('nav');

// Observe nav-drawer open state and update nav border-radius
function updateNavDrawerOpenClass() {
    if (!nav || !drawer) return;
    if (drawer.classList.contains('open')) {
        nav.classList.add('nav-drawer-open');
    } else {
        nav.classList.remove('nav-drawer-open');
    }
}

function closeDrawer() {
    hamburger.classList.remove('open');
    drawer.classList.remove('open');
    overlay.classList.remove('open');
    updateNavDrawerOpenClass();
}

hamburger.addEventListener('click', e => {
    e.stopPropagation();
    const open = drawer.classList.toggle('open');
    hamburger.classList.toggle('open', open);
    overlay.classList.toggle('open', open);
    updateNavDrawerOpenClass();
});

overlay.addEventListener('click', closeDrawer);

// ── Specialties grid population ───────────────────────────────────────────
window.renderSpecialtiesGrid = function () {
    const specialties = [
        "Anaesthesia",
        "Anatomy",
        "Biochemistry",
        "Dental",
        "Dermatology",
        "Emergency Medicine",
        "ENT",
        "Internal Medicine",
        "Microbiology",
        "Obstetrics & Gynaecology",
        "Ophthalmology",
        "Orthopaedics",
        "Pathology",
        "Pediatrics",
        "Pharmacology",
        "Physiology",
        "Psychiatry",
        "Public Health",
        "Radiology",
        "Surgery"
    ].sort((a, b) => a.localeCompare(b));
    const grid = document.getElementById('specialties-grid');
    if (!grid) return;
    grid.innerHTML = '';
    specialties.forEach((name, idx) => {
        const pill = document.createElement('span');
        pill.className = 'specialty-pill';
        pill.textContent = name;
        pill.style.animationDelay = `${idx * 55}ms`;
        grid.appendChild(pill);
    });
};

window.addEventListener('DOMContentLoaded', function () {
    window.renderSpecialtiesGrid();

    // ── Native Floating Chat Drawer & Teaser Controller ─────────────────────
    const fab = document.getElementById('emma-fab');
    const drawer = document.getElementById('emma-drawer');
    const closeBtn = document.getElementById('emma-drawer-close');
    const teaser = document.getElementById('emma-teaser');
    const teaserClose = document.getElementById('emma-teaser-close');
    const form = document.getElementById('emma-input-form');
    const input = document.getElementById('emma-input-field');
    const messages = document.getElementById('emma-messages');

    function openDrawer() {
        if (drawer) drawer.classList.remove('emma-hidden');
        if (teaser) teaser.classList.add('emma-hidden');
    }

    function toggleDrawer() {
        if (!drawer) return;
        const isHidden = drawer.classList.contains('emma-hidden');
        if (isHidden) {
            openDrawer();
        } else {
            drawer.classList.add('emma-hidden');
        }
    }

    if (fab) fab.addEventListener('click', toggleDrawer);
    if (teaser) teaser.addEventListener('click', openDrawer);
    if (teaserClose) {
        teaserClose.addEventListener('click', (e) => {
            e.stopPropagation();
            if (teaser) teaser.classList.add('emma-hidden');
        });
    }
    if (closeBtn && drawer) {
        closeBtn.addEventListener('click', () => drawer.classList.add('emma-hidden'));
    }

    // Show teaser after 2s delay on first page load
    if (teaser) {
        setTimeout(() => {
            teaser.classList.remove('emma-hidden');
        }, 2000);
    }

    // ── Save Dropdown Menu ─────────────────────────────────────────────
    const saveBtn = document.getElementById('emma-save-btn');
    const saveMenu = document.getElementById('emma-save-menu');
    const actionCopy = document.getElementById('emma-action-copy');
    const actionDownload = document.getElementById('emma-action-download');

    if (saveBtn && saveMenu) {
        saveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            saveMenu.classList.toggle('emma-hidden');
        });

        document.addEventListener('click', () => {
            saveMenu.classList.add('emma-hidden');
        });
    }

    function getTranscriptText() {
        if (!messages) return '';
        const msgDivs = messages.querySelectorAll('.emma-msg');
        let text = 'EMMA Emergency Medicine Mentoring Agent — Conversation Transcript\n';
        text += 'Date: ' + new Date().toLocaleString() + '\n';
        text += '='.repeat(60) + '\n\n';

        msgDivs.forEach(div => {
            if (div.classList.contains('emma-msg-typing')) return;
            const isUser = div.classList.contains('emma-msg-user');
            const sender = isUser ? 'User' : 'EMMA';
            const content = div.innerText || div.textContent || '';
            text += `[${sender}]:\n${content.trim()}\n\n`;
        });
        return text;
    }

    if (actionCopy) {
        actionCopy.addEventListener('click', async () => {
            const transcript = getTranscriptText();
            try {
                await navigator.clipboard.writeText(transcript);
                actionCopy.innerText = '✓ Copied!';
                setTimeout(() => {
                    actionCopy.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy to Clipboard';
                }, 2000);
            } catch (err) {
                console.error('Copy failed:', err);
            }
        });
    }

    if (actionDownload) {
        actionDownload.addEventListener('click', () => {
            const transcript = getTranscriptText();
            const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const now = new Date();
            const timestamp = now.getFullYear().toString() +
                String(now.getMonth() + 1).padStart(2, '0') +
                String(now.getDate()).padStart(2, '0') + '-' +
                String(now.getHours()).padStart(2, '0') +
                String(now.getMinutes()).padStart(2, '0') +
                String(now.getSeconds()).padStart(2, '0');
            const a = document.createElement('a');
            a.href = url;
            a.download = `emma-transcript-${timestamp}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    // ── Fullscreen Toggle ──────────────────────────────────────────────
    const fullscreenBtn = document.getElementById('emma-fullscreen-btn');
    const fullscreenIcon = document.getElementById('emma-fullscreen-icon');

    if (fullscreenBtn && drawer && fullscreenIcon) {
        fullscreenBtn.addEventListener('click', () => {
            const isFull = drawer.classList.toggle('emma-fullscreen');
            if (isFull) {
                fullscreenBtn.title = 'Exit Fullscreen';
                fullscreenIcon.innerHTML = '<polyline points="4 14 10 14 10 20"></polyline><polyline points="20 10 14 10 14 4"></polyline><line x1="14" y1="10" x2="21" y2="3"></line><line x1="3" y1="21" x2="10" y2="14"></line>';
            } else {
                fullscreenBtn.title = 'Toggle Fullscreen';
                fullscreenIcon.innerHTML = '<polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line>';
            }
        });
    }

    let isFirstMessageSent = false;

    if (form && input && messages) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = input.value.trim();
            if (!text) return;

            // Append User Message
            const userDiv = document.createElement('div');
            userDiv.className = 'emma-msg emma-msg-user';
            userDiv.textContent = text;
            messages.appendChild(userDiv);
            input.value = '';
            messages.scrollTop = messages.scrollHeight;

            // Append Typing Indicator
            const typingDiv = document.createElement('div');
            typingDiv.className = 'emma-msg emma-msg-typing';
            typingDiv.textContent = 'EMMA is thinking...';
            messages.appendChild(typingDiv);
            messages.scrollTop = messages.scrollHeight;

            let wakeupTimer = null;
            if (!isFirstMessageSent) {
                isFirstMessageSent = true;
                wakeupTimer = setTimeout(() => {
                    if (messages.contains(typingDiv)) {
                        typingDiv.textContent = 'EMMA is waking up. This might take up to a minute...';
                    }
                }, 3500);
            }

            // Fetch from /chat API with automatic fallback
            try {
                const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
                let apiUrl = isLocal ? 'http://localhost:8080/chat' : 'https://emma-webhook.onrender.com/chat';

                let sid = window.sessionStorage.getItem('emma_sid');
                if (!sid) {
                    sid = 'emma-session-' + Math.random().toString(36).substring(2, 9);
                    window.sessionStorage.setItem('emma_sid', sid);
                }

                let res;
                try {
                    res = await fetch(apiUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, session_id: sid })
                    });
                } catch (localErr) {
                    // Fallback to live Render endpoint if local port 8080 is down
                    if (isLocal) {
                        res = await fetch('https://emma-webhook.onrender.com/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: text, session_id: sid })
                        });
                    } else {
                        throw localErr;
                    }
                }
                const data = await res.json();

                if (wakeupTimer) clearTimeout(wakeupTimer);

                // Remove typing indicator
                if (messages.contains(typingDiv)) {
                    messages.removeChild(typingDiv);
                }

                // Append AI Response (split into individual speech bubbles per paragraph)
                const answerText = data.answer || data.text || "Sorry, I didn't receive a valid response.";
                const paragraphs = answerText.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
                if (paragraphs.length === 0) {
                    paragraphs.push(answerText);
                }

                paragraphs.forEach((para, idx) => {
                    setTimeout(() => {
                        const aiDiv = document.createElement('div');
                        aiDiv.className = 'emma-msg emma-msg-ai emma-pop-in';
                        const normalizedPara = para.replace(/^[•·]\s*/gm, '- ');
                        if (window.marked) {
                            aiDiv.innerHTML = window.marked.parse(normalizedPara);
                        } else {
                            aiDiv.textContent = para;
                        }
                        if (window.renderMathInElement) {
                            window.renderMathInElement(aiDiv, {
                                delimiters: [
                                    { left: '$$', right: '$$', display: true },
                                    { left: '$', right: '$', display: false },
                                    { left: '\\(', right: '\\)', display: false },
                                    { left: '\\[', right: '\\]', display: true }
                                ],
                                throwOnError: false
                            });
                        }
                        messages.appendChild(aiDiv);
                        messages.scrollTop = messages.scrollHeight;
                    }, idx * 280);
                });
            } catch (err) {
                if (wakeupTimer) clearTimeout(wakeupTimer);
                if (messages.contains(typingDiv)) {
                    messages.removeChild(typingDiv);
                }
                const errDiv = document.createElement('div');
                errDiv.className = 'emma-msg emma-msg-ai';
                errDiv.textContent = 'Connection error. Please ensure the backend server is running.';
                messages.appendChild(errDiv);
                messages.scrollTop = messages.scrollHeight;
            }
        });
    }
});
