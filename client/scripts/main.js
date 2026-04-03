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
    try {
        const res = await fetch('./README.md');
        if (!res.ok) throw new Error('Not found');
        const text = await res.text();

        const renderer = new marked.Renderer();
        renderer.code = ({ text, lang }) => {
            if (lang === 'mermaid') return `<div class="mermaid">${text}</div>`;
            return `<pre><code>${text}</code></pre>`;
        };

        document.getElementById('readme-loading').style.display = 'none';
        document.getElementById('readme-content').innerHTML = marked.parse(text, { renderer });

        const readmeContent = document.getElementById('readme-content');

        // Wrap tables in scroll + fade containers
        readmeContent.querySelectorAll('table').forEach(table => {
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

        readmeContent.querySelectorAll('.table-fade-wrap').forEach(outer => {
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
        readmeContent.querySelectorAll('.mermaid').forEach(diagram => {
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
        document.getElementById('readme-loading').textContent =
            'Could not load README.md. Make sure index.html and README.md are in the same folder.';
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
