// Auto-refresh stats every 30 seconds
async function refreshStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('stat-mia').textContent = `${data.mia?.pending || 0} pending`;
        document.getElementById('stat-zara').textContent = `${data.zara?.pending || 0} pending`;
        document.getElementById('stat-luna').textContent = `${data.luna?.pending || 0} pending`;
    } catch (e) {
        console.error('Stats refresh failed:', e);
    }
}

async function approve(id) {
    const caption = document.getElementById(`caption-${id}`)?.value;
    if (caption) {
        await fetch(`/caption/${id}`, {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({caption})
        });
    }
    const res = await fetch(`/approve/${id}`, {method: 'POST'});
    if (res.ok) {
        const card = document.getElementById(`card-${id}`);
        card.style.opacity = '0';
        card.style.transition = 'opacity 0.3s';
        setTimeout(() => card.remove(), 300);
        refreshStats();
    }
}

async function reject(id) {
    const res = await fetch(`/reject/${id}`, {method: 'POST'});
    if (res.ok) {
        const card = document.getElementById(`card-${id}`);
        card.style.opacity = '0';
        card.style.transition = 'opacity 0.3s';
        setTimeout(() => card.remove(), 300);
        refreshStats();
    }
}

function copyCaption(id) {
    const ta = document.getElementById(`caption-${id}`);
    if (!ta) return;
    navigator.clipboard.writeText(ta.value).then(() => {
        const btn = ta.parentElement.querySelector('.btn-copy');
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy Caption', 2000);
    });
}

// Load stats on page load + refresh every 30s
refreshStats();
setInterval(refreshStats, 30000);
setInterval(() => location.reload(), 60000);  // full reload every 60s
