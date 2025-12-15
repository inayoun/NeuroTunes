let selectedSubject = null;
let focusData = {};
const CLIP_TITLES = ["Chim Chim Cheree", "Take Me Out to the Ballgame", "Mary Had a Little Lamb", "Jingle Bells", "Emperor Waltz", "Hedwig's Theme", "Imperial March", "Eine Kleine Nachtmusik"];
const CLIP_METADATA = {
    "clip_01": { title: "Chim Chim Cheree (lyrics)", genre: "Lyrics", bpm: 212 },
    "clip_02": { title: "Take Me Out to the Ballgame (lyrics)", genre: "Lyrics", bpm: 189 },
    "clip_03": { title: "Jingle Bells (lyrics)", genre: "Lyrics", bpm: 200 },
    "clip_04": { title: "Mary Had a Little Lamb (lyrics)", genre: "Lyrics", bpm: 160 },
    "clip_11": { title: "Chim Chim Cheree (instrumental)", genre: "Instrumental", bpm: 212 },
    "clip_12": { title: "Take Me Out to the Ballgame (instrumental)", genre: "Instrumental", bpm: 189 },
    "clip_13": { title: "Jingle Bells (instrumental)", genre: "Instrumental", bpm: 200 },
    "clip_14": { title: "Mary Had a Little Lamb (instrumental)", genre: "Instrumental", bpm: 160 },
    "clip_21": { title: "Emperor Waltz", genre: "Classical", bpm: 178 },
    "clip_22": { title: "Hedwig's Theme (Harry Potter)", genre: "Classical", bpm: 166 },
    "clip_23": { title: "Imperial March (Star Wars Theme)", genre: "Classical", bpm: 104 },
    "clip_24": { title: "Eine Kleine Nachtmusik", genre: "Classical", bpm: 140 }
};

function selectSubject(subjectId) {
    selectedSubject = subjectId;
    document.querySelectorAll('.subject-card').forEach(card => card.classList.remove('selected'));
    event.currentTarget.classList.add('selected');
    document.getElementById(`subject-${subjectId.toLowerCase()}`).checked = true;
    document.getElementById('analyze-btn').disabled = false;
}

function navigateTo(screen) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(`${screen}-screen`).classList.add('active');
    window.scrollTo(0, 0);
}

async function startAnalysis() {
    if (!selectedSubject) return;
    navigateTo('loading');
    startLoadingAnimation();
    await loadAndProcessData();
}

async function loadAndProcessData() {
    const loadingTitle = document.getElementById('loading-title');
    const totalTime = CLIP_TITLES.length * 500;
    const startTime = Date.now();
    
    const updateProgress = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / totalTime) * 100, 100);
        currentLoadingProgress = progress;
        
        const clipIndex = Math.floor((progress / 100) * CLIP_TITLES.length);
        if (clipIndex < CLIP_TITLES.length) {
            loadingTitle.textContent = `Sampling ${CLIP_TITLES[clipIndex]}...`;
        }
        
        if (progress < 100) {
            requestAnimationFrame(updateProgress);
        }
    };
    
    updateProgress();
    
    await new Promise(resolve => setTimeout(resolve, totalTime));
    try {
        await loadFocusData(selectedSubject);
        navigateTo('profile');
        renderProfile();
    } catch (error) {
        console.error('Error loading data:', error);
        alert('Error loading focus data. Please try again.');
        navigateTo('start');
    }
}

async function loadFocusData(subjectId) {
    try {
        let response = await fetch(`./data/${subjectId}_focus.json`);
        if (response.ok) {
            focusData = await response.json();
            console.log(`Loaded pre-generated focus data for ${subjectId}`);
        } else {
            response = await fetch('./data/ai_clips.json');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const clips = await response.json();
            if (!clips || clips.length === 0) throw new Error('No clip data found');
            focusData = processFocusData(clips, subjectId);
            console.log(`Processed focus data for ${subjectId} from ai_clips.json`);
        }
        if (!focusData || !focusData.tracks || focusData.tracks.length === 0) {
            throw new Error(`No data found for subject ${subjectId}`);
        }
    } catch (error) {
        console.error('Error loading focus data:', error);
        throw error;
    }
}

function processFocusData(clips, subjectId) {
    const subjectClips = clips.filter(c => c.subject === subjectId);
    const tracks = subjectClips.map(clip => {
        const meta = CLIP_METADATA[clip.clip_id];
        return { ...clip, ...meta, fi_mean: clip.EI_mean, fi_std: clip.EI_sd, tempo_bucket: tempoBucket(meta.bpm), fi_mean_norm: 0 };
    });
    const fiMeans = tracks.map(t => t.fi_mean);
    const minFi = Math.min(...fiMeans);
    const maxFi = Math.max(...fiMeans);
    tracks.forEach(t => { t.fi_mean_norm = 100 * (t.fi_mean - minFi) / (maxFi - minFi + 1e-10); });
    const fiStds = tracks.map(t => t.fi_std);
    fiStds.sort((a, b) => a - b);
    const p33 = fiStds[Math.floor(fiStds.length * 0.33)];
    const p66 = fiStds[Math.floor(fiStds.length * 0.66)];
    tracks.forEach(t => {
        if (t.fi_std <= p33) t.stability_label = "Stable";
        else if (t.fi_std <= p66) t.stability_label = "Medium";
        else t.stability_label = "Spiky";
    });
    const genreStats = {};
    ['Lyrics', 'Instrumental', 'Classical'].forEach(genre => {
        const genreTracks = tracks.filter(t => t.genre === genre);
        const fiMeans = genreTracks.map(t => t.fi_mean);
        const fiStds = genreTracks.map(t => t.fi_std);
        const meanFi = fiMeans.reduce((a, b) => a + b, 0) / fiMeans.length;
        const meanStd = fiStds.reduce((a, b) => a + b, 0) / fiStds.length;
        genreStats[genre] = { fi_mean: meanFi, fi_std: meanStd, score: meanFi - meanStd };
    });
    const bestGenre = Object.keys(genreStats).reduce((a, b) => genreStats[a].score > genreStats[b].score ? a : b);
    const worstGenre = Object.keys(genreStats).reduce((a, b) => genreStats[a].score < genreStats[b].score ? a : b);
    const tempoStats = {};
    ['Slow', 'Medium', 'Fast'].forEach(bucket => {
        const bucketTracks = tracks.filter(t => t.tempo_bucket === bucket);
        if (bucketTracks.length > 0) {
            const fiMeans = bucketTracks.map(t => t.fi_mean);
            tempoStats[bucket] = { fi_mean: fiMeans.reduce((a, b) => a + b, 0) / fiMeans.length };
        }
    });
    const tempoMeans = Object.values(tempoStats).map(s => s.fi_mean);
    const minTempo = Math.min(...tempoMeans);
    const maxTempo = Math.max(...tempoMeans);
    Object.keys(tempoStats).forEach(bucket => {
        tempoStats[bucket].fi_mean_norm = 100 * (tempoStats[bucket].fi_mean - minTempo) / (maxTempo - minTempo + 1e-10);
    });
    const tempoWinner = Object.keys(tempoStats).reduce((a, b) => tempoStats[a].fi_mean > tempoStats[b].fi_mean ? a : b);
    return { subject_id: subjectId, genre_stats: genreStats, best_genre: bestGenre, worst_genre: worstGenre, tempo_stats: tempoStats, tempo_winner: tempoWinner, tracks: tracks };
}

function tempoBucket(bpm) {
    if (bpm < 150) return "Slow";
    if (bpm <= 180) return "Medium";
    return "Fast";
}

function flipCard() {
    const card = document.getElementById('flip-card');
    if (!card.classList.contains('flipped')) card.classList.add('flipped');
}

(() => {
    const card = document.getElementById('flip-card');
    const section31 = document.getElementById('section-3-1');
    if (!card || !section31) return;

    let cardFlipped = false;
    let hasScrolledInSection = false;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting && entry.intersectionRatio > 0.8) {
                if (!hasScrolledInSection) {
                    hasScrolledInSection = true;
                    setTimeout(() => {
                        if (!cardFlipped) {
                            card.classList.add('flipped');
                            cardFlipped = true;
                        }
                    }, 800);
                }
            }
        });
    }, { threshold: [0, 0.5, 0.8, 1] });

    observer.observe(section31);
})();

function renderProfile() {
    document.getElementById('best-genre-text').textContent = `${focusData.best_genre} showed the strongest, most stable Focus Index.`;
    renderGenreScatter();
    renderTempoChart();
    renderTrackScatter();
    renderWaveLegend();
}

function renderGenreScatter() {
    const genres = Object.keys(focusData.genre_stats);
    const data = genres.map(genre => {
        const stats = focusData.genre_stats[genre];
        return {
            x: [stats.fi_mean], y: [1 / (stats.fi_std + 0.01)], mode: 'markers+text', type: 'scatter', name: genre,
            text: [genre], textposition: 'top center',
            marker: { size: 16, color: genre === 'Lyrics' ? '#ff6b9d' : genre === 'Instrumental' ? '#00d4ff' : '#ffd700' }
        };
    });
    const layout = {
        xaxis: { title: 'Focus Level →', gridcolor: '#2a2a4a' }, yaxis: { title: 'Stability →', gridcolor: '#2a2a4a' },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(20,30,50,0.3)', font: { color: '#e0e0e0' },
        showlegend: false, hovermode: 'closest'
    };
    Plotly.newPlot('genre-scatter', data, layout, { displayModeBar: false, staticPlot: true });
}

function renderTempoChart() {
    const bpmFiPairs = focusData.tracks.map(t => {
        const meta = CLIP_METADATA[t.clip_id];
        return {
            bpm: meta?.bpm || 0,
            fi: t.fi_mean_norm,
            title: meta?.title || t.title || 'Unknown'
        };
    }).filter(p => p.bpm > 0);
    
    bpmFiPairs.sort((a, b) => a.bpm - b.bpm);
    
    const bpms = bpmFiPairs.map(p => p.bpm);
    const fiValues = bpmFiPairs.map(p => p.fi);
    const titles = bpmFiPairs.map(p => p.title);
    
    console.log('Tempo chart titles:', titles);
    
    const data = [{
        x: bpms,
        y: fiValues,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#c800ff', width: 3 },
        marker: { size: 8, color: '#c800ff' },
        text: titles,
        hovertemplate: '%{text}<extra></extra>',
        hoverinfo: 'text'
    }];
    const layout = {
        xaxis: { title: 'BPM (Beats Per Minute)', gridcolor: '#2a2a4a' }, 
        yaxis: { title: 'Focus Index', gridcolor: '#2a2a4a', range: [0, 100] },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(20,30,50,0.3)', font: { color: '#e0e0e0' }, showlegend: false,
        hovermode: 'closest'
    };
    Plotly.newPlot('tempo-chart', data, layout, { displayModeBar: false, staticPlot: false });
    
    const maxIdx = fiValues.indexOf(Math.max(...fiValues));
    const bestBpm = bpms[maxIdx];
    document.getElementById('tempo-description').textContent = `BPM ${bestBpm} produced the highest focus score for this brain.`;
}

function renderTrackScatter() {
    const genreColors = {
        'Lyrics': '#ff6b9d',
        'Instrumental': '#00d4ff',
        'Classical': '#ffd700'
    };

    const tempoSymbols = {
        'Slow': 'circle',
        'Medium': 'square',
        'Fast': 'diamond'
    };

    const data = focusData.tracks.map(track => ({
        x: [track.fi_mean_norm],
        y: [1 / (track.fi_std + 0.01)],
        mode: 'markers',
        type: 'scatter',
        name: track.title,
        marker: {
            size: 12,
            color: genreColors[track.genre],
            symbol: tempoSymbols[track.tempo_bucket]
        },
        customdata: track,
        hovertemplate: '<b>' + track.title + '</b><extra></extra>'
    }));

    const layout = {
        xaxis: { title: 'Focus Level →', gridcolor: '#2a2a4a' },
        yaxis: { title: 'Stability ↑', gridcolor: '#2a2a4a' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(20,30,50,0.3)',
        font: { color: '#e0e0e0' },
        showlegend: false,
        hovermode: 'closest',
        margin: { l: 60, r: 20, t: 20, b: 60 },
        autosize: true
    };

    Plotly.newPlot('track-scatter', data, layout, { displayModeBar: false, staticPlot: false, responsive: true });
    document.getElementById('track-scatter').on('plotly_click', function(eventData) {
        const point = eventData.points[0];
        const track = point.data.customdata;
        console.log('Track clicked:', track);
        displayTrackInfo(track);
        renderTrackDetail(track);
    });
}

function displayTrackInfo(track) {
    const container = document.getElementById('track-info-container');
    container.innerHTML = `
        <div class="track-info-item">
            <div class="track-info-label">Title</div>
            <div class="track-info-value">${track.title}</div>
        </div>
        <div class="track-info-item">
            <div class="track-info-label">Type</div>
            <div class="track-info-value">${track.genre}</div>
        </div>
        <div class="track-info-item">
            <div class="track-info-label">BPM</div>
            <div class="track-info-value">${track.bpm}</div>
        </div>
        <div class="track-info-item">
            <div class="track-info-label">Focus Index</div>
            <div class="track-info-value">${track.fi_mean_norm.toFixed(1)}/100</div>
        </div>
        <div class="track-info-item">
            <div class="track-info-label">Stability</div>
            <div class="track-info-value">${track.stability_label}</div>
        </div>
    `;
}

let waveAnimationFrame = null;

const waveMetrics = [
    { key: 'EI', label: 'Focus Index', color: '#00d4ff', lineWidth: 3 },
    { key: 'alpha', label: 'Alpha (8-13 Hz)', color: '#ff8fc4', lineWidth: 1.5 },
    { key: 'beta', label: 'Beta (13-30 Hz)', color: '#9a7bff', lineWidth: 1.5 },
    { key: 'theta', label: 'Theta (4-8 Hz)', color: '#ffcc6e', lineWidth: 1.5 }
];

function renderWaveLegend() {
    const legend = document.getElementById('wave-legend');
    legend.innerHTML = waveMetrics.map((m) => `
        <span><span class="swatch" style="background:${m.color}"></span>${m.label}</span>
    `).join('');
    legend.style.display = 'none';
}

async function renderTrackDetail(track) {
    console.log('Loading window data for clip:', track.clip_id);
    
    document.getElementById('wave-placeholder').classList.add('hidden');
    
    try {
        const response = await fetch('./data/ai_windows.csv');
        if (!response.ok) throw new Error(`Failed to load CSV: ${response.status}`);
        const csvText = await response.text();
        const lines = csvText.split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        const clipWindows = [];
        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            const values = lines[i].split(',');
            const row = {};
            headers.forEach((header, idx) => { row[header] = values[idx]; });
            const rowClipId = row.clip_id ? row.clip_id.trim() : '';
            if (rowClipId === track.clip_id) clipWindows.push(row);
        }
        
        console.log(`Found ${clipWindows.length} windows for clip ${track.clip_id}`);
        if (clipWindows.length === 0) {
            showWaveError('No window data available for this track.');
            return;
        }
        
        clipWindows.sort((a, b) => parseFloat(a.t0) - parseFloat(b.t0));
        
        const waveData = clipWindows.map(w => ({
            t0: parseFloat(w.t0),
            EI: parseFloat(w.EI),
            alpha: parseFloat(w.alpha),
            beta: parseFloat(w.beta),
            theta: parseFloat(w.theta)
        }));
        
        startWaveAnimation(waveData);
    } catch (error) {
        console.error('Error loading window data:', error);
        showWaveError('Error loading waveform data.');
    }
}

function showWaveError(message) {
    const canvas = document.getElementById('wave-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#808080';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function startWaveAnimation(waveData) {
    cancelAnimationFrame(waveAnimationFrame);
    const canvas = document.getElementById('wave-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('wave-legend').style.display = 'flex';
    
    if (!waveData.length) {
        showWaveError('No window data available for this clip.');
        return;
    }
    
    const normalized = waveMetrics.map((m) => normalizeSeries(waveData.map((row) => row[m.key])));
    const times = waveData.map((row) => row.t0);
    const totalFrames = Math.max(60, times.length);
    let wavePointer = 0;
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const visible = Math.min(times.length, 600);
        const startIndex = Math.max(0, wavePointer - visible);
        const sliceTimes = times.slice(startIndex, wavePointer);
        
        waveMetrics.forEach((metric, idx) => {
            const series = normalized[idx].slice(startIndex, wavePointer);
            drawSeries(ctx, sliceTimes, series, metric.color, canvas, metric.lineWidth || 2);
        });
        
        wavePointer += 1;
        if (wavePointer < totalFrames) {
            waveAnimationFrame = requestAnimationFrame(draw);
        }
    }
    
    wavePointer = Math.min(600, times.length);
    waveAnimationFrame = requestAnimationFrame(draw);
}

function drawSeries(ctx, times, series, color, canvas, lineWidth = 2) {
    if (series.length === 0) return;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    const padding = 20;
    const height = canvas.height - padding * 2;
    const width = canvas.width - padding * 2;
    const step = width / Math.max(1, series.length - 1);
    
    series.forEach((value, idx) => {
        const x = padding + idx * step;
        const y = padding + height * (1 - value);
        if (idx === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
}

function normalizeSeries(values) {
    const finiteValues = values.filter((v) => Number.isFinite(v));
    if (finiteValues.length === 0) {
        return values.map(() => 0.5);
    }
    const min = Math.min(...finiteValues);
    const max = Math.max(...finiteValues);
    const range = max - min || 1;
    return values.map((v) => {
        if (!Number.isFinite(v)) return 0.5;
        return (v - min) / range;
    });
}

window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    const blobElement = document.querySelector('body::before');
    if (document.body.style) {
        const offsetX = Math.sin(scrollY * 0.005) * 120;
        const offsetY = Math.cos(scrollY * 0.007) * 120;
        document.body.style.setProperty('--blob-offset-x', `${offsetX}px`);
        document.body.style.setProperty('--blob-offset-y', `${offsetY}px`);
    }
});

let loadingWaveAnimationFrame = null;
let currentLoadingProgress = 0;

function drawLoadingWaveform() {
    const canvas = document.getElementById('loading-waveform');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerY = height / 2;
    const time = Date.now() * 0.003;
    
    ctx.clearRect(0, 0, width, height);
    
    const points = [];
    for (let x = 0; x < width; x++) {
        const normalizedX = (x / width) * 20 * Math.PI;
        const amp1 = Math.sin(normalizedX * 0.5 + time) * 0.2;
        const amp2 = Math.sin(normalizedX * 1.2 + time * 1.3) * 0.15;
        const amp3 = Math.sin(normalizedX * 2.5 + time * 0.7) * 0.1;
        const amp4 = Math.sin(normalizedX * 3.7 + time * 1.1) * 0.08;
        const totalAmp = (amp1 + amp2 + amp3 + amp4) * height;
        const y = centerY + totalAmp;
        points.push({ x, y });
    }
    
    ctx.strokeStyle = '#808080';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    points.forEach((point, i) => {
        if (i === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    });
    ctx.stroke();
    
    const fillWidth = (width * currentLoadingProgress) / 100;
    ctx.strokeStyle = 'rgba(200, 0, 255, 0.9)';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    
    for (let i = 0; i < points.length && points[i].x <= fillWidth; i++) {
        if (i === 0) {
            ctx.moveTo(points[i].x, points[i].y);
        } else {
            ctx.lineTo(points[i].x, points[i].y);
        }
    }
    ctx.stroke();
    
    loadingWaveAnimationFrame = requestAnimationFrame(drawLoadingWaveform);
}

function startLoadingAnimation() {
    currentLoadingProgress = 0;
    drawLoadingWaveform();
}

document.addEventListener('DOMContentLoaded', () => { 
    console.log('NeuroFocus initialized');
});
