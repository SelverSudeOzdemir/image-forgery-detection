// === STATE ===
let selectedFile = null,
  uploadedFilename = null;

// === ELEMENTS ===
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const previewName = document.getElementById("previewName");
const previewSize = document.getElementById("previewSize");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorBox = document.getElementById("errorBox");
const errorMsg = document.getElementById("errorMsg");

// === DRAG & DROP ===
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("drag-over")
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) handleFile(f);
});

function handleFile(f) {
  const allowed = [
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/tiff",
  ];
  if (
    !allowed.includes(f.type) &&
    !f.name.match(/\.(tif|tiff|gif|bmp|jpg|jpeg|png|webp)$/i)
  ) {
    showError(
      "Desteklenmeyen format. PNG, JPG, GIF, BMP, TIFF veya WEBP seçin."
    );
    return;
  }
  selectedFile = f;
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImg.src = ev.target.result;
    previewName.textContent = f.name;
    previewSize.textContent = formatSize(f.size);
    preview.classList.remove("hidden");
    analyzeBtn.disabled = false;
    results.classList.add("hidden");
    errorBox.classList.add("hidden");
  };
  reader.readAsDataURL(f);
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// === ANALYZE ===
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  results.classList.add("hidden");
  errorBox.classList.add("hidden");
  loading.classList.remove("hidden");
  analyzeBtn.disabled = true;

  // Reset algo steps
  ["alg1", "alg2", "alg3"].forEach((id) => {
    const el = document.getElementById(id);
    el.classList.remove("active", "done");
  });
  ["s1", "s2", "s3"].forEach((id) => {
    document.getElementById(id).textContent = "⏳";
  });

  try {
    // Upload
    const fd = new FormData();
    fd.append("file", selectedFile);
    const ur = await fetch("/api/upload", { method: "POST", body: fd });
    const ud = await ur.json();
    if (!ud.success) {
      showError(ud.message);
      return;
    }
    uploadedFilename = ud.filename;

    // Trigger animation steps
    document.getElementById("alg1").classList.add("active");
    setTimeout(() => {
      document.getElementById("alg1").classList.add("done");
      document.getElementById("s1").textContent = "✅";
      document.getElementById("alg2").classList.add("active");
    }, 600);
    setTimeout(() => {
      document.getElementById("alg2").classList.add("done");
      document.getElementById("s2").textContent = "✅";
      document.getElementById("alg3").classList.add("active");
    }, 1400);

    // Analyze
    const ar = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: uploadedFilename }),
    });
    const ad = await ar.json();

    document.getElementById("alg3").classList.add("done");
    document.getElementById("s3").textContent = "✅";

    loading.classList.add("hidden");
    if (!ad.success) {
      showError(ad.message);
      return;
    }

    displayResults(ad.results);
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError("Bağlantı hatası: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// === DISPLAY RESULTS ===
function displayResults(d) {
  displayVerdict(d.final_decision);
  displayVotes(d.final_decision.votes);
  displayTraditional(d.traditional);
  displayYolo(d.yolo);
  displayAI(d.ai);
  displayTable(d);
}

function displayVerdict(d) {
  const block = document.querySelector(".verdict-block");
  const mainEl = document.getElementById("verdictMain");
  const agreeEl = document.getElementById("verdictAgreement");
  const ringFill = document.getElementById("ringFill");
  const confVal = document.getElementById("confValue");

  const isFake = d.is_fake;
  block.className = "verdict-block " + (isFake ? "is-fake" : "is-real");
  mainEl.textContent = isFake ? "MANİPÜLE EDİLMİŞ" : "ORİJİNAL";
  mainEl.className = "verdict-main " + (isFake ? "fake" : "real");
  agreeEl.textContent = d.agreement;

  // Animate ring
  const pct = d.confidence;
  const circumference = 314;
  const offset = circumference - (pct / 100) * circumference;
  setTimeout(() => {
    ringFill.style.strokeDashoffset = offset;
  }, 100);

  // Animate counter
  let start = 0;
  const target = Math.round(pct);
  const step = Math.ceil(target / 40);
  const timer = setInterval(() => {
    start = Math.min(start + step, target);
    confVal.textContent = start;
    if (start >= target) clearInterval(timer);
  }, 25);
}

function displayVotes(votes) {
  const pills = {
    voteTraditional: votes.traditional,
    voteYolo: votes.yolo,
    voteAI: votes.ai,
  };
  for (const [id, isFake] of Object.entries(pills)) {
    const el = document.getElementById(id);
    el.className = "vote-pill " + (isFake ? "fake" : "real");
    el.querySelector(".vote-icon").textContent = isFake ? "✗" : "✓";
  }
}

function metricRow(name, valueHTML, extraText = "", cls = "") {
  return `<div class="metric-row ${cls}">
        <div class="metric-name">${name}</div>
        <div class="metric-val ${cls}">${valueHTML}</div>
        ${extraText ? `<div class="metric-extra">${extraText}</div>` : ""}
    </div>`;
}

function fakeClass(isFake) {
  return isFake ? "fake" : "real";
}
function fakeLabel(isFake) {
  return isFake ? "✗ SAHTE" : "✓ ORİJİNAL";
}

function displayTraditional(d) {
  const el = document.getElementById("tradRows");
  let html = "";
  ["sift", "surf", "akaze", "orb"].forEach((k) => {
    const r = d[k];
    const fc = fakeClass(r.is_fake);
    html += metricRow(
      r.algorithm,
      fakeLabel(r.is_fake),
      `Güven: ${r.confidence.toFixed(1)}%  Eşleşme: ${r.match_count}  Bölge: ${
        r.suspicious_regions
      }`,
      fc
    );
  });
  const o = d.overall;
  html += metricRow(
    "GENEL KARAR",
    fakeLabel(o.is_fake),
    `Oy: ${o.votes}/4  Güven: ${o.confidence.toFixed(1)}%`,
    "overall"
  );
  el.innerHTML = html;
}

function displayYolo(d) {
  const el = document.getElementById("yoloRows");
  const fc = fakeClass(d.is_fake);
  let html = metricRow(
    "YOLO CASIA",
    fakeLabel(d.is_fake),
    `Sınıf: ${d.predicted_label} (${d.yolo_confidence?.toFixed(1)}%)`,
    fc
  );
  if (d.ela)
    html += metricRow(
      "ELA Skoru",
      `${d.ela.manipulation_score.toFixed(1)}%`,
      `Blok Var: ${d.ela.block_variance?.toFixed(3)}`,
      d.ela.is_suspicious ? "fake" : "real"
    );
  if (d.noise)
    html += metricRow(
      "Gürültü Var.",
      d.noise.noise_variation.toFixed(3),
      `Ort: ${d.noise.noise_mean?.toFixed(3)}`,
      d.noise.is_suspicious ? "fake" : "real"
    );
  if (d.quality)
    html += metricRow(
      "Görüntü Kalitesi",
      `${d.quality.quality_score.toFixed(1)}%`,
      `Keskinlik: ${d.quality.sharpness?.toFixed(
        1
      )}  Kontrast: ${d.quality.contrast?.toFixed(1)}`
    );
  el.innerHTML = html;
}

function displayAI(d) {
  const el = document.getElementById("aiRows");
  let html = "";
  if (d.cnn) {
    const fc = fakeClass(d.cnn.is_fake);
    html += metricRow(
      "CNN",
      fakeLabel(d.cnn.is_fake),
      `Güven: ${d.cnn.confidence.toFixed(
        1
      )}%  Kenar: ${d.cnn.edge_inconsistency?.toFixed(2)}`,
      fc
    );
  }
  if (d.lstm) {
    const fc = fakeClass(d.lstm.is_fake);
    html += metricRow(
      "LSTM",
      fakeLabel(d.lstm.is_fake),
      `Güven: ${d.lstm.confidence.toFixed(
        1
      )}%  Ghost: ${d.lstm.ghost_variation?.toFixed(3)}`,
      fc
    );
  }
  if (d.overall) {
    html += metricRow(
      "GENEL KARAR",
      fakeLabel(d.overall.is_fake),
      `Oy: ${d.overall.votes}/2  Güven: ${d.overall.confidence.toFixed(1)}%`,
      "overall"
    );
  }
  el.innerHTML = html;
}

function displayTable(d) {
  const rows = [
    {
      n: "SIFT",
      c: d.traditional.sift.confidence,
      f: d.traditional.sift.is_fake,
      detail: `${d.traditional.sift.match_count} eşleşme`,
    },
    {
      n: "SURF",
      c: d.traditional.surf.confidence,
      f: d.traditional.surf.is_fake,
      detail: `${d.traditional.surf.match_count} eşleşme`,
    },
    {
      n: "AKAZE",
      c: d.traditional.akaze.confidence,
      f: d.traditional.akaze.is_fake,
      detail: `${d.traditional.akaze.match_count} eşleşme`,
    },
    {
      n: "ORB",
      c: d.traditional.orb.confidence,
      f: d.traditional.orb.is_fake,
      detail: `${d.traditional.orb.match_count} eşleşme`,
    },
    {
      n: "YOLOv8 CASIA",
      c: d.yolo.confidence,
      f: d.yolo.is_fake,
      detail: `Sınıf: ${d.yolo.predicted_label}`,
    },
    {
      n: "ELA",
      c: d.yolo.ela?.manipulation_score ?? 0,
      f: d.yolo.ela?.is_suspicious ?? false,
      detail: `Blok Var: ${d.yolo.ela?.block_variance?.toFixed(3) ?? "-"}`,
    },
    {
      n: "CNN",
      c: d.ai.cnn?.confidence ?? 0,
      f: d.ai.cnn?.is_fake ?? false,
      detail: `Skor: ${d.ai.cnn?.raw_score?.toFixed(4) ?? "-"}`,
    },
    {
      n: "LSTM",
      c: d.ai.lstm?.confidence ?? 0,
      f: d.ai.lstm?.is_fake ?? false,
      detail: `Skor: ${d.ai.lstm?.raw_score?.toFixed(4) ?? "-"}`,
    },
  ];

  const body = document.getElementById("tableBody");
  body.innerHTML = rows
    .map(
      (r) => `
        <tr>
            <td class="td-algo">${r.n}</td>
            <td class="td-conf">${r.c.toFixed(1)}</td>
            <td class="td-decision ${r.f ? "fake" : "real"}">${
        r.f ? "✗ SAHTE" : "✓ ORİJİNAL"
      }</td>
            <td class="td-detail">${r.detail}</td>
        </tr>
    `
    )
    .join("");
}

// === ERROR ===
function showError(msg) {
  errorMsg.textContent = msg;
  errorBox.classList.remove("hidden");
  loading.classList.add("hidden");
  analyzeBtn.disabled = false;
}
document
  .getElementById("closeError")
  .addEventListener("click", () => errorBox.classList.add("hidden"));

// === RESET ===
document.getElementById("resetBtn").addEventListener("click", () => {
  if (uploadedFilename) {
    fetch("/api/cleanup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: uploadedFilename }),
    });
  }
  selectedFile = null;
  uploadedFilename = null;
  fileInput.value = "";
  previewImg.src = "";
  preview.classList.add("hidden");
  results.classList.add("hidden");
  analyzeBtn.disabled = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// === DOWNLOAD ===
document.getElementById("downloadBtn").addEventListener("click", () => {
  const stamp = new Date().toLocaleString("tr-TR");
  const tableEl = document.getElementById("detailTable")?.outerHTML ?? "";
  const verdictText = document.getElementById("verdictMain")?.textContent ?? "";
  const conf = document.getElementById("confValue")?.textContent ?? "";
  const agreement =
    document.getElementById("verdictAgreement")?.textContent ?? "";

  const html = `<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>ForensicAI Raporu — ${stamp}</title>
<style>
body{font-family:'Segoe UI',sans-serif;background:#0b0c0e;color:#e8eaed;margin:0;padding:40px}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:1.6rem;border-bottom:2px solid #c8f542;padding-bottom:12px;margin-bottom:24px;color:#c8f542}
.verdict{padding:24px;border:1px solid;border-radius:8px;margin-bottom:24px}
.verdict.fake{border-color:#ff4d6d;background:rgba(255,77,109,0.08)}
.verdict.real{border-color:#3dffa0;background:rgba(61,255,160,0.08)}
.v-main{font-size:2rem;font-weight:800;margin-bottom:8px}
.v-main.fake{color:#ff4d6d}.v-main.real{color:#3dffa0}
.v-meta{font-size:0.8rem;color:#6b7280;font-family:monospace}
table{width:100%;border-collapse:collapse;margin-top:24px}
th{background:#181b20;padding:10px 16px;text-align:left;font-size:0.75rem;letter-spacing:0.06em;color:#6b7280;border-bottom:1px solid rgba(255,255,255,0.07)}
td{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.85rem}
.fake{color:#ff4d6d}.real{color:#3dffa0}
.footer{margin-top:32px;font-size:0.72rem;color:#374151;font-family:monospace;border-top:1px solid rgba(255,255,255,0.07);padding-top:16px}
</style>
</head><body><div class="wrap">
<h1>[FORENSICAI] Görüntü Analiz Raporu</h1>
<div class="verdict ${verdictText.includes("MANİPÜLE") ? "fake" : "real"}">
<div class="v-main ${
    verdictText.includes("MANİPÜLE") ? "fake" : "real"
  }">${verdictText}</div>
<div class="v-meta">Güven: ${conf}% · ${agreement}</div>
</div>
${tableEl}
<div class="footer">ForensicAI Ar-Ge Projesi · Rapor Tarihi: ${stamp}</div>
</div></body></html>`;

  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ForensicAI_Rapor_${Date.now()}.html`;
  a.click();
});
