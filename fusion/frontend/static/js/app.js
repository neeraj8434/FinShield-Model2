(function () {
  "use strict";

  // ── Three.js: Background wireframe shapes on hero-canvas ──────────
  function initHeroBackground() {
    var canvas = document.getElementById("hero-canvas");
    if (!canvas || typeof THREE === "undefined") return;

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 30;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    var goldColor = 0xc8a951;
    var dimGold = 0x6b5a2e;

    // Grid
    var gridMat = new THREE.LineBasicMaterial({ color: goldColor, transparent: true, opacity: 0.03 });
    for (var i = -30; i <= 30; i += 8) {
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(i, -20, -15), new THREE.Vector3(i, 20, -15)]), gridMat));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-30, i * 0.66, -15), new THREE.Vector3(30, i * 0.66, -15)]), gridMat));
    }

    // Particles
    var pGeo = new THREE.BufferGeometry();
    var pos = new Float32Array(60 * 3);
    for (var p = 0; p < 180; p++) pos[p] = (Math.random() - 0.5) * 60;
    pGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(pGeo, new THREE.PointsMaterial({ color: goldColor, size: 0.06, transparent: true, opacity: 0.3 })));

    var mouseX = 0, mouseY = 0;
    document.addEventListener("mousemove", function (e) {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
      mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    (function animate() {
      requestAnimationFrame(animate);
      camera.position.x += (mouseX * 1.5 - camera.position.x) * 0.015;
      camera.position.y += (-mouseY * 1 - camera.position.y) * 0.015;
      camera.lookAt(scene.position);
      renderer.render(scene, camera);
    })();

    window.addEventListener("resize", function () {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  // ── Three.js: Face scan 3D visualization ──────────────────────────
  function initFaceScan() {
    var canvas = document.getElementById("face-scan-canvas");
    if (!canvas || typeof THREE === "undefined") return;

    var w = 500, h = 500;
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 4.5);

    var renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    var goldColor = 0xc8a951;
    var creamColor = 0xf5e6c8;

    // Face wireframe — sphere with distortion to look like a head
    var headGeo = new THREE.SphereGeometry(1.2, 24, 24);
    var verts = headGeo.attributes.position;
    for (var vi = 0; vi < verts.count; vi++) {
      var y = verts.getY(vi);
      var z = verts.getZ(vi);
      // Chin taper
      if (y < -0.3) {
        var t = Math.abs(y + 0.3) / 0.9;
        verts.setX(vi, verts.getX(vi) * (1 - t * 0.4));
        verts.setZ(vi, z * (1 - t * 0.3));
      }
      // Forehead slight bulge
      if (y > 0.5) {
        verts.setZ(vi, z + 0.1);
      }
    }
    headGeo.computeVertexNormals();

    var headMat = new THREE.MeshBasicMaterial({ color: goldColor, wireframe: true, transparent: true, opacity: 0.3 });
    var head = new THREE.Mesh(headGeo, headMat);
    scene.add(head);

    // Nose ridge line
    var nosePts = [new THREE.Vector3(0, 0.1, 1.25), new THREE.Vector3(0, -0.25, 1.35), new THREE.Vector3(0, -0.4, 1.2)];
    scene.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(nosePts),
      new THREE.LineBasicMaterial({ color: goldColor, transparent: true, opacity: 0.5 })
    ));

    // Eye markers
    [[-0.35, 0.15], [0.35, 0.15]].forEach(function (pos) {
      var eyeGeo = new THREE.RingGeometry(0.08, 0.12, 16);
      var eye = new THREE.Mesh(eyeGeo, new THREE.MeshBasicMaterial({ color: creamColor, transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
      eye.position.set(pos[0], pos[1], 1.15);
      scene.add(eye);
    });

    // Mouth line
    var mouthCurve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(-0.25, -0.55, 1.1),
      new THREE.Vector3(0, -0.6, 1.15),
      new THREE.Vector3(0.25, -0.55, 1.1)
    );
    scene.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(mouthCurve.getPoints(12)),
      new THREE.LineBasicMaterial({ color: creamColor, transparent: true, opacity: 0.4 })
    ));

    // Scanning ring
    var ringGeo = new THREE.TorusGeometry(1.6, 0.015, 8, 64);
    var ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: goldColor, transparent: true, opacity: 0.5 }));
    scene.add(ring);

    // Scan line (horizontal)
    var scanLineMat = new THREE.LineBasicMaterial({ color: 0x3ecf8e, transparent: true, opacity: 0.6 });
    var scanLineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1.8, 0, 0.5), new THREE.Vector3(1.8, 0, 0.5)]);
    var scanLine = new THREE.Line(scanLineGeo, scanLineMat);
    scene.add(scanLine);

    // Corner bracket decorations
    var bracketMat = new THREE.LineBasicMaterial({ color: goldColor, transparent: true, opacity: 0.6 });
    var bSize = 1.8;
    var bLen = 0.4;
    [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(function (c) {
      var pts = [
        new THREE.Vector3(c[0] * bSize, c[1] * (bSize - bLen), 0.3),
        new THREE.Vector3(c[0] * bSize, c[1] * bSize, 0.3),
        new THREE.Vector3(c[0] * (bSize - bLen), c[1] * bSize, 0.3)
      ];
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), bracketMat));
    });

    // Data points floating around
    var dpGeo = new THREE.BufferGeometry();
    var dpPos = new Float32Array(40 * 3);
    for (var di = 0; di < 120; di++) {
      dpPos[di] = (Math.random() - 0.5) * 4;
    }
    dpGeo.setAttribute("position", new THREE.BufferAttribute(dpPos, 3));
    scene.add(new THREE.Points(dpGeo, new THREE.PointsMaterial({ color: goldColor, size: 0.03, transparent: true, opacity: 0.5 })));

    (function animate() {
      requestAnimationFrame(animate);
      var t = Date.now() * 0.001;
      head.rotation.y = Math.sin(t * 0.3) * 0.3;
      head.rotation.x = Math.sin(t * 0.2) * 0.05;
      ring.rotation.x = t * 0.4;
      ring.rotation.z = t * 0.15;
      // Scan line sweeps up and down
      var scanY = Math.sin(t * 1.2) * 1.5;
      scanLine.position.y = scanY;
      scanLineMat.opacity = 0.3 + Math.abs(Math.sin(t * 1.2)) * 0.4;
      renderer.render(scene, camera);
    })();
  }

  // ── Navbar scroll ─────────────────────────────────────────────────
  function initNavbar() {
    var nav = document.querySelector(".navbar");
    if (!nav) return;
    window.addEventListener("scroll", function () {
      nav.classList.toggle("scrolled", window.scrollY > 60);
    });
  }

  // ── Scroll reveal ─────────────────────────────────────────────────
  function initReveal() {
    var items = document.querySelectorAll(".reveal");
    if (!items.length) return;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) e.target.classList.add("visible"); });
    }, { threshold: 0.15 });
    items.forEach(function (el) { obs.observe(el); });
  }

  // ── Input masks ───────────────────────────────────────────────────
  function initInputMasks() {
    var inp = document.getElementById("form_id_number");
    var sel = document.getElementById("doc_type");
    if (!inp || !sel) return;
    var hint = document.getElementById("id-hint");

    function updateMask() {
      if (sel.value === "aadhaar") {
        inp.placeholder = "0000 0000 0000";
        inp.maxLength = 14;
        inp.setAttribute("data-mask", "aadhaar");
        if (hint) hint.textContent = "12-digit Aadhaar number";
      } else {
        inp.placeholder = "ABCDE1234F";
        inp.maxLength = 10;
        inp.setAttribute("data-mask", "pan");
        if (hint) hint.textContent = "10-character PAN (e.g. ABCDE1234F)";
      }
      inp.value = "";
    }
    sel.addEventListener("change", updateMask);
    updateMask();

    inp.addEventListener("input", function () {
      var mask = this.getAttribute("data-mask");
      var val = this.value;
      if (mask === "aadhaar") {
        val = val.replace(/[^0-9]/g, "").substring(0, 12);
        var fmt = "";
        for (var i = 0; i < val.length; i++) {
          if (i > 0 && i % 4 === 0) fmt += " ";
          fmt += val[i];
        }
        this.value = fmt;
      } else {
        val = val.toUpperCase().replace(/[^A-Z0-9]/g, "").substring(0, 10);
        var pf = "";
        for (var j = 0; j < val.length; j++) {
          if (j < 5) pf += val[j].replace(/[^A-Z]/g, "");
          else if (j < 9) pf += val[j].replace(/[^0-9]/g, "");
          else pf += val[j].replace(/[^A-Z]/g, "");
        }
        this.value = pf;
      }
    });
  }

  // ── Upload zones ──────────────────────────────────────────────────
  function initUploadZones() {
    document.querySelectorAll(".upload-zone").forEach(function (zone) {
      var input = zone.querySelector("input[type='file']");
      var filenameEl = zone.querySelector(".upload-filename");
      var previewEl = zone.querySelector(".upload-preview");

      zone.addEventListener("dragover", function (e) { e.preventDefault(); zone.classList.add("dragover"); });
      zone.addEventListener("dragleave", function () { zone.classList.remove("dragover"); });
      zone.addEventListener("drop", function (e) {
        e.preventDefault(); zone.classList.remove("dragover");
        if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; showFile(e.dataTransfer.files[0]); }
      });
      if (input) input.addEventListener("change", function () { if (this.files && this.files[0]) showFile(this.files[0]); });

      function showFile(file) {
        zone.classList.add("has-file");
        if (filenameEl) filenameEl.textContent = file.name;
        if (previewEl && file.type.startsWith("image/")) {
          var r = new FileReader();
          r.onload = function (e) { previewEl.src = e.target.result; previewEl.style.display = "block"; };
          r.readAsDataURL(file);
        }
      }
    });
  }

  // ── Form submission ───────────────────────────────────────────────
  function initForm() {
    var form = document.getElementById("kyc-form");
    var btn = document.getElementById("submit-btn");
    if (!form || !btn) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var docImage = document.getElementById("doc_image");
      var video = document.getElementById("video");
      if (!docImage.files[0]) { showToast("Please upload a document image", "error"); return; }
      if (!video.files[0]) { showToast("Please upload a video", "error"); return; }

      btn.classList.add("loading"); btn.disabled = true;
      var fd = new FormData();
      fd.append("doc_image", docImage.files[0]);
      fd.append("video", video.files[0]);
      fd.append("doc_type", document.getElementById("doc_type").value);
      fd.append("form_name", document.getElementById("form_name").value.trim());

      var dateVal = document.getElementById("form_dob").value;
      if (dateVal) {
        var parts = dateVal.split("-");
        fd.append("form_dob", parts[2] + "/" + parts[1] + "/" + parts[0]);
      }
      fd.append("form_id_number", document.getElementById("form_id_number").value.replace(/\s/g, ""));

      fetch("/api/predict", { method: "POST", body: fd })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          btn.classList.remove("loading"); btn.disabled = false;
          if (data.error) { showToast("Error: " + data.error, "error"); return; }
          renderResults(data);
          loadHistory();
        })
        .catch(function (err) {
          btn.classList.remove("loading"); btn.disabled = false;
          showToast("Request failed: " + err.message, "error");
        });
    });
  }

  // ── Render results ────────────────────────────────────────────────
  function renderResults(data) {
    var section = document.getElementById("results-section");
    if (!section) return;
    section.style.display = "block";
    section.scrollIntoView({ behavior: "smooth", block: "start" });

    var decision = data.decision || "REVIEW";
    var bannerEl = document.getElementById("decision-banner");
    bannerEl.className = "decision-banner " + decision;
    document.getElementById("decision-label").textContent = "VERIFICATION RESULT";
    var verdictMap = { APPROVE: "APPROVED", REVIEW: "MANUAL REVIEW", REJECT: "REJECTED" };
    document.getElementById("decision-verdict").textContent = verdictMap[decision] || decision;

    var risk = (data.fused_risk != null ? data.fused_risk : 0.5);
    var riskPct = Math.round(risk * 100);
    var riskFill = document.getElementById("risk-fill");
    riskFill.style.width = riskPct + "%";
    riskFill.className = "risk-gauge-fill " + (riskPct < 35 ? "low" : riskPct < 65 ? "medium" : "high");
    document.getElementById("risk-value").textContent = riskPct + "%";

    renderModality("mod-doc", data.document);
    renderModality("mod-video", data.video);
    renderModality("mod-semantic", data.semantic);
    renderModality("mod-face", data.face_match);

    // Modality scores / weights
    var wc = document.getElementById("weights-container");
    if (wc && data.modality_scores && data.modality_scores.length) {
      wc.innerHTML = "";
      var wNames = { document: "Document Forgery", video: "Video Deepfake" };
      data.modality_scores.forEach(function (ms) {
        var row = document.createElement("div"); row.className = "weights-row";
        row.innerHTML = '<span class="weights-name">' + (wNames[ms.name] || ms.name) +
          '</span><span class="weights-val">P(fake)=' + (ms.fake_probability * 100).toFixed(1) +
          '% | Reliability=' + (ms.reliability * 100).toFixed(0) +
          '% | Weight=' + (ms.weight * 100).toFixed(0) + '%</span>';
        wc.appendChild(row);
      });
    }

    // Semantic checks
    var cc = document.getElementById("checks-container");
    if (cc && data.semantic && data.semantic.checks) {
      cc.innerHTML = "";
      data.semantic.checks.forEach(function (c) {
        var cls = "check-item ", icon = "";
        if (c.status === "pass") { cls += "check-pass"; icon = "P"; }
        else if (c.status === "fail") { cls += "check-fail"; icon = "F"; }
        else if (c.status === "warn") { cls += "check-warn"; icon = "W"; }
        else { cls += "check-unknown"; icon = "?"; }
        var div = document.createElement("div"); div.className = cls;
        div.innerHTML = '<div class="check-status">' + icon + '</div><div><span class="check-field">' + c.field + '</span> <span class="check-detail">' + esc(c.detail) + '</span></div>';
        cc.appendChild(div);
      });
    }

    // Face match detail
    var fc = document.getElementById("face-container");
    if (fc && data.face_match) {
      var fm = data.face_match;
      var fmS = (fm.status || "").toLowerCase();
      var fCls = "face-match-result ", fLbl = "";
      if (fmS === "match") { fCls += "match"; fLbl = "FACE MATCHED"; }
      else if (fmS === "mismatch") { fCls += "mismatch"; fLbl = "FACE MISMATCH"; }
      else { fCls += "unavailable"; fLbl = "UNAVAILABLE"; }
      var fDet = fm.similarity != null ? "Cosine similarity: " + fm.similarity.toFixed(4) + " (threshold: " + fm.threshold + ")" : (fm.detail || "");
      fc.innerHTML = '<div class="' + fCls + '"><div><div class="face-match-label">' + fLbl + '</div><div class="face-match-detail">' + esc(fDet) + '</div></div></div>';
    }

    // Reason codes
    var rc = document.getElementById("reason-container");
    if (rc && data.reason_codes && data.reason_codes.length) {
      rc.innerHTML = ""; rc.parentElement.style.display = "block";
      data.reason_codes.forEach(function (r) {
        var d = document.createElement("div"); d.className = "reason-item"; d.textContent = r; rc.appendChild(d);
      });
    } else if (rc) { rc.parentElement.style.display = "none"; }

    // XAI
    var xc = document.getElementById("xai-container");
    if (xc && data.document && data.document.xai_b64) {
      xc.parentElement.style.display = "block";
      xc.innerHTML = '<img class="xai-image" src="' + data.document.xai_b64 + '" alt="Explainability heatmap">';
    } else if (xc) { xc.parentElement.style.display = "none"; }
  }

  function renderModality(id, mod) {
    var el = document.getElementById(id);
    if (!el) return;
    var verdict = el.querySelector(".modality-verdict");
    var meta = el.querySelector(".modality-meta");
    if (!mod) { verdict.textContent = "N/A"; verdict.className = "modality-verdict verdict-na"; meta.textContent = ""; return; }

    if (id === "mod-doc") {
      var isReal = (mod.verdict || "").toUpperCase() === "REAL";
      verdict.textContent = isReal ? "AUTHENTIC" : "FORGED";
      verdict.className = "modality-verdict " + (isReal ? "verdict-real" : "verdict-fake");
      meta.textContent = "Confidence: " + ((mod.confidence || 0) * 100).toFixed(1) + "%";
    } else if (id === "mod-video") {
      var vSt = (mod.status || "").toLowerCase();
      if (vSt === "error" || vSt === "insufficient_evidence") {
        verdict.textContent = vSt === "error" ? "ERROR" : "INSUFFICIENT";
        verdict.className = "modality-verdict verdict-na"; meta.textContent = mod.detail || "";
      } else {
        var vReal = (mod.prediction || "").toUpperCase() === "REAL";
        verdict.textContent = vReal ? "AUTHENTIC" : "DEEPFAKE";
        verdict.className = "modality-verdict " + (vReal ? "verdict-real" : "verdict-fake");
        meta.textContent = "Confidence: " + ((mod.confidence || 0) * 100).toFixed(1) + "%";
      }
    } else if (id === "mod-semantic") {
      var hasFail = mod.has_critical_fail;
      verdict.textContent = hasFail ? "MISMATCH FOUND" : "CONSISTENT";
      verdict.className = "modality-verdict " + (hasFail ? "verdict-fake" : "verdict-real");
      meta.textContent = (mod.checks || []).length + " checks performed";
    } else if (id === "mod-face") {
      var fs = (mod.status || "").toLowerCase();
      if (fs === "match") { verdict.textContent = "MATCHED"; verdict.className = "modality-verdict verdict-real"; }
      else if (fs === "mismatch") { verdict.textContent = "MISMATCH"; verdict.className = "modality-verdict verdict-fake"; }
      else { verdict.textContent = "N/A"; verdict.className = "modality-verdict verdict-na"; }
      meta.textContent = mod.similarity != null ? "Similarity: " + mod.similarity.toFixed(4) : (mod.detail || "");
    }
  }

  // ── History ───────────────────────────────────────────────────────
  var historyData = [];
  function initHistory() {
    var tb = document.getElementById("history-toggle");
    var cb = document.getElementById("sidebar-close");
    var ov = document.getElementById("sidebar-overlay");
    var cl = document.getElementById("sidebar-clear");
    if (tb) tb.addEventListener("click", toggleSidebar);
    if (cb) cb.addEventListener("click", closeSidebar);
    if (ov) ov.addEventListener("click", closeSidebar);
    if (cl) cl.addEventListener("click", clearAllHistory);
    loadHistory();
  }
  function toggleSidebar() {
    document.getElementById("history-sidebar").classList.toggle("open");
    document.getElementById("sidebar-overlay").classList.toggle("show");
  }
  function closeSidebar() {
    document.getElementById("history-sidebar").classList.remove("open");
    document.getElementById("sidebar-overlay").classList.remove("show");
  }
  function loadHistory() {
    fetch("/api/history").then(function (r) { return r.json(); }).then(function (d) {
      historyData = d; renderHistory(); updateHistoryCount();
    }).catch(function () {});
  }
  function renderHistory() {
    var body = document.getElementById("sidebar-body");
    if (!body) return;
    if (!historyData.length) { body.innerHTML = '<div class="sidebar-empty">No verification history yet.</div>'; return; }
    body.innerHTML = "";
    historyData.forEach(function (item) {
      var card = document.createElement("div"); card.className = "h-card";
      var decision = item.decision || "REVIEW";
      var thumbHtml = item.thumbnail ? '<img class="h-card-thumb" src="' + item.thumbnail + '" alt="">' : '<div class="h-card-thumb"></div>';
      var dateStr = "";
      if (item.timestamp) {
        var d = new Date(item.timestamp * 1000);
        dateStr = d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) + " " + d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
      }
      card.innerHTML =
        '<div class="h-card-top">' + thumbHtml +
        '<div class="h-card-info"><div class="h-card-name">' + esc(item.form_name || "Unknown") + '</div>' +
        '<div class="h-card-meta">' + esc(item.doc_type || "").toUpperCase() + '</div>' +
        '<div class="h-card-badge h-badge-' + decision + '">' + decision + '</div></div></div>' +
        '<div class="h-card-bottom"><span class="h-card-meta">' + dateStr + '</span>' +
        '<button class="h-card-delete" data-id="' + item.id + '">Delete</button></div>';
      card.addEventListener("click", function (e) { if (!e.target.classList.contains("h-card-delete")) loadHistoryEntry(item.id); });
      var del = card.querySelector(".h-card-delete");
      if (del) del.addEventListener("click", function (e) { e.stopPropagation(); deleteHistoryEntry(this.getAttribute("data-id")); });
      body.appendChild(card);
    });
  }
  function updateHistoryCount() {
    var b = document.getElementById("history-count");
    if (b) { b.textContent = historyData.length; b.style.display = historyData.length > 0 ? "inline" : "none"; }
  }
  function loadHistoryEntry(id) {
    fetch("/api/history/" + id).then(function (r) { return r.json(); }).then(function (e) {
      if (e && e.result) { renderResults(e.result); closeSidebar(); }
    }).catch(function () {});
  }
  function deleteHistoryEntry(id) { fetch("/api/history/" + id, { method: "DELETE" }).then(function () { loadHistory(); }).catch(function () {}); }
  function clearAllHistory() {
    if (!confirm("Clear all verification history?")) return;
    fetch("/api/history", { method: "DELETE" }).then(function () { loadHistory(); }).catch(function () {});
  }

  // ── Utils ─────────────────────────────────────────────────────────
  function showToast(msg, type) {
    var c = document.getElementById("toast-container"); if (!c) return;
    var t = document.createElement("div"); t.className = "toast toast-" + (type || "info"); t.textContent = msg;
    c.appendChild(t);
    setTimeout(function () { t.classList.add("show"); }, 10);
    setTimeout(function () { t.classList.remove("show"); setTimeout(function () { t.remove(); }, 300); }, 4000);
  }
  function esc(s) { if (!s) return ""; var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (a) {
      a.addEventListener("click", function (e) {
        var t = document.querySelector(this.getAttribute("href"));
        if (t) { e.preventDefault(); t.scrollIntoView({ behavior: "smooth", block: "start" }); }
      });
    });
  }

  // ── Init ──────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", function () {
    initHeroBackground();
    initFaceScan();
    initNavbar();
    initReveal();
    initInputMasks();
    initUploadZones();
    initForm();
    initHistory();
    initSmoothScroll();
  });
})();
