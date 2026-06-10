document.addEventListener("DOMContentLoaded", () => {
  const state = {
    runs: [],
    selectedRunId: null,
    selectedRun: null,
    page: 0,
    limit: 8,
    topK: 10,
    search: "",
    selectedCvId: null,
    selectedCvDetail: null,
    selectedJdId: null,
    selectedJdDetail: null,
    cards: null,
  };

  const nodes = {
    apiStatus: document.getElementById("apiStatus"),
    apiUrl: document.getElementById("apiUrl"),
    runButton: document.getElementById("runButton"),
    runCount: document.getElementById("runCount"),
    runList: document.getElementById("runList"),
    runTitle: document.getElementById("runTitle"),
    resultMeta: document.getElementById("resultMeta"),
    searchInput: document.getElementById("searchInput"),
    topKInput: document.getElementById("topK"),
    summary: document.getElementById("summary"),
    prevPageButton: document.getElementById("prevPageButton"),
    nextPageButton: document.getElementById("nextPageButton"),
    pageLabel: document.getElementById("pageLabel"),
    metricAlgorithm: document.getElementById("metricAlgorithm"),
    metricModel: document.getElementById("metricModel"),
    metricCoverage: document.getElementById("metricCoverage"),
    metricRuntime: document.getElementById("metricRuntime"),
    cvList: document.getElementById("cvList"),
    detailPane: document.getElementById("detailPane"),
  };

  let searchTimer = null;
  nodes.apiUrl.value = window.location.origin;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function baseUrl() {
    return nodes.apiUrl.value.trim().replace(/\/+$/, "");
  }

  async function fetchJson(path) {
    const response = await fetch(`${baseUrl()}${path}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return response.json();
  }

  function setStatus(text, kind = "muted") {
    nodes.apiStatus.textContent = text;
    nodes.apiStatus.dataset.kind = kind;
  }

  function formatRun(run) {
    return run.run_name || `Run ${run.id}`;
  }

  function formatCount(value) {
    return new Intl.NumberFormat("en-US").format(Number(value || 0));
  }

  function formatRuntime(runtimeMs) {
    if (!runtimeMs) {
      return "Not finished";
    }
    if (runtimeMs < 1000) {
      return `${runtimeMs} ms`;
    }
    const seconds = runtimeMs / 1000;
    if (seconds < 60) {
      return `${seconds.toFixed(1)} s`;
    }
    return `${(seconds / 60).toFixed(1)} min`;
  }

  function formatDate(value) {
    if (!value) {
      return "Unknown";
    }
    return new Date(value).toLocaleString();
  }

  function firstMeaningfulValue(payload, keys, fallback = "") {
    if (!payload) {
      return fallback;
    }
    for (const key of keys) {
      const value = payload[key];
      if (value !== null && value !== undefined && String(value).trim()) {
        return String(value).trim();
      }
    }
    return fallback;
  }

  function cvNameFromPayload(payload, fallback) {
    return firstMeaningfulValue(
      payload,
      ["Tên ứng viên", "TÃªn ứng viên", "name"],
      fallback,
    );
  }

  function cvRoleFromPayload(payload) {
    return firstMeaningfulValue(
      payload,
      ["Vị trí ứng tuyển", "Vá»‹ trÃ­ á»©ng tuyá»ƒn", "role"],
      "",
    );
  }

  function jdTitleFromPayload(payload, fallback) {
    return firstMeaningfulValue(
      payload,
      ["Vị trí cần tuyển", "Vá»‹ trÃ­ cáº§n tuyá»ƒn", "title"],
      fallback,
    );
  }

  function jdCompanyFromPayload(payload) {
    return firstMeaningfulValue(
      payload,
      ["Tên công ty", "TÃªn cÃ´ng ty", "company"],
      "",
    );
  }

  function payloadGrid(payload) {
    const entries = Object.entries(payload || {}).filter(([, value]) => value !== null && value !== "");
    if (!entries.length) {
      return '<div class="emptyState">No structured fields.</div>';
    }
    return `
      <div class="detailGrid">
        ${entries.map(([key, value]) => `
          <div class="detailRow">
            <div class="detailKey">${escapeHtml(key)}</div>
            <div class="detailValue">${escapeHtml(value)}</div>
          </div>
        `).join("")}
      </div>
    `;
  }

  function renderRunList() {
    nodes.runCount.textContent = `${state.runs.length} loaded`;
    if (!state.runs.length) {
      nodes.runList.innerHTML = '<div class="emptyState">No runs found.</div>';
      return;
    }

    nodes.runList.innerHTML = state.runs.map((run) => {
      const active = Number(run.id) === Number(state.selectedRunId);
      const finished = run.finished_at ? "Finished" : "Running";
      return `
        <button
          type="button"
          class="runCard${active ? " isActive" : ""}"
          data-run-id="${run.id}"
        >
          <h3 class="runTitle">${escapeHtml(formatRun(run))}</h3>
          <p class="runMeta">${escapeHtml(run.algorithm || "Unknown algorithm")}</p>
          <div class="runFoot">
            <span>${escapeHtml(finished)}</span>
            <span>${escapeHtml(formatRuntime(run.runtime_ms))}</span>
            <span>${escapeHtml(formatDate(run.created_at))}</span>
          </div>
        </button>
      `;
    }).join("");
  }

  function renderRunSummary() {
    const run = state.selectedRun;
    if (!run) {
      nodes.runTitle.textContent = "No run selected";
      nodes.resultMeta.textContent = "No results loaded";
      nodes.summary.textContent = "Waiting for data";
      nodes.metricAlgorithm.textContent = "-";
      nodes.metricModel.textContent = "-";
      nodes.metricCoverage.textContent = "-";
      nodes.metricRuntime.textContent = "-";
      return;
    }

    nodes.runTitle.textContent = formatRun(run);
    nodes.resultMeta.textContent = `${run.counts.cvs} CVs / ${run.counts.jds} JDs / ${run.counts.matches} matches`;
    nodes.summary.textContent = `${run.algorithm || "-"} • ${formatDate(run.created_at)}`;
    nodes.metricAlgorithm.textContent = run.algorithm || "-";
    nodes.metricModel.textContent = run.model_name || "Not specified";
    nodes.metricCoverage.textContent = `${formatCount(run.counts.cvs)} / ${formatCount(run.counts.jds)}`;
    nodes.metricRuntime.textContent = formatRuntime(run.runtime_ms);
  }

  function renderCards() {
    const cards = state.cards;
    if (!cards || !cards.items.length) {
      nodes.cvList.innerHTML = '<div class="emptyState">No candidates matched this selection.</div>';
      return;
    }

    nodes.cvList.innerHTML = cards.items.map((item) => {
      const active = Number(item.cv.id) === Number(state.selectedCvId);
      const topScore = item.matches[0] ? Number(item.matches[0].score).toFixed(4) : "----";
      return `
        <button
          type="button"
          class="cvCard${active ? " isActive" : ""}"
          data-cv-id="${item.cv.id}"
        >
          <div class="cvTop">
            <div>
              <h3 class="cvTitle">${escapeHtml(item.cv.name || item.cv.external_key || `CV ${item.cv.id}`)}</h3>
              <p class="cvMeta">${escapeHtml(item.cv.role || "Role not available")}</p>
            </div>
            <div class="scorePill">${topScore}</div>
          </div>
          <div class="matchList">
            ${item.matches.map((match) => `
              <div class="matchRow">
                <div>
                  <strong>#${match.rank} ${escapeHtml(match.jd.title || `JD ${match.jd.id}`)}</strong>
                  <p class="matchMeta">${escapeHtml(match.jd.company || "Company not available")}</p>
                </div>
                <div class="detailKey">${Number(match.score).toFixed(4)}</div>
              </div>
            `).join("")}
          </div>
          <div class="cvFoot">
            <span>${item.matches.length} visible matches</span>
          </div>
        </button>
      `;
    }).join("");
  }

  function renderDetailPane() {
    if (!state.selectedCvDetail) {
      nodes.detailPane.innerHTML = '<div class="emptyState">Select a candidate to inspect details.</div>';
      return;
    }

    const cv = state.selectedCvDetail.cv;
    const payload = cv.payload || {};
    const cvName = cvNameFromPayload(payload, cv.external_key || `CV ${cv.id}`);
    const cvRole = cvRoleFromPayload(payload) || "Role not available";
    const jdBlock = state.selectedJdDetail ? renderJdDetailBlock() : '<div class="emptyState">Select a job match to inspect the full JD.</div>';

    nodes.detailPane.innerHTML = `
      <section class="detailBlock">
        <div class="detailHeader">
          <h3 class="detailTitle">${escapeHtml(cvName)}</h3>
          <p class="detailMeta">${escapeHtml(cvRole)}</p>
        </div>
        ${payloadGrid(payload)}
      </section>

      <section class="detailBlock">
        <div class="detailHeader">
          <h3 class="detailTitle">Top job matches</h3>
          <p class="detailMeta">${state.selectedCvDetail.matches.length} records returned</p>
        </div>
        <div class="matchList">
          ${state.selectedCvDetail.matches.map((match) => {
            const jdTitle = jdTitleFromPayload(match.jd.payload, match.jd.external_key || `JD ${match.jd.id}`);
            const jdCompany = jdCompanyFromPayload(match.jd.payload) || "Company not available";
            return `
              <div class="matchRow">
                <div>
                  <strong>#${match.rank} ${escapeHtml(jdTitle)}</strong>
                  <p class="matchMeta">${escapeHtml(jdCompany)}</p>
                </div>
                <button type="button" class="matchAction" data-jd-id="${match.jd.id}">
                  ${Number(match.score).toFixed(4)}
                </button>
              </div>
            `;
          }).join("")}
        </div>
      </section>

      <section class="detailBlock">
        <div class="detailHeader">
          <h3 class="detailTitle">Full text</h3>
        </div>
        <pre class="detailText">${escapeHtml(cv.text_content || "No text content available.")}</pre>
      </section>

      <section class="detailBlock">
        <div class="detailHeader">
          <h3 class="detailTitle">Job description</h3>
        </div>
        ${jdBlock}
      </section>
    `;
  }

  function renderJdDetailBlock() {
    const jd = state.selectedJdDetail;
    const payload = jd.payload || {};
    const title = jdTitleFromPayload(payload, jd.external_key || `JD ${jd.id}`);
    const company = jdCompanyFromPayload(payload) || "Company not available";

    return `
      <div class="detailHeader">
        <h3 class="detailTitle">${escapeHtml(title)}</h3>
        <p class="detailMeta">${escapeHtml(company)}</p>
      </div>
      ${payloadGrid(payload)}
      <pre class="detailText">${escapeHtml(jd.text_content || "No text content available.")}</pre>
    `;
  }

  function renderPagination() {
    const total = state.cards ? state.cards.total : 0;
    const pageCount = Math.max(1, Math.ceil(total / state.limit));
    nodes.pageLabel.textContent = `Page ${state.page + 1} of ${pageCount}`;
    nodes.prevPageButton.disabled = state.page <= 0;
    nodes.nextPageButton.disabled = state.page + 1 >= pageCount;
  }

  async function loadRuns() {
    state.runs = await fetchJson("/api/runs?limit=100");
    if (!state.runs.length) {
      state.selectedRunId = null;
      state.selectedRun = null;
      state.cards = null;
      state.selectedCvDetail = null;
      state.selectedJdDetail = null;
      renderRunList();
      renderRunSummary();
      renderCards();
      renderDetailPane();
      renderPagination();
      return;
    }

    if (!state.selectedRunId || !state.runs.some((run) => Number(run.id) === Number(state.selectedRunId))) {
      state.selectedRunId = state.runs[0].id;
    }
    renderRunList();
  }

  async function loadSelectedRun() {
    if (!state.selectedRunId) {
      return;
    }

    const params = new URLSearchParams({
      limit: String(state.limit),
      offset: String(state.page * state.limit),
      top_k: String(state.topK),
    });
    if (state.search) {
      params.set("search", state.search);
    }

    const [run, cards] = await Promise.all([
      fetchJson(`/api/runs/${state.selectedRunId}`),
      fetchJson(`/api/runs/${state.selectedRunId}/cv_cards?${params.toString()}`),
    ]);

    state.selectedRun = run;
    state.cards = cards;

    const currentVisible = cards.items.some((item) => Number(item.cv.id) === Number(state.selectedCvId));
    state.selectedCvId = currentVisible
      ? state.selectedCvId
      : cards.items[0]?.cv.id ?? null;

    renderRunSummary();
    renderCards();
    renderPagination();

    if (state.selectedCvId) {
      await loadCvDetail(state.selectedCvId);
    } else {
      state.selectedCvDetail = null;
      state.selectedJdDetail = null;
      renderDetailPane();
    }
  }

  async function loadCvDetail(cvId) {
    if (!state.selectedRunId || !cvId) {
      return;
    }
    state.selectedCvId = cvId;
    renderCards();

    const detail = await fetchJson(`/api/runs/${state.selectedRunId}/cvs/${cvId}?top_k=${state.topK}`);
    state.selectedCvDetail = detail;

    const nextJdId = detail.matches.some((match) => Number(match.jd.id) === Number(state.selectedJdId))
      ? state.selectedJdId
      : detail.matches[0]?.jd.id ?? null;

    state.selectedJdId = nextJdId;
    state.selectedJdDetail = nextJdId
      ? await fetchJson(`/api/runs/${state.selectedRunId}/jds/${nextJdId}`)
      : null;

    renderDetailPane();
  }

  async function loadJdDetail(jdId) {
    if (!state.selectedRunId || !jdId) {
      return;
    }
    state.selectedJdId = jdId;
    state.selectedJdDetail = await fetchJson(`/api/runs/${state.selectedRunId}/jds/${jdId}`);
    renderDetailPane();
  }

  async function refresh() {
    nodes.runButton.disabled = true;
    setStatus("Checking API...", "muted");
    try {
      await fetchJson("/api/health");
      setStatus("API connected", "ok");
      await loadRuns();
      await loadSelectedRun();
    } catch (error) {
      setStatus("API error", "error");
      nodes.summary.textContent = error.message;
      nodes.cvList.innerHTML = '<div class="emptyState">Unable to load results.</div>';
      nodes.detailPane.innerHTML = '<div class="emptyState">Inspector unavailable.</div>';
    } finally {
      nodes.runButton.disabled = false;
    }
  }

  nodes.runButton.addEventListener("click", refresh);

  nodes.runList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-run-id]");
    if (!button) {
      return;
    }
    state.selectedRunId = Number(button.dataset.runId);
    state.page = 0;
    state.selectedCvId = null;
    state.selectedCvDetail = null;
    state.selectedJdId = null;
    state.selectedJdDetail = null;
    renderRunList();
    loadSelectedRun().catch((error) => {
      nodes.summary.textContent = error.message;
    });
  });

  nodes.cvList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-cv-id]");
    if (!button) {
      return;
    }
    loadCvDetail(Number(button.dataset.cvId)).catch((error) => {
      nodes.summary.textContent = error.message;
    });
  });

  nodes.detailPane.addEventListener("click", (event) => {
    const button = event.target.closest("[data-jd-id]");
    if (!button) {
      return;
    }
    loadJdDetail(Number(button.dataset.jdId)).catch((error) => {
      nodes.summary.textContent = error.message;
    });
  });

  nodes.prevPageButton.addEventListener("click", () => {
    if (state.page <= 0) {
      return;
    }
    state.page -= 1;
    loadSelectedRun().catch((error) => {
      nodes.summary.textContent = error.message;
    });
  });

  nodes.nextPageButton.addEventListener("click", () => {
    const total = state.cards ? state.cards.total : 0;
    if ((state.page + 1) * state.limit >= total) {
      return;
    }
    state.page += 1;
    loadSelectedRun().catch((error) => {
      nodes.summary.textContent = error.message;
    });
  });

  nodes.topKInput.addEventListener("change", () => {
    state.topK = Math.max(1, Math.min(Number(nodes.topKInput.value || 10), 100));
    nodes.topKInput.value = String(state.topK);
    state.selectedJdId = null;
    loadSelectedRun().catch((error) => {
      nodes.summary.textContent = error.message;
    });
  });

  nodes.searchInput.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => {
      state.search = nodes.searchInput.value.trim();
      state.page = 0;
      state.selectedCvId = null;
      loadSelectedRun().catch((error) => {
        nodes.summary.textContent = error.message;
      });
    }, 250);
  });

  refresh();
});
