/* VECTOR FLEET — private dashboard renderer. Vanilla JS, no external libraries. */

(function () {
  "use strict";

  var KEYSTONE_REPO = "vector-unified";

  var STATUS_LABELS = {
    production: "production",
    shipped: "shipped",
    wip: "wip",
    blocked: "blocked",
  };

  var STATUS_ORDER = { production: 0, shipped: 1, wip: 2, blocked: 3 };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  // Format a metric value: percentages/ratios kept readable, ints as-is.
  function fmt(v) {
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v)) return String(v);
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
    return (Math.round(v * 10000) / 10000).toString();
  }

  // Compute the lift string for a headline metric.
  // value/baseline when baseline > 0, else "vs floor".
  function liftText(hm) {
    if (!hm || typeof hm.value !== "number") return "";
    var b = hm.baseline;
    if (typeof b === "number" && b > 0) {
      var ratio = hm.value / b;
      if (ratio >= 1) return ratio.toFixed(ratio >= 10 ? 0 : 1) + "x";
      // Sub-1 ratio (e.g. an MAE where lower is better): show as a fraction.
      return ratio.toFixed(2) + "x";
    }
    return "vs floor";
  }

  function statusBadge(status) {
    var s = STATUS_LABELS[status] ? status : "blocked";
    var badge = el("span", "status-badge status-" + s, STATUS_LABELS[s] || status);
    return badge;
  }

  function renderLegend(container) {
    container.innerHTML = "";
    var items = [
      ["production", "var(--st-production)"],
      ["shipped", "var(--st-shipped)"],
      ["wip", "var(--st-wip)"],
      ["blocked", "var(--st-blocked)"],
    ];
    items.forEach(function (pair) {
      var item = el("span", "legend-item");
      var dot = el("span", "legend-dot");
      dot.style.background = pair[1];
      item.appendChild(dot);
      item.appendChild(document.createTextNode(pair[0]));
      container.appendChild(item);
    });
  }

  function renderCard(m) {
    var isKeystone = m.repo === KEYSTONE_REPO;
    var isPrivate = m.visibility === "private";
    var card = el("div", "card" + (isKeystone ? " keystone" : ""));

    if (isKeystone) {
      card.appendChild(el("span", "keystone-tag", "keystone"));
    }

    // Head: repo name (+ private tag) and status badge.
    var head = el("div", "card-head");
    var left = el("div");
    var repoLine = el("div", "repo-line");
    repoLine.appendChild(el("span", "repo-name", m.repo));
    if (isPrivate) repoLine.appendChild(el("span", "tag-private", "private"));
    left.appendChild(repoLine);
    left.appendChild(el("p", "domain", m.domain || ""));
    head.appendChild(left);
    head.appendChild(statusBadge(m.status));
    card.appendChild(head);

    // Meta row: embedding dim, arch tag, live link.
    var meta = el("div", "meta-row");
    if (typeof m.embeddingDim === "number") {
      var dim = el("span");
      dim.appendChild(el("span", "k", "dim "));
      dim.appendChild(document.createTextNode(String(m.embeddingDim) + "d"));
      meta.appendChild(dim);
    }
    if (m.archTag) {
      meta.appendChild(el("span", "arch-tag", m.archTag));
    }
    if (m.liveUrl) {
      var a = el("a", "live-link", "live ↗");
      a.href = m.liveUrl;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      meta.appendChild(a);
    }
    card.appendChild(meta);

    // Headline metric.
    var hm = m.headlineMetric || {};
    var headline = el("div", "headline");
    headline.appendChild(el("p", "headline-name", hm.name || "headline metric"));
    var vrow = el("div", "headline-value-row");
    vrow.appendChild(el("span", "headline-value", fmt(hm.value)));
    if (typeof hm.baseline === "number") {
      vrow.appendChild(
        el("span", "headline-baseline", "baseline " + fmt(hm.baseline))
      );
    }
    var lt = liftText(hm);
    if (lt) vrow.appendChild(el("span", "lift", lt));
    headline.appendChild(vrow);
    card.appendChild(headline);

    // Compact metrics list.
    if (m.metrics && Object.keys(m.metrics).length) {
      var metrics = el("div", "metrics");
      Object.keys(m.metrics).forEach(function (key) {
        var mk = el("span", "mk", key);
        mk.title = key;
        metrics.appendChild(mk);
        metrics.appendChild(el("span", "mv", fmt(m.metrics[key])));
      });
      card.appendChild(metrics);
    }

    // Strengths / gaps.
    var sg = el("div", "sg");
    if (m.strengths) {
      var st = el("div", "strength");
      st.appendChild(el("span", "label", "Strengths"));
      st.appendChild(el("p", null, m.strengths));
      sg.appendChild(st);
    }
    if (m.gaps) {
      var gp = el("div", "gap");
      gp.appendChild(el("span", "label", "Gaps"));
      gp.appendChild(el("p", null, m.gaps));
      sg.appendChild(gp);
    }
    card.appendChild(sg);

    return card;
  }

  function render(data) {
    document.getElementById("thesis").textContent = data.thesis || "";
    var footNote = document.getElementById("foot-note");
    if (footNote) footNote.textContent = data.note || "";

    renderLegend(document.getElementById("legend"));

    var grid = document.getElementById("grid");
    grid.innerHTML = "";

    var models = (data.models || []).slice();
    // Keystone first, then by maturity, then by name.
    models.sort(function (a, b) {
      if (a.repo === KEYSTONE_REPO) return -1;
      if (b.repo === KEYSTONE_REPO) return 1;
      var sa = STATUS_ORDER[a.status] != null ? STATUS_ORDER[a.status] : 9;
      var sb = STATUS_ORDER[b.status] != null ? STATUS_ORDER[b.status] : 9;
      if (sa !== sb) return sa - sb;
      return String(a.repo).localeCompare(String(b.repo));
    });

    models.forEach(function (m) {
      grid.appendChild(renderCard(m));
    });
  }

  function showError(msg) {
    var grid = document.getElementById("grid");
    grid.innerHTML = "";
    grid.appendChild(el("div", "error", msg));
  }

  fetch("./data/fleet.json")
    .then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    })
    .then(render)
    .catch(function (err) {
      showError(
        "Could not load fleet data (" +
          err.message +
          "). If you opened index.html directly, run a static server instead: " +
          "python3 -m http.server (browsers block fetch() over file://)."
      );
    });
})();
