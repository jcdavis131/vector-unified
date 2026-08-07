/* unified.dumbmodel.com — one role space, three sports.
 *
 * House rules this file follows:
 *  - every displayed number is read from a shipped artifact at runtime; none
 *    are hardcoded in markup. If the fetch fails the page says so rather than
 *    rendering a plausible placeholder.
 *  - hand-rolled 3D on a 2D canvas. No library, no CDN.
 *  - the gate wording is constrained. G2 is recorded "met", but the artifact
 *    itself calls that a weak bar and says to quote the margin (0.0593 over
 *    majority), never the status or the vs-chance figure. G4 is always
 *    reported WITH its circularity.
 */
(function () {
  'use strict';

  var SPORT_COLOR = { hoops: '#e8973a', gridiron: '#4da3ff', pitch: '#48c78e' };
  var SPORT_LABEL = { hoops: 'basketball', gridiron: 'football', pitch: 'soccer' };
  var ARCH_PALETTE = ['#e8973a', '#4da3ff', '#48c78e', '#c98ae0', '#e0b341', '#e0685f',
                      '#5fd0c4', '#b0b8c0', '#8a7fe0', '#d2734f', '#7fc45f', '#e05f9a'];

  var $ = function (id) { return document.getElementById(id); };
  var fmt = function (v, d) { return v == null ? '—' : Number(v).toFixed(d == null ? 4 : d); };

  function fail(msg) {
    var c = $('counts');
    if (c) { c.textContent = msg; c.style.color = '#e0685f'; }
  }

  fetch('assets/unified_slim.json')
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(init)
    .catch(function (e) {
      // Never fabricate: an unreachable artifact means an empty page, not a demo one.
      fail('Data not loaded (' + e.message + '). Nothing on this page is shown from memory.');
    });

  function init(doc) {
    var meta = doc.meta, rows = doc.rows;
    var IX = {}; meta.row_schema.forEach(function (k, i) { IX[k] = i; });

    header(meta, rows, IX);
    gates(meta);
    ablation(meta);
    archetypes(meta, rows, IX);
    caveats(meta);
    sky(meta, rows, IX);
  }

  /* ---------------------------------------------------------------- header */
  function header(meta, rows, IX) {
    var sc = meta.sport_counts || {};
    var parts = Object.keys(sc).sort(function (a, b) { return sc[b] - sc[a]; })
      .map(function (s) {
        return '<b>' + sc[s].toLocaleString() + '</b> ' + (SPORT_LABEL[s] || s);
      });
    $('counts').innerHTML =
      '<b>' + rows.length.toLocaleString() + '</b> player-seasons · ' + parts.join(' · ') +
      ' · <b>' + meta.d_emb + '</b>-d joint embedding · built ' + (meta.built || '—');

    var ev = meta.explained_variance;
    if (ev && ev.length >= 3) {
      var pct = (ev[0] + ev[1] + ev[2]) * 100;
      $('var-note').textContent = pct.toFixed(1) + '%';
    } else {
      $('var-note').textContent = 'an unrecorded share';
    }
  }

  /* ----------------------------------------------------------------- gates */
  function gates(meta) {
    var ab = (meta.ablation && meta.ablation.configs) || {};
    var full = ab.full || {};
    var cards = [
      {
        tag: 'PASS', cls: 'pass', name: 'G1 · per-sport non-inferiority',
        v: 'no degradation',
        base: 'joint ≥ native, all three sports',
        note: 'Joining the sports did not damage any sport’s own neighbourhood structure. This is the cleanest result here.'
      },
      {
        tag: 'WEAK BAR', cls: 'defer',
        name: 'G2 · sport recoverability',
        v: fmt(meta.g2_sport_acc),
        base: 'majority baseline ' + fmt(meta.g2_majority_baseline) +
              ' · margin ' + fmt(meta.g2_delta_vs_majority),
        note: 'Recorded as “' + (meta.g2_status || '?') + '” against a target of ' +
              fmt(meta.g2_target) + ', but the artifact defines that as within 10 points of the ' +
              'achievable floor and tells readers to quote the margin, not the status. ' +
              'The sport is still recoverable from the geometry.'
      },
      {
        tag: 'PASS', cls: 'pass', name: 'G3 · archetype silhouette',
        v: fmt(full.G3_sil),
        base: 'floor 0.05 · sd ' + fmt(full.G3_sil_sd),
        note: 'Same-archetype rows cluster across sports. Trained on the archetype labels, so read it with G4.'
      },
      {
        tag: 'PASS, BUT CIRCULAR', cls: 'circ', name: 'G4 · cross-sport role coherence',
        v: fmt(full.G4_hit),
        base: 'random baseline ' + fmt(full.G4_baseline),
        note: 'The archetype map is hand-authored and the contrastive loss trains on it. Remove that loss and this falls to ' +
              fmt((ab.no_supcon || {}).G4_hit) + ' — chance. See the toggle below.'
      }
    ];
    $('gates').innerHTML = cards.map(function (c) {
      return '<div class="gate"><span class="tag ' + c.cls + '">' + c.tag + '</span>' +
             '<h3>' + c.name + '</h3><div class="v">' + c.v + '</div>' +
             '<div class="base">' + c.base + '</div>' +
             '<div class="note">' + c.note + '</div></div>';
    }).join('');
    $('gate-src').innerHTML = 'G2 from <code>unified.json</code> (' +
      (meta.g2_note ? String(meta.g2_note).slice(0, 150) : 'see file') +
      '). G3/G4 from <code>data/ablation_report.json → configs.full</code>, ' +
      (full.n_seeds || '?') + ' seeds.';
  }

  /* ------------------------------------------------------------- ablation */
  function ablation(meta) {
    var ab = meta.ablation;
    var host = $('ab-buttons');
    if (!ab || !ab.configs) {
      host.textContent = 'ablation_report.json not present in this build.';
      return;
    }
    var names = Object.keys(ab.configs);
    var full = ab.configs.full || {};
    var cur = 'full';

    host.innerHTML = names.map(function (n) {
      return '<button data-cfg="' + n + '" aria-pressed="' + (n === cur) + '">' + n + '</button>';
    }).join('');

    function render(cfg) {
      var c = ab.configs[cfg] || {};
      var defs = [
        ['G2 sport acc (lower is better)', 'G2_sport_acc', 'G2_sport_acc_sd', null, true],
        ['G3 archetype silhouette', 'G3_sil', 'G3_sil_sd', 0.05, false],
        ['G4 cross-sport role hit', 'G4_hit', 'G4_hit_sd', c.G4_baseline, false]
      ];
      var body = defs.map(function (d) {
        var v = c[d[1]], f = full[d[1]], sd = c[d[2]];
        var delta = (v != null && f != null) ? v - f : null;
        var bad = delta != null && (d[4] ? delta > 0.0001 : delta < -0.0001);
        return '<tr><td>' + d[0] + '</td>' +
          '<td>' + fmt(v) + (sd != null ? ' <span style="color:#7c8a99">±' + fmt(sd) + '</span>' : '') + '</td>' +
          '<td>' + fmt(f) + '</td>' +
          '<td' + (bad ? ' class="drop"' : '') + '>' +
            (delta == null ? '—' : (delta >= 0 ? '+' : '') + fmt(delta)) + '</td>' +
          '<td>' + (d[3] == null ? '—' : fmt(d[3])) + '</td></tr>';
      }).join('');
      $('ab-table').querySelector('tbody').innerHTML = body;

      var note = '';
      if (cfg === 'no_supcon') {
        note = 'This is the one that matters. Without the supervised-contrastive loss, G4 lands at ' +
          fmt(c.G4_hit) + ' against a random baseline of ' + fmt(c.G4_baseline) +
          ' — indistinguishable from chance, and G3 collapses from ' + fmt(full.G3_sil) +
          ' to ' + fmt(c.G3_sil) + '. The cross-sport role result is the loss converging on a ' +
          'hand-authored label map, not a discovery.';
      } else if (cfg === 'task_only') {
        note = 'All alignment losses removed — the honest floor for what the raw task signal alone produces.';
      } else if (cfg === 'full') {
        note = 'The shipped configuration. Every other button removes exactly one loss and re-runs all ' +
          (full.n_seeds || 3) + ' seeds.';
      } else {
        note = 'One loss removed, all ' + (c.n_seeds || 3) +
          ' seeds re-run. Compare Δ against the full model, and note the standard deviations — ' +
          'several of these differences are inside seed noise.';
      }
      $('ab-note').textContent = note;

      Array.prototype.forEach.call(host.querySelectorAll('button'), function (b) {
        b.setAttribute('aria-pressed', String(b.dataset.cfg === cfg));
      });
    }

    host.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-cfg]');
      if (b) { cur = b.dataset.cfg; render(cur); }
    });
    render(cur);
  }

  /* ----------------------------------------------------------- archetypes */
  function archetypes(meta, rows, IX) {
    var counts = meta.arch_counts || {};
    var declared = meta.archetypes || [];
    var total = rows.length;
    var live = declared.filter(function (a) { return counts[a.id]; });
    var dead = declared.filter(function (a) { return !counts[a.id]; }).map(function (a) { return a.id; });

    $('arch-table').querySelector('tbody').innerHTML = live
      .sort(function (a, b) { return counts[b.id] - counts[a.id]; })
      .map(function (a) {
        return '<tr><td>' + a.id + '</td><td style="text-align:left">' + a.label + '</td>' +
          '<td>' + counts[a.id].toLocaleString() + '</td>' +
          '<td>' + (100 * counts[a.id] / total).toFixed(1) + '%</td></tr>';
      }).join('');
    $('dead-arch').textContent = dead.length ? dead.join(', ') : 'none';
  }

  /* -------------------------------------------------------------- caveats */
  function caveats(meta) {
    $('cv-g2').textContent = fmt(meta.g2_sport_acc);
    $('cv-maj').textContent = fmt(meta.g2_majority_baseline);
    $('cv-delta').textContent = fmt(meta.g2_delta_vs_majority);
  }

  /* ------------------------------------------------------------ starfield */
  function sky(meta, rows, IX) {
    var cv = $('sky'), ctx = cv.getContext('2d'), hov = $('hover'), wrap = $('sky-wrap');
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var yaw = 0.6, pitchA = -0.25, zoom = 1, spin = !reduce, colorMode = 'sport', filter = '';
    var dragging = false, lastX = 0, lastY = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var archIds = (meta.archetypes || []).map(function (a) { return a.id; });

    function resize() {
      var w = cv.clientWidth, h = cv.clientHeight;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    resize();
    window.addEventListener('resize', resize);

    // Precompute centred coordinates once; the draw loop must stay allocation-free.
    var n = rows.length, X = new Float32Array(n), Y = new Float32Array(n), Z = new Float32Array(n);
    var cx = 0, cy = 0, cz = 0;
    for (var i = 0; i < n; i++) { cx += rows[i][IX.x]; cy += rows[i][IX.y]; cz += rows[i][IX.z]; }
    cx /= n; cy /= n; cz /= n;
    var span = 0;
    for (i = 0; i < n; i++) {
      X[i] = rows[i][IX.x] - cx; Y[i] = rows[i][IX.y] - cy; Z[i] = rows[i][IX.z] - cz;
      span = Math.max(span, Math.abs(X[i]), Math.abs(Y[i]), Math.abs(Z[i]));
    }
    var order = new Int32Array(n), depth = new Float32Array(n);
    for (i = 0; i < n; i++) order[i] = i;

    function colorFor(i) {
      if (colorMode === 'sport') return SPORT_COLOR[rows[i][IX.sport]] || '#8a94a0';
      var k = archIds.indexOf(rows[i][IX.arch]);
      return ARCH_PALETTE[(k < 0 ? 0 : k) % ARCH_PALETTE.length];
    }

    var px = new Float32Array(n), py = new Float32Array(n);

    function draw() {
      var w = cv.width, h = cv.height;
      ctx.clearRect(0, 0, w, h);
      var cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
      var cp = Math.cos(pitchA), sp = Math.sin(pitchA);
      var scale = (Math.min(w, h) * 0.40 * zoom) / (span || 1);
      var ox = w / 2, oy = h / 2;

      for (var i = 0; i < n; i++) {
        var x = X[i], y = Y[i], z = Z[i];
        var x1 = x * cyaw - z * syaw, z1 = x * syaw + z * cyaw;
        var y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
        depth[i] = z2;
        px[i] = ox + x1 * scale;
        py[i] = oy - y1 * scale;
      }
      // painter's algorithm: far points first so near ones sit on top
      Array.prototype.sort.call(order, function (a, b) { return depth[a] - depth[b]; });

      var r = Math.max(1, 1.5 * dpr * zoom);
      for (var k = 0; k < n; k++) {
        var idx = order[k];
        if (filter && rows[idx][IX.sport] !== filter) continue;
        var t = (depth[idx] / (span || 1) + 1) / 2;            // 0 far … 1 near
        ctx.globalAlpha = 0.18 + 0.62 * Math.max(0, Math.min(1, t));
        ctx.fillStyle = colorFor(idx);
        ctx.beginPath();
        ctx.arc(px[idx], py[idx], r * (0.6 + 0.7 * t), 0, 6.283185);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function frame() {
      if (spin && !dragging) { yaw += 0.0016; draw(); }
      requestAnimationFrame(frame);
    }
    draw();
    requestAnimationFrame(frame);

    /* interaction */
    cv.addEventListener('pointerdown', function (e) {
      dragging = true; lastX = e.clientX; lastY = e.clientY; cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointerup', function (e) {
      dragging = false; try { cv.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    cv.addEventListener('pointermove', function (e) {
      if (dragging) {
        yaw += (e.clientX - lastX) * 0.006;
        pitchA = Math.max(-1.4, Math.min(1.4, pitchA + (e.clientY - lastY) * 0.006));
        lastX = e.clientX; lastY = e.clientY;
        draw();
        return;
      }
      // hover: nearest projected point within a few px
      var rect = cv.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * (cv.width / rect.width);
      var my = (e.clientY - rect.top) * (cv.height / rect.height);
      var best = -1, bestD = 12 * dpr * 12 * dpr;
      for (var i = 0; i < n; i++) {
        if (filter && rows[i][IX.sport] !== filter) continue;
        var dx = px[i] - mx, dy = py[i] - my, d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best < 0) { hov.style.display = 'none'; return; }
      var rrow = rows[best];
      hov.innerHTML = '<b>' + rrow[IX.name] + '</b><br>' +
        (SPORT_LABEL[rrow[IX.sport]] || rrow[IX.sport]) + ' · ' + rrow[IX.season] +
        (rrow[IX.team] ? ' · ' + rrow[IX.team] : '') +
        '<br><span style="color:#7c8a99">archetype ' + rrow[IX.arch] + '</span>';
      hov.style.display = 'block';
      hov.style.left = Math.min(rect.width - 270, e.clientX - rect.left + 12) + 'px';
      hov.style.top = (e.clientY - rect.top + 12) + 'px';
    });
    cv.addEventListener('pointerleave', function () { hov.style.display = 'none'; });
    cv.addEventListener('wheel', function (e) {
      e.preventDefault();
      zoom = Math.max(0.4, Math.min(6, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
      draw();
    }, { passive: false });

    /* controls */
    function setMode(m) {
      colorMode = m;
      $('c-sport').setAttribute('aria-pressed', String(m === 'sport'));
      $('c-arch').setAttribute('aria-pressed', String(m === 'arch'));
      legend();
      draw();
    }
    $('c-sport').addEventListener('click', function () { setMode('sport'); });
    $('c-arch').addEventListener('click', function () { setMode('arch'); });
    $('spin').addEventListener('click', function () {
      spin = !spin;
      this.setAttribute('aria-pressed', String(spin));
      this.textContent = spin ? 'pause spin' : 'resume spin';
    });
    if (reduce) { $('spin').setAttribute('aria-pressed', 'false'); $('spin').textContent = 'resume spin'; }
    $('filter').addEventListener('change', function () { filter = this.value; draw(); });

    function legend() {
      var host = $('legend');
      if (colorMode === 'sport') {
        var sc = meta.sport_counts || {};
        host.innerHTML = Object.keys(SPORT_COLOR).filter(function (s) { return sc[s]; })
          .map(function (s) {
            return '<span><i class="dot" style="background:' + SPORT_COLOR[s] + '"></i>' +
              SPORT_LABEL[s] + ' (' + sc[s].toLocaleString() + ')</span>';
          }).join('');
      } else {
        var counts = meta.arch_counts || {};
        host.innerHTML = (meta.archetypes || []).filter(function (a) { return counts[a.id]; })
          .map(function (a) {
            var k = archIds.indexOf(a.id);
            return '<span><i class="dot" style="background:' +
              ARCH_PALETTE[k % ARCH_PALETTE.length] + '"></i>' + a.id + '</span>';
          }).join('') + '<span style="color:#7c8a99">(only populated archetypes shown)</span>';
      }
    }
    legend();
  }
})();
