// Run the page's real load->transform->centroid->join path against the actual JSON files,
// outside a browser, and check every numeric claim the page's prose makes. This is the
// validate gate: prose cannot drift away from the data without this failing.
//
//   node scripts/check_unified_map.mjs
//
// Do NOT pipe it. A pipe has eaten the exit code three times in this project's ledger.
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const readPub = f => JSON.parse(fs.readFileSync(path.join(ROOT, 'public', f), 'utf-8'));

let fails = 0;
const ok = (cond, msg) => {
  console.log((cond ? '  PASS  ' : '  FAIL  ') + msg);
  if (!cond) fails++;
};

const map = readPub('assets/unified_map.json');
const ros = readPub('assets/unified_roster.json');
const html = fs.readFileSync(path.join(ROOT, 'public', 'index.html'), 'utf-8');

console.log('\n-- file contract --');
ok(Array.isArray(map.rows), 'unified_map.json has a rows array');
ok(map.rows.length === map.n_rows, `rows ${map.rows.length} == declared n_rows ${map.n_rows}`);
ok(map.rows.length === 20721, 'row count is the full 20,721 player-seasons');
ok(Array.isArray(map.sports) && map.sports.length === 3, 'exactly 3 sports');
ok(JSON.stringify(map.sports.slice().sort()) === JSON.stringify(['gridiron', 'hoops', 'pitch']),
   'the three sports are hoops, gridiron and pitch');
ok(Array.isArray(ros.tiles) && ros.tiles.length === 12, '12 roster tiles');

console.log('\n-- the export is the current model, not an older one --');
ok(/best_epoch=27/.test(map.model || ''), `model label names best_epoch=27 (${map.model})`);
ok(typeof map.built_utc === 'string' && /UTC$/.test(map.built_utc),
   `built_utc is a real timestamp, not the exporter's hardcoded string (${map.built_utc})`);
ok(map.exporter_built_field === '2026-07-30',
   'the exporter\'s own hardcoded "built" is preserved verbatim, not silently replaced');

console.log('\n-- no fabricated values --');
ok(map.rows.every(r => Number.isFinite(r.x) && Number.isFinite(r.y)),
   'every row has finite coordinates');
const distinctXY = new Set(map.rows.map(r => r.x + ',' + r.y)).size;
ok(distinctXY > map.rows.length * 0.99,
   `each row has its own position, not one drawn from a few centres (${distinctXY} distinct of ${map.rows.length})`);
ok(map.rows.every(r => r.name && r.sport), 'every row carries a name and a sport');
ok(!map.rows.some(r => /^Player \d+/.test(r.name)),
   'no "Player 70 QB" placeholder names survive');
const blob = JSON.stringify(map) + JSON.stringify(ros);
for (const bad of ['HOOPS_GUARD', 'EQUITY_MOM', 'SCHOOL_W', 'CHIMERA_HUB']) {
  ok(!blob.includes(bad), `invented archetype name "${bad}" is absent from the data files`);
}
const modMap = (html.match(/<script id="mod-map">([\s\S]*?)<\/script>/) || [])[1] || '';
ok(modMap.length > 1000, 'the mod-map script was found for inspection');
ok(!/lcg|Math\.random|box-?muller|genPoints/i.test(modMap),
   'the map module contains no generator: no LCG, no Math.random, no synthesised points');
ok(/fetch\("assets\/unified_map\.json"/.test(modMap),
   'the map module fetches the real export');
ok(/loadError/.test(modMap) && !/catch[\s\S]{0,200}gen\(/.test(modMap),
   'the map module has an error path and no generated fallback behind it');

console.log('\n-- the page claims no domain the model lacks --');
for (const bad of ['+ equities + schools', 'chimera vectors', '20,719']) {
  ok(!html.includes(bad), `page no longer says "${bad}"`);
}
ok(html.includes('20,721'), 'page states the real row count 20,721');

console.log('\n-- the view transform the page applies --');
let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
for (const r of map.rows) {
  if (r.x < x0) x0 = r.x; if (r.x > x1) x1 = r.x;
  if (r.y < y0) y0 = r.y; if (r.y > y1) y1 = r.y;
}
const dx = (x1 - x0) || 1, dy = (y1 - y0) || 1;
const points = map.rows.map(r => ({
  x: (r.x - x0) / dx, y: 1 - (r.y - y0) / dy,
  name: r.name, sport: r.sport, season: r.season,
}));
ok(points.every(p => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1),
   'every transformed point lands inside the 0..1 canvas box');

console.log('\n-- sport centroids, as the page computes them --');
const acc = new Map();
for (const p of points) {
  const a = acc.get(p.sport) || { sx: 0, sy: 0, n: 0, xs: [], ys: [] };
  a.sx += p.x; a.sy += p.y; a.n++; a.xs.push(p.x); a.ys.push(p.y);
  acc.set(p.sport, a);
}
const sd = v => {
  const m = v.reduce((s, q) => s + q, 0) / v.length;
  return Math.sqrt(v.reduce((s, q) => s + (q - m) ** 2, 0) / v.length);
};
const centers = [...acc.entries()]
  .map(([sport, a]) => ({ sport, x: a.sx / a.n, y: a.sy / a.n, n: a.n,
                          spread: (sd(a.xs) + sd(a.ys)) / 2 }))
  .sort((p, q) => q.n - p.n);
console.log('  centroid table (what the prose must match):');
for (const c of centers) {
  console.log(`    ${c.sport.padEnd(10)} x ${c.x.toFixed(3)}  y ${c.y.toFixed(3)}` +
              `  spread ${c.spread.toFixed(3)}  n ${String(c.n).padStart(6)}`);
}

console.log('\n-- prose claims checked against those numbers --');
const cx = centers.map(c => c.x);
const centroidSpan = Math.max(...cx) - Math.min(...cx);
ok(Math.abs(centroidSpan - 0.11) < 0.015,
   `step 3: sport centroids sit within ~0.11 across the width (actual ${centroidSpan.toFixed(3)})`);
const meanSpread = centers.reduce((s, c) => s + c.spread, 0) / centers.length;
ok(Math.abs(meanSpread - 0.24) < 0.02,
   `step 3: a sport's own spread averages ~0.24 (actual ${meanSpread.toFixed(3)})`);
ok(centroidSpan < meanSpread,
   'step 3: the sports overlap more than they separate (centroid span < mean spread)');
ok(Math.abs(map.g2_sport_acc - 0.632) < 0.002 && Math.abs(map.g2_majority_baseline - 0.626) < 0.002,
   `step 3: sport classifier 0.632 vs majority 0.626 (actual ${map.g2_sport_acc.toFixed(3)} / ${map.g2_majority_baseline})`);
ok(Math.abs(map.g2_delta_vs_majority - 0.0063) < 0.0005,
   `footer: G2 margin over majority is 0.0063 (actual ${map.g2_delta_vs_majority})`);

const hoops = centers.find(c => c.sport === 'hoops');
const grid = centers.find(c => c.sport === 'gridiron');
const pitch = centers.find(c => c.sport === 'pitch');
ok(hoops.n === 12966 && grid.n === 5325 && pitch.n === 2430,
   'steps 4-6: counts are 12,966 / 5,325 / 2,430');
ok(Math.round((hoops.n / map.rows.length) * 100) === 63,
   `step 4: basketball is 63% of the map (actual ${Math.round((hoops.n / map.rows.length) * 100)}%)`);
ok(hoops.spread === Math.max(...centers.map(c => c.spread)),
   'step 4: basketball has the largest spread of the three');
ok(pitch.spread === Math.min(...centers.map(c => c.spread)),
   'step 6: football has the smallest spread of the three');

const pairs = [
  ['hoops-gridiron', Math.hypot(hoops.x - grid.x, hoops.y - grid.y)],
  ['hoops-pitch', Math.hypot(hoops.x - pitch.x, hoops.y - pitch.y)],
  ['gridiron-pitch', Math.hypot(grid.x - pitch.x, grid.y - pitch.y)],
];
const widest = pairs.slice().sort((a, b) => b[1] - a[1])[0];
ok(widest[0] === 'gridiron-pitch' && Math.abs(widest[1] - 0.111) < 0.01,
   `step 5: the widest centroid pair is gridiron-pitch at 0.111 (actual ${widest[0]} ${widest[1].toFixed(3)})`);
ok(widest[1] < meanSpread / 2,
   'step 5: even the widest pair is less than half the average within-sport spread');

console.log('\n-- roster tiles join to real map points --');
for (const t of ros.tiles) {
  const hit = points.find(p => p.name === t.name && p.season === t.season && p.sport === t.sport);
  ok(!!hit, `${t.name} (${t.sport} ${t.season}) is findable on the map`);
}
ok(ros.tiles.every(t => t.nearest_other_sport && t.nearest_other_sport.sport !== t.sport),
   "every tile's nearest neighbour really is from a different sport");
ok(ros.tiles.every(t => typeof t.nearest_other_sport.cosine === 'number' &&
                        t.nearest_other_sport.cosine > 0 && t.nearest_other_sport.cosine <= 1),
   'every cosine is a real similarity in (0, 1]');
const nbNames = ros.tiles.map(t => t.nearest_other_sport.name);
ok(new Set(nbNames).size >= 10,
   `neighbours are not degenerate: ${new Set(nbNames).size} distinct across 12 tiles`);
for (const s of ['hoops', 'gridiron', 'pitch']) {
  ok(ros.tiles.filter(t => t.sport === s).length === 4, `4 tiles for ${s}`);
  ok(ros.tiles.filter(t => t.sport === s && t.pick === 'most typical').length === 1,
     `exactly one "most typical" pick for ${s}`);
}

console.log(fails === 0 ? '\nALL CHECKS PASSED\n' : `\n${fails} CHECK(S) FAILED\n`);
process.exit(fails === 0 ? 0 : 1);
