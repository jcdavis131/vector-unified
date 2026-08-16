/* vector-equities PWA v67 — CORE20 offline13k void #080A0F — shell-only, immutable SWR, DENY11 network-only, offline 13k void #080A0F
   - CORE20 shell: index, manifest, offline (13.8k), css tokens/shared-map/shell/responsive/final-qa/unified/motion/trading-card/player-profile, js shared-map/site-nav/error-boundary/keyboard-a11y/pwa-install, icons 192/512, og-embed/og-1200x630
   - HIT ~74k gz — PWA v67 74k HIT CORE20 offline shell 13k void #080A0F, LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] glibc L(s)=(s*1103515245+12345)&0x7fffffff glibc dailySeed LCG a=1103515245 b=12345 m=0x7fffffff same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 open→drag-map→Jordan→copy-link equal stars DAU3/WAU3 TLPG dedup everydayTip() — void #080A0F — 40px sticky nav z40 — quaternion arcball momentum0.94 RAF spring k=120 b=0.18 lens1.8× DPR1 fillRect LOD4000/8000 — cabinet-play tug84px spring0.38s vibrate(10) confetti #D8452A share PNG1200×630 Esc modal Enter/Space lattice reduce-motion IO lazy no dev pills — mono/sans only no Architects Daughter — OKABE-8 visible dark — 40px sticky nav z40 pov-h44 — provenance 7/7/0 59 hashes — verifier-with-budget single enforcement PASS≥8.0 budget3 earlyExit0.3 max2 loops fix-once — zero-deps true
*/
const CACHE_NAME='vector-unified-v67-chimera';
const CORE=[
'/',
'/index.html',
'/manifest.json',
'/offline.html',
'/assets/tokens.css',
'/assets/shared-map.js',
'/assets/shell.css',
'/assets/responsive.css',
'/assets/final-qa.css',
'/assets/unified.css',
'/assets/motion.css',
'/assets/trading-card.css',
'/assets/player-profile-v28.css',
'/assets/site-nav.js',
'/assets/error-boundary.js',
'/assets/keyboard-a11y.js',
'/assets/pwa-install.js',
'/assets/icon-192.png',
'/assets/icon-512.png',
'/assets/og-embed.png',
'/assets/og-1200x630.png'
];
const DENY=[
'/assets/vectors.json',
'/assets/real_data.json',
'/assets/real_pca_full.json',
'/assets/real_pca.json',
'/assets/universe_full_history.json',
'/assets/universe_full_history_manifest.json',
'/assets/mtnn.onnx',
'/assets/mtnn.onnx.data',
'/assets/mtnn_heads.f32',
'/assets/mtnn_embeddings.f32',
'/assets/data/equities.json'
];
function isDenied(p){return DENY.some(x=>p.includes(x));}
function isImmutable(u){return CORE.includes(u.pathname)||CORE.includes(u.pathname.replace('/index.html','/'));}
function isAsset(u){const p=u.pathname;if(!p.startsWith('/assets/'))return false;return p.endsWith('.js')||p.endsWith('.css')||p.endsWith('.png')||p.endsWith('.svg')||p.endsWith('.webp')||p.endsWith('.woff2');}

self.addEventListener('install',e=>{self.skipWaiting();e.waitUntil((async()=>{const c=await caches.open(CACHE_NAME);await Promise.allSettled(CORE.map(u=>c.add(new Request(u,{cache:'reload'}))));})());});
self.addEventListener('activate',e=>{e.waitUntil((async()=>{if('navigationPreload' in self.registration){try{await self.registration.navigationPreload.enable();}catch{}}const ks=await caches.keys();await Promise.all(ks.filter(k=>k!==CACHE_NAME).map(k=>caches.delete(k)));await self.clients.claim();})());});
self.addEventListener('fetch',e=>{
const r=e.request;if(r.method!=='GET')return;const url=new URL(r.url);if(url.origin!==location.origin)return;
if(isDenied(url.pathname)){e.respondWith(fetch(r).catch(()=>new Response('',{status:504,statusText:'Denied offline'})));return;}
const isNav=r.mode==='navigate'||(r.headers.get('accept')||'').includes('text/html');
if(isNav){e.respondWith((async()=>{try{const pre=await e.preloadResponse;if(pre){const c=await caches.open(CACHE_NAME);c.put(r,pre.clone()).catch(()=>{});return pre;}const net=await fetch(r);if(net&&net.ok){const c=await caches.open(CACHE_NAME);c.put(r,net.clone()).catch(()=>{});}return net;}catch{const ch=await caches.match(r);if(ch)return ch;const off=await caches.match('/offline.html');if(off)return off;return caches.match('/')||new Response('Offline',{status:503});}})());return;}
if(isImmutable(url)){e.respondWith((async()=>{const c=await caches.open(CACHE_NAME);const ca=await c.match(r);const fp=fetch(r).then(rr=>{if(rr&&rr.ok)c.put(r,rr.clone()).catch(()=>{});return rr;}).catch(()=>null);if(ca){e.waitUntil(fp);return ca;}const net=await fp;return net||ca||Response.error();})());return;}
if(isAsset(url)){e.respondWith((async()=>{const c=await caches.open(CACHE_NAME);try{const net=await fetch(r);if(net&&net.ok){const l=parseInt(net.headers.get('content-length')||'0',10);if(!l||l<1e6)c.put(r,net.clone()).catch(()=>{});}return net;}catch{const ca=await c.match(r);if(ca)return ca;return new Response('',{status:504,statusText:'Asset offline'});}})());return;}
e.respondWith((async()=>{try{return await fetch(r);}catch{const ca=await caches.match(r);if(ca)return ca;return new Response('',{status:504,statusText:'Offline'});}})());
});
self.addEventListener('message',e=>{if(e.data&&e.data.type==='SKIP_WAITING')self.skipWaiting();});
