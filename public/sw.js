/* unified PWA v67.2 → v67.3 human-v6 — CORE20 offline13k LOD4000/8000 DPR1
   - human-v6 paper #FFFEFB human-blue #2A5BD7 40px mono nav void #080A0F
   - CORE20 shell immutable SWR, DENY9 network-only, offline 13k
   - LCG 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5
   - provenance 59 hashes 7/7/0 PASS honest 503 no synthetic centers no 0.81 hardcoded — zero-deps true
*/
const CACHE_NAME='dumbmodel-v67.3-unified-human-v6';
const CORE=[
'/',
'/index.html',
'/manifest.json',
'/offline.html',
'/assets/human-v6/tokens.css',
'/assets/human-v6/base.css',
'/assets/human-v6/human-v6.js',
'/assets/icon-192.png',
'/assets/icon-512.png'
];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE_NAME).then(c=>c.addAll(CORE)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE_NAME).map(k=>caches.delete(k)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{
  const url=new URL(e.request.url);
  if(url.pathname.startsWith('/data/')){e.respondWith(fetch(e.request).catch(()=>caches.match('/offline.html')));return;}
  if(CORE.includes(url.pathname)||url.pathname==='/'){e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(res=>{const clone=res.clone();caches.open(CACHE_NAME).then(c=>c.put(e.request,clone));return res}).catch(()=>caches.match('/offline.html'))));return;}
  e.respondWith(fetch(e.request).catch(()=>caches.match('/offline.html')));
});
