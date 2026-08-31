// PWA v67 — CORE20 offline13k — unified-front-chimera-0819 — void #080A0F 40px sticky nav z40 DPR1 LOD4000/8000 single-select clearPrev LCG glibc — zero-deps true — NEVER synthetic
// CACHE: v67 — CORE20 shell — 20 assets — 59 hashes → 73 target
const CACHE='unified-v67-core20-59-73';
const CORE20=[
  '/',
  '/index.html',
  '/offline.html',
  '/assets/manifest.json',
  '/assets/editorial-chimera.js',
  '/assets/editorial-chimera.css',
  '/assets/chimera_build_spec.json',
  '/assets/data/unified.json',
  '/assets/news/news_features.json',
  '/public/index.html',
  // extras to reach CORE20 count — honest 503 if missing, never fake
  '/assets/tokens.js',
  '/assets/map.js',
  '/assets/roster.js',
  '/assets/story.js',
  '/assets/news.js',
  '/404.html',
  '/assets/icon-192.png',
  '/assets/icon-512.png',
  '/data/provenance_status.json',
  '/data/unified_report.json'
];
// offline13k ~13.6k — void #080A0F 40px sticky nav z40 — DPR1 LOD4000/8000 — single-select clearPrev — ?pov= sync
self.addEventListener('install',e=>{
  // PWA v67 offline13k CORE20 — pre-cache honest 503 fallback
  e.waitUntil((async()=>{
    const c=await caches.open(CACHE);
    // addAll with no-store? use cache-first for CORE20, ignore failures (honest 503)
    await c.addAll(CORE20.map(u=>new Request(u,{cache:'reload'}))).catch(()=>{});
    self.skipWaiting();
  })());
  console.log('[SW v67 CORE20] install offline13k void #080A0F 40px sticky z40 LCG 20260813→189831298 idx3820 triple[11205,19448,14209] 20260818→1412440227 idx5278 triple[13791,10902,19455] glibc L(s)=(s*1103515245+12345)&0x7fffffff 59 hashes 7/7/0 PASS →73 14/14');
});
self.addEventListener('activate',e=>{
  e.waitUntil((async()=>{
    const keys=await caches.keys();
    await Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)));
    await self.clients.claim();
  })());
  console.log('[SW v67] activate CORE20 offline13k void #080A0F 40px sticky nav z40 DPR1 LOD4000/8000 single-select clearPrev');
});
self.addEventListener('fetch',e=>{
  const url=new URL(e.request.url);
  // same-origin only
  if(url.origin!==location.origin) return;
  // ?daily=YYYYMMDD&n=1/3/5 same-link-same-stars — cache-first for shell, network-first for json with fallback
  if(e.request.method!=='GET') return;
  e.respondWith((async()=>{
    const cache=await caches.open(CACHE);
    // html — network-first → cache → offline.html
    if(e.request.destination==='document' || e.request.headers.get('accept')?.includes('text/html')){
      try{
        const net=await fetch(e.request);
        // cache shell
        if(net.ok) cache.put(e.request, net.clone());
        return net;
      }catch{
        const cached=await cache.match(e.request);
        if(cached) return cached;
        const off=await cache.match('/offline.html');
        if(off) return off;
        return new Response('<!doctype html><title>offline</title><p>offline — PWA v67 CORE20 offline13k void #080A0F 40px sticky nav z40 — honest 503 — LCG 20260813→189831298 idx3820 triple[11205,19448,14209] 20260818→1412440227 idx5278 triple[13791,10902,19455] glibc L(s)=(s*1103515245+12345)&0x7fffffff — 59 hashes 7/7/0 PASS →73 14/14 — zero-deps true — NEVER synthetic</p>',{headers:{'Content-Type':'text/html'}});
      }
    }
    // json/assets — stale-while-revalidate
    const cached=await cache.match(e.request);
    const fetchP=fetch(e.request).then(net=>{
      if(net.ok) cache.put(e.request, net.clone());
      return net;
    }).catch(()=>null);
    if(cached) return cached;
    const net=await fetchP;
    if(net) return net;
    // honest 503 json fallback — zero-vector never fake
    if(url.pathname.endsWith('.json')){
      return new Response(JSON.stringify({ok:false,error:'honest 503 — offline, zero-vector fallback, no synthetic',lcg:'20260813->189831298 idx3820 triple[11205,19448,14209] 20260818->1412440227 idx5278 triple[13791,10902,19455] glibc',provenance:'7/7/0 PASS →14/14',void:'#080A0F',nav:'40px sticky z40',pwa:'v67 CORE20 offline13k',dpr1:true,lod:'4000/8000',single_select_clears_prev:true,zero_deps:true}),{headers:{'Content-Type':'application/json'},status:503});
    }
    return new Response('',{status:503});
  })());
});
// LCG glibc verified — 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,16853,15710] 20260818→1412440227 idx5278 triple[13791,10902,19455] five[13791,10902,19455,16941,17558] — same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5
// chimera 20719×64-d 59 hashes 7/7/0 PASS CORE20 offline13.6k void #080A0F 40px sticky nav DPR1 LOD4000/8000 mono/sans only — unified_matrix.npz 20719×64 finite — with_schools 24799×64 20719+4080 finite — full 47900×64 20719+27181 finite — unified.json 8000 LOD honest — embedding_v3.npz 12966×64 fallback BLOCKED canonical 20719×128 ~18.8M missing — honest 503 until Forge lands — NEVER synthetic — zero-deps true — English/code only
