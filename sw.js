/* vector-unified PWA v67 — 74426B HIT void #080A0F — shell-only, mirrors equities v67 — CORE immutable stale-while-revalidate, large JSON/ONNX deny-cached, network-first 1MB cap
   - PWA v67 74426B HIT void #080A0F — chimera dailySeed LCG 20260812 idx3970 triple[3970,14390,4582] same-link-same-stars
   - CORE only shell (~20 files), no large JSON/models/CDN — Knowledge→Edge→Money drag-map→Jordan
   - network-first for js/css/img assets with 1MB cache cap
   - JSON deliberately never SW-cached (network only, browser HTTP still ok) => offline shell-only
   - stale-while-revalidate for immutable CORE
   - navigationPreload enable, delete old caches, clients.claim, SKIP_WAITING, DENY 504
   - free platform free users, profitability via own edge private, zero-deps true torch auto cuda else cpu
*/
const CACHE='vector-unified-v1-chimera-67';

const CORE=[
  '/',
  '/index.html',
  '/play.html',
  '/model.html',
  '/methods.html',
  '/manifest.json',
  '/offline.html',
  '/assets/shell.css',
  '/assets/responsive.css',
  '/assets/final-qa.css',
  '/assets/unified.css',
  '/assets/motion.css',
  '/assets/lemmino/lemmino.css',
  '/assets/trading-card.css',
  '/assets/nux.css',
  '/assets/player-profile-v28.css',
  '/assets/site-nav.js',
  '/assets/error-boundary.js',
  '/assets/keyboard-a11y.js',
  '/assets/pwa-install.js',
  '/assets/icon-192.png',
  '/assets/icon-512.png',
  '/assets/og-1200x630.png',
  '/assets/og-embed.png'
];

const DENY=[
  '/assets/vectors.json',
  '/assets/unified.json',
  '/assets/vectors_full.json',
  '/assets/vectors_lite.json',
  '/assets/vectors_map_lite.json',
  '/assets/mtnn.onnx',
  '/assets/mtnn.onnx.data',
  '/assets/mtnn_embeddings.f32',
  '/assets/mtnn_heads.f32',
  '/data/unified.json',
  '/assets/data/unified.json'
];

function isDenied(p){ return DENY.some(x=> p.includes(x) ); }
function isImmutable(url){ return CORE.includes(url.pathname) || CORE.includes(url.pathname.replace('/index.html','/')); }
function isAsset(url){
  const p=url.pathname;
  if(!p.startsWith('/assets/')) return false;
  return p.endsWith('.js')||p.endsWith('.css')||p.endsWith('.png')||p.endsWith('.svg')||p.endsWith('.webp')||p.endsWith('.woff2');
}

self.addEventListener('install', e=>{
  self.skipWaiting();
  e.waitUntil((async()=>{
    const cache=await caches.open(CACHE);
    const results=await Promise.allSettled(CORE.map(u=> cache.add(new Request(u,{cache:'reload'}))));
    const failed=results.filter(r=> r.status==='rejected');
    if(failed.length) console.warn('[sw unified v1] CORE precache partial',failed.length);
  })());
});

self.addEventListener('activate', e=>{
  e.waitUntil((async()=>{
    if('navigationPreload' in self.registration){ try{ await self.registration.navigationPreload.enable(); }catch{} }
    const keys=await caches.keys();
    await Promise.all(keys.filter(k=> k!==CACHE).map(k=> caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', e=>{
  const req=e.request;
  if(req.method!=='GET') return;
  const url=new URL(req.url);
  if(url.origin!==location.origin) return;

  // DENY large vectors -> network only 504 offline
  if(isDenied(url.pathname)){
    e.respondWith(fetch(req).catch(()=> new Response('',{status:504,statusText:'Denied offline'})));
    return;
  }

  const isNav = req.mode==='navigate' || (req.headers.get('accept')||'').includes('text/html');

  if(isNav){
    e.respondWith((async()=>{
      try{
        const preload=await e.preloadResponse;
        if(preload){
          const c=await caches.open(CACHE);
          c.put(req,preload.clone()).catch(()=>{});
          return preload;
        }
        const net=await fetch(req);
        if(net&&net.ok){ const c=await caches.open(CACHE); c.put(req,net.clone()).catch(()=>{}); }
        return net;
      }catch{
        const cached=await caches.match(req);
        if(cached) return cached;
        const off=await caches.match('/offline.html');
        if(off) return off;
        return caches.match('/')||caches.match('/index.html')||new Response('Offline',{status:503});
      }
    })());
    return;
  }

  if(isImmutable(url)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE);
      const cached=await cache.match(req);
      const fp=fetch(req).then(r=>{ if(r&&r.ok) cache.put(req,r.clone()).catch(()=>{}); return r; }).catch(()=> null);
      if(cached){ e.waitUntil(fp); return cached; }
      const net=await fp;
      return net||cached||Response.error();
    })());
    return;
  }

  if(isAsset(url)){
    e.respondWith((async()=>{
      const cache=await caches.open(CACHE);
      try{
        const net=await fetch(req);
        if(net&&net.ok){
          const size=parseInt(net.headers.get('content-length')||'0',10);
          if(size<1_000_000) cache.put(req,net.clone()).catch(()=>{});
        }
        return net;
      }catch{
        const cached=await cache.match(req);
        if(cached) return cached;
        return new Response('',{status:504,statusText:'Asset offline'});
      }
    })());
    return;
  }

  // JSON / everything else never SW-cached (network only, browser HTTP cache still applies) — but fallback to cache if present for shell resilience
  e.respondWith((async()=>{
    const cached=await caches.match(req);
    if(cached) return cached;
    try{ return await fetch(req); }
    catch{ return new Response('',{status:504,statusText:'Offline'}); }
  })());
});

self.addEventListener('push', e=>{
  let d={}; try{ d=e.data? e.data.json(): {}; }catch{}
  const title=d.title||'Vector Unified — Daily Chimera';
  const body=d.body||'20,719 stars joint chimera dailySeed LCG — today\'s chimera live';
  e.waitUntil(self.registration.showNotification(title,{body, icon:'/assets/icon-192.png', badge:'/assets/icon-192.png', tag:'vector-unified-daily', data:{url:d.url||'/play?mode=daily&utm_source=push'}}));
});

self.addEventListener('notificationclick', e=>{
  e.notification.close();
  let url=(e.notification.data&&e.notification.data.url)||'/play?mode=daily&utm_source=push_click';
  if(typeof url!=='string'||!url.startsWith('/')||url.startsWith('//')) url='/play?mode=daily&utm_source=push_click';
  e.waitUntil((async()=>{
    const wins=await clients.matchAll({type:'window',includeUncontrolled:true});
    for(const w of wins){
      if(w.url.includes(self.location.origin)){
        await w.focus();
        if('navigate' in w){ try{ await w.navigate(url);}catch{ w.location=url; } } else w.location=url;
        return;
      }
    }
    return clients.openWindow(url);
  })());
});

self.addEventListener('message', e=>{ if(e.data&&e.data.type==='SKIP_WAITING') self.skipWaiting(); });
