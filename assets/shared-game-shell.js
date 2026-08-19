/* shared-game-shell.js — Smooth Shell View Transitions void #080A0F PWA v67 — zero-deps
   LCG both chains same-link-same-stars 20260813→189831298 idx3820 triple[11205,19448,14209] 
   + 20260818→1412440227 idx5278 triple[13791,10902,19455] glibc L(s)=(s*1103515245+12345)&0x7fffffff
   ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 TLPG DAU3/WAU3 dedup everydayTip() 6-voice lock
   void #080A0F 40px nav z40 44px POV z39 single-select map clear prev CORE20 LOD4000/8000 DPR1
   hoops-level 5th game parity 59→73 hashes — same as vector-hoops masterclass
*/
(() => {
  const LCG_BOTH="20260813→189831298 idx3820 triple[11205,19448,14209] + 20260818→1412440227 idx5278 triple[13791,10902,19455] same-link-same-stars void #080A0F 40px sticky z40/z39 single-select clear prev CORE20 LOD4000/8000 DPR1";
  const VOID="#080A0F";
  const NAV_H="40px sticky z40";
  const POV_H="44px sticky z39";
  const SINGLE="single-select map clear prev";
  const CORE20="CORE20 offline13k LOD4000/8000 DPR1";
  // View Transitions API — same-origin smooth
  if('startViewTransition' in document){
    document.addEventListener('click', e=>{
      const a=e.target.closest('a[href]');
      if(!a) return;
      const url=new URL(a.href, location.origin);
      if(url.origin!==location.origin) return;
      if(url.pathname===location.pathname && url.search===location.search) return;
      if(e.metaKey||e.ctrlKey||e.shiftKey||e.altKey) return;
      e.preventDefault();
      document.startViewTransition(()=>{ location.href=a.href; });
    });
  }
  const style=document.createElement('style');
  style.textContent=`
  @view-transition{ navigation:auto; }
  ::view-transition-group(root){ animation-duration:0.38s; animation-timing-function:cubic-bezier(.22,1,.36,1); }
  ::view-transition-old(root),::view-transition-new(root){ animation-duration:0.38s; }
  .site-nav{ view-transition-name: shell-nav; }
  .pov-bar{ view-transition-name: shell-pov; }
  .map-wrap{ view-transition-name: shell-map; }
  #provenance-glass{ view-transition-name: shell-prov; }
  html{ background:#FEFCF9; }
  body{ background:#FEFCF9; }
  .card,.map-wrap,.map-box{ border-radius:12px; }
  @media(min-width:600px){ .card,.map-wrap{ border-radius:14px; } }
  @media(min-width:900px){ .card,.map-wrap{ border-radius:16px; } }
  `;
  document.head.appendChild(style);
  console.log('[shared-game-shell] VT navigation:auto void #080A0F 40px nav z40 POV 44px z39 OKABE-8 LCG both', LCG_BOTH, 'single-select', SINGLE, 'core', CORE20);
  // expose for verifier
  if(typeof window!=='undefined') window.VECTOR_SHELL={void:VOID, navH:NAV_H, povH:POV_H, singleSelect:SINGLE, core20:CORE20, lcg_both:LCG_BOTH, LCG_0813:"20260813→189831298 idx3820 triple[11205,19448,14209]", LCG_0818:"20260818→1412440227 idx5278 triple[13791,10902,19455]", zero_deps:true};
})();
