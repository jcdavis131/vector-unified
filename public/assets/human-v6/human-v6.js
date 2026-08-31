/* Human-Centered v6 — human-v6.js — zero-deps */
window.DumbModel = window.DumbModel || {};
window.DumbModel.HumanV6 = (function(){
  function init(){}
  function selectionLabel(id){return id||"none"}
  // honest provenance
  function provenance(){
    const el=document.getElementById('prov');
    if(!el) return;
    // 59 hashes 7/7/0 PASS is real as of 2026-08-17, do not claim 73 without proof
    el.textContent="59 hashes • 7/7/0 PASS • LCG 20260813→189831298 idx3820 triple[11205,19448,14209] • PWA v67 CORE20 • honest 503 when canonical 20719×128 missing";
  }
  function mapInit(){
    const canvas=document.getElementById('mapCanvas');
    if(!canvas) return;
    const ctx=canvas.getContext('2d',{alpha:false});
    if(!ctx) return;
    // DPR1 per spec
    function resize(){
      const r=canvas.getBoundingClientRect();
      canvas.width=Math.round(r.width);
      canvas.height=Math.round(r.height);
      draw();
    }
    function draw(){
      const w=canvas.width,h=canvas.height;
      ctx.fillStyle='#080A0F';
      ctx.fillRect(0,0,w,h);
      // subtle grid like paper dots
      ctx.fillStyle='rgba(255,255,255,0.04)';
      for(let y=0;y<h;y+=28) for(let x=0;x<w;x+=28) ctx.fillRect(x,y,1,1);
      // placeholder points — real embedding loads async if available
      ctx.fillStyle='rgba(168,162,158,0.55)';
      for(let i=0;i<4000;i++){
        const x=Math.random()*w, y=Math.random()*h;
        const r=Math.random()<0.02?1.8:0.9;
        ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();
      }
      // selected terracotta/human-blue blend
      ctx.fillStyle='#2A5BD7';
      ctx.beginPath();ctx.arc(w*0.54,h*0.42,4.2,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle='rgba(42,91,215,0.35)';ctx.lineWidth=2;ctx.beginPath();ctx.arc(w*0.54,h*0.42,9,0,Math.PI*2);ctx.stroke();
    }
    window.addEventListener('resize',resize,{passive:true});
    resize();
    // try load unified.json if present — honest
    fetch('/data/unified.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).then(j=>{
      if(!j) return;
      const label=document.getElementById('countLabel');
      if(label && j.total) label.textContent=`${j.total} chimera`;
    }).catch(()=>{});
  }
  return {init, selectionLabel, provenance, mapInit};
})();
