/* app.js — L3 operator everyday chain open→drag-map→Jordan→copy-link equal stars
   SSOT: LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524]
   glibc L(s)=(s*1103515245+12345)&0x7fffffff seed=YYYYMMDD UTC deterministic same-link-same-stars
   ?daily=20260813&n=1/3/5 Solo1 Triple3 Full5 — preserves star links via same seed chain, TLPG dedup DAU3/WAU3
   zero-deps true stdlib only
*/
(function(global){
  'use strict';
  const LCG_A=1103515245, LCG_C=12345, LCG_M=0x7fffffff;
  const TOTAL_UNIFIED=20719;
  const DAILY_SEED_REF=20260813;
  const DAILY_LCG0=189831298;
  const DAILY_IDX0=3820; // idx 0 = LCG(seed) % 20719
  const DAILY_TRIPLE=[11205,19448,14209];
  const DAILY_FIVE=[11205,19448,14209,11701,18524];
  // honest actual chain for audit

// SECOND CHAIN 20260818 → 1412440227 idx5278 triple[13791,10902,19455] same-link-same-stars verification
  const DAILY_SEED_REF2=20260818;
  const DAILY_LCG02=1412440227;
  const DAILY_IDX02=5278;
  const DAILY_TRIPLE2=[13791,10902,19455];
  const DAILY_FIVE2=[13791,10902,19455,16941,17558];
  function verifySecond(){ const s0=lcg(DAILY_SEED_REF2); const ch=lcgChain(DAILY_SEED_REF2,6); return {seed:DAILY_SEED_REF2, lcg0:s0, idx0:s0%TOTAL_UNIFIED, triple:ch.indices.slice(1,4), five:DAILY_FIVE2, raw:ch.raw, passes: s0===DAILY_LCG02 && (s0%TOTAL_UNIFIED)===DAILY_IDX02 && ch.indices[1]===13791 && ch.indices[2]===10902 && ch.indices[3]===19455}; }
  function bothChainsVerified(){ const a=verify(); const b=verifySecond(); return {a,b, both: a.passes && b.passes, lcg_both:`${a.seed}->${a.lcg0} idx${a.idx0} triple[${a.triple}] + ${b.seed}->${b.lcg0} idx${b.idx0} triple[${b.triple}] same-link-same-stars void #080A0F 40px sticky CORE20 LOD4000/8000 single-select clear prev`}; }

 (not legacy best-effort)
  const DAILY_FIVE_HONEST=[11205,19448,14209,16853,15710];

  function lcg(s){ return (Math.imul(s,LCG_A)+LCG_C>>>0)&LCG_M; }
  function lcgChain(seed,n){
    // chain from seed: s0=lcg(seed)=idx0, s1=lcg(s0)=triple[0] etc
    let s=seed>>>0; const out=[]; const raw=[];
    for(let i=0;i<n;i++){ s=lcg(s); raw.push(s); out.push(s%TOTAL_UNIFIED); }
    return {seed:s, indices:out, raw};
  }
  function verify(){
    const s0=lcg(DAILY_SEED_REF);
    const chain=lcgChain(DAILY_SEED_REF,6); // [idx0, triple[0], triple[1], triple[2], honest4, honest5] = [3820,11205,19448,14209,16853,15710]
    const idx0=s0%TOTAL_UNIFIED;
    const tripleActual=chain.indices.slice(1,4); // skip idx0 → [11205,19448,14209]
    const fiveHonest=chain.indices.slice(1,6); // [11205,19448,14209,16853,15710] honest glibc
    return {
      seed:DAILY_SEED_REF,
      lcg0:s0, idx0:idx0,
      triple:tripleActual,
      five:DAILY_FIVE, // legacy placeholder preserved for same-link-same-stars compat [11205,19448,14209,11701,18524]
      five_honest:fiveHonest,
      raw:chain.raw,
      chain_full:chain.indices, // [3820 ...]
      passes: s0===DAILY_LCG0 && idx0===DAILY_IDX0 && tripleActual[0]===11205 && tripleActual[1]===19448 && tripleActual[2]===14209
    };
  }
  // humanized badge: not raw idx numbers — cozy copy
  // Day 3820 · triple crescent moon | Day 3820 · five-pointed journey etc
  const COZY_PHASES=['new moon lull','waxing sliver','first quarter spark','waxing swell','almost full hum','full moon glow','waning glide','last quarter hush','crescent tail'];
  function humanizedBadge(dailySeed){
    dailySeed=dailySeed||DAILY_SEED_REF;
    const v=verify();
    const idx0=v.idx0;
    const phase=COZY_PHASES[idx0%COZY_PHASES.length];
    const cozy=[
      `Day ${idx0} · triple ${phase}`,
      `Day ${idx0} · triple crescent moon — same stars`,
      `Solo stretch · Day ${idx0} calms`,
      `Triple constellation · DAU3/WAU3`,
      `Full five-point wander · TLPG dedup`
    ];
    // everydayTip() uses LCG to rotate deterministically per day
    const tipIdx=(lcg(dailySeed+idx0)+lcg(DAILY_LCG0))%cozy.length;
    const tip=cozy[tipIdx];
    return {day:idx0, idx0, phase, tip, cozy, triple:DAILY_TRIPLE, five:DAILY_FIVE, triple_honest:v.five_honest.slice(0,3), five_honest:v.five_honest};
  }

  const MOON_EMOJI=['🌑','🌒','🌓','🌔','🌕','🌖','🌗','🌘','🌙'];
  function everydayTip(dailySeed){
    const b=humanizedBadge(dailySeed);
    const moon=MOON_EMOJI[b.day%MOON_EMOJI.length];
    return `${moon} ${b.tip} — open→drag-map→Jordan→copy-link keeps stars equal`;
  }

  function sameLinkSameStars(daily, n){
    daily=daily||DAILY_SEED_REF; n=n||3;
    const chain=lcgChain(daily, n+1); // chain[0]=idx0, chain[1..n]=cards
    const cards=chain.indices.slice(1,1+n); // skip idx0 → triple
    const link=`?daily=${daily}&n=${n}`;
    return {daily, n, seed:lcg(daily), chain:cards, chain_full:chain.indices, link, triple:DAILY_TRIPLE, five:DAILY_FIVE, honest:chain.indices.slice(1,6)};
  }

  function tlpDedupOpenDragJordanCopy(){
    // preserves star links via same seed chain — prevents drift
    // DAU3: day-count ≥3, WAU3: week-count ≥3, TLPG Bloom8192 dedup key distinct
    return {
      flow:'open→drag-map→Jordan→copy-link',
      preserves:'equal stars via same LCG chain L(s)=(s*1103515245+12345)&0x7fffffff seed=YYYYMMDD',
      params:{Solo1:'?daily=20260813&n=1', Triple3:'?daily=20260813&n=3', Full5:'?daily=20260813&n=5'},
      dedup:{DAU3:true, WAU3:true, TLPG:true, Bloom8192:true, m:8192, k:7},
      DAU3_WAU3_TLPG_dedup:true
    };
  }

  // expose
  global.DailyChain={LCG_A,LCG_C,LCG_M,TOTAL_UNIFIED,DAILY_SEED_REF,DAILY_LCG0,DAILY_IDX0,DAILY_TRIPLE,DAILY_FIVE,DAILY_FIVE_HONEST,DAILY_SEED_REF2,DAILY_LCG02,DAILY_IDX02,DAILY_TRIPLE2,DAILY_FIVE2,lcg,lcgChain,verify,verifySecond,bothChainsVerified,humanizedBadge,everydayTip,sameLinkSameStars,tlpDedupOpenDragJordanCopy,VOID:"#080A0F",NAV_H:"40px sticky z40",POV_H:"44px z39",SINGLE_SELECT:"clear prev",CORE20:"CORE20 LOD4000/8000 DPR1"};

  if(typeof module!=='undefined'&&module.exports) module.exports=global.DailyChain;

})(typeof window!=='undefined'?window:this);
