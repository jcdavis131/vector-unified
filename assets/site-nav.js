/* Vector Unified site-nav — mirrors hoops shell, brand VECTOR UNIFIED */
(function (global) {
  'use strict';
  var LINKS = [
    { href: '/index.html', label: 'Void', title: '20,719 stars void — chimera daily LCG' },
    { href: '/play.html', label: 'Play', title: 'Chimera daily: guess archetype from cross-sport neighbour' },
    { href: '/model.html', label: 'Lab', title: 'Inside unified: encoders, CORAL, GRL, SupCon' },
    { href: '/methods.html', label: 'Methods', title: 'Every number recomputable — sources + math' },
    { href: '/model.html#ablation', label: 'Ablation', title: 'G1/G2/G3/G4, each loss must earn keep' },
  ];
  function mount() {
    var nav = document.querySelector('.site-nav');
    if (!nav) return;
    var path = nav.getAttribute('data-active') || location.pathname || '/';
    var linksHtml = LINKS.map(function (l) {
      var isActive = path === l.href ||
        (path === '/' && l.href === '/index.html') ||
        (path.startsWith('/model') && l.href.includes('model')) ||
        (path.startsWith('/methods') && l.href.includes('methods')) ||
        (path.startsWith('/play') && l.href.includes('play'));
      return '<a class="site-nav__link' + (isActive ? ' is-active' : '') + '" href="' + l.href + '"' +
        (l.title ? ' title="' + l.title + '"' : '') +
        (isActive ? ' aria-current="page"' : '') + '>' + l.label + '</a>';
    }).join('');
    nav.innerHTML =
      '<a class="site-nav__brand" href="/index.html">VECTOR<span class="site-nav__accent">UNIFIED</span></a>' +
      '<div class="site-nav__links">' + linksHtml + '</div>';
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
  global.VUSiteNav = { mount: mount, links: LINKS };
})(window);
