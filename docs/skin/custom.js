// typed-lm — theme interactions and runtime SEO/GEO metadata injection for mdBook.

// Keep only Light and Dark in the theme picker (mdBook renders the list
// server-side; extra options are removed and "Coal" is relabelled "Dark").
(function () {
  function cleanThemeList() {
    var list = document.getElementById('mdbook-theme-list');
    if (!list || list.dataset.tlCleaned) return;
    list.dataset.tlCleaned = '1';
    ['default_theme', 'rust', 'navy', 'ayu'].forEach(function (suffix) {
      var button = document.getElementById('mdbook-theme-' + suffix);
      if (button && button.parentElement) button.parentElement.remove();
    });
    var coal = document.getElementById('mdbook-theme-coal');
    if (coal) coal.textContent = 'Dark';
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', cleanThemeList);
  } else {
    cleanThemeList();
  }
})();

// Sidebar brand block.
(function () {
  var scrollbox = document.querySelector('.sidebar-scrollbox');
  if (scrollbox && !scrollbox.querySelector('.brand')) {
    var brand = document.createElement('div');
    brand.className = 'brand';
    brand.innerHTML =
      '<span class="brand-mark">tl</span>' +
      '<span><span class="brand-name">typed-lm</span>' +
      '<span class="brand-tagline">typed decisions in Rust</span></span>';
    scrollbox.insertBefore(brand, scrollbox.firstChild);
  }
})();

// Favicon (mdBook links a bundled favicon.png; point it at our SVG instead).
(function () {
  function setFavicon() {
    var link = document.head.querySelector('link[rel="icon"]');
    if (!link) {
      link = document.createElement('link');
      link.setAttribute('rel', 'icon');
      document.head.appendChild(link);
    }
    link.setAttribute('type', 'image/svg+xml');
    link.setAttribute('href', '/typed-lm/favicon.svg');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setFavicon);
  } else {
    setFavicon();
  }
})();

// Runtime SEO/GEO metadata: canonical link, Open Graph, Twitter and JSON-LD.
(function () {
  var SITE = 'https://neurono-ml.github.io/typed-lm';

  function setMeta(attribute, key, value) {
    if (!value) return;
    var element = document.head.querySelector('meta[' + attribute + '="' + key + '"]');
    if (!element) {
      element = document.createElement('meta');
      element.setAttribute(attribute, key);
      document.head.appendChild(element);
    }
    element.setAttribute('content', value);
  }

  function inject(description) {
    // mdBook 0.5 does not emit a canonical link, so build one from the site
    // root. The document lives under the site subpath (for example
    // /typed-lm/), captured here so the subpath is never doubled.
    var link = document.head.querySelector('link[rel="canonical"]');
    var canonical;
    if (link && link.getAttribute('href')) {
      canonical = link.getAttribute('href');
    } else {
      var siteRoot = SITE + '/';
      var withinSite = window.location.pathname.startsWith('/typed-lm/')
        ? window.location.pathname.slice('/typed-lm/'.length)
        : window.location.pathname.replace(/^\//, '');
      canonical = siteRoot + withinSite;
      link = document.createElement('link');
      link.setAttribute('rel', 'canonical');
      link.setAttribute('href', canonical);
      document.head.appendChild(link);
    }
    setMeta('property', 'og:site_name', 'typed-lm');
    setMeta('property', 'og:type', 'article');
    setMeta('property', 'og:title', document.title);
    setMeta('property', 'og:description', description);
    setMeta('property', 'og:url', canonical);
    setMeta('name', 'twitter:card', 'summary');
    setMeta('name', 'twitter:title', document.title);
    setMeta('name', 'twitter:description', description);
    setMeta('name', 'description', description);

    var existing = document.getElementById('tl-jsonld');
    if (existing) existing.remove();
    var script = document.createElement('script');
    script.type = 'application/ld+json';
    script.id = 'tl-jsonld';
    script.textContent = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'TechArticle',
      headline: document.title,
      description: description,
      inLanguage: 'en',
      isPartOf: {
        '@type': 'Book',
        name: 'typed-lm documentation',
        url: SITE + '/',
      },
      about: [
        'Rust',
        'LLM inference',
        'semantic routing',
        'Candle',
        'LoRA',
        'quantization',
      ],
      programmingLanguage: 'Rust',
    });
    document.head.appendChild(script);
  }

  function boot() {
    var content = document.querySelector('#content main .content, #content .content, main');
    if (!content) return;
    var paragraph = content.querySelector('p');
    var description = paragraph
      ? paragraph.textContent.trim().replace(/\s+/g, ' ').slice(0, 300)
      : document.title;
    inject(description);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();

// Scroll reveal plus animated counters on the home page.
document.addEventListener('DOMContentLoaded', function () {
  var reveals = document.querySelectorAll('.reveal');

  function animateCount(element) {
    var target = parseFloat(element.getAttribute('data-target')) || 0;
    var suffix = element.getAttribute('data-suffix') || '';
    var start = null;
    function step(timestamp) {
      if (!start) start = timestamp;
      var progress = Math.min((timestamp - start) / 950, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = Math.round(target * eased) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
            entry.target.querySelectorAll('.js-count').forEach(animateCount);
          }
        });
      },
      { threshold: 0.12 }
    );
    reveals.forEach(function (element) {
      observer.observe(element);
    });
  } else {
    reveals.forEach(function (element) {
      element.classList.add('is-visible');
    });
    document.querySelectorAll('.js-count').forEach(animateCount);
  }
});

// Font-size control (A- / A / A+), persisted in localStorage.
(function () {
  var STORAGE_KEY = 'tl-font-scale';
  var MINIMUM_SCALE = 0;
  var MAXIMUM_SCALE = 2;
  var DEFAULT_SCALE = 0;

  function readScale() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      var parsed = parseInt(raw, 10);
      if (parsed >= MINIMUM_SCALE && parsed <= MAXIMUM_SCALE) return parsed;
    } catch (error) {
      // storage unavailable — fall through
    }
    return DEFAULT_SCALE;
  }

  function applyScale(scale) {
    document.documentElement.setAttribute('data-sk-font-scale', String(scale));
    try {
      window.localStorage.setItem(STORAGE_KEY, String(scale));
    } catch (error) {
      // storage unavailable — attribute still applies
    }
    document.querySelectorAll('.sk-font-btn').forEach(function (button) {
      var action = button.getAttribute('data-sk-font-action');
      if (action === 'reset') {
        button.setAttribute('aria-pressed', scale === DEFAULT_SCALE ? 'true' : 'false');
      } else {
        button.disabled =
          (action === 'decrease' && scale <= MINIMUM_SCALE) ||
          (action === 'increase' && scale >= MAXIMUM_SCALE);
      }
    });
  }

  function injectControls() {
    var bar = document.querySelector('#mdbook-menu-bar .left-buttons');
    if (!bar || bar.querySelector('.sk-font-btn')) return;
    var group = document.createElement('div');
    group.className = 'sk-font-group';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'Text size');
    group.innerHTML =
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="decrease" title="Decrease text size" aria-label="Decrease text size">A&#8722;</button>' +
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="reset" title="Reset text size" aria-label="Reset text size">A</button>' +
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="increase" title="Increase text size" aria-label="Increase text size">A+</button>';
    bar.appendChild(group);
    group.addEventListener('click', function (event) {
      var button = event.target.closest('.sk-font-btn');
      if (!button) return;
      var action = button.getAttribute('data-sk-font-action');
      var scale = readScale();
      if (action === 'decrease') scale = Math.max(MINIMUM_SCALE, scale - 1);
      else if (action === 'increase') scale = Math.min(MAXIMUM_SCALE, scale + 1);
      else scale = DEFAULT_SCALE;
      applyScale(scale);
    });
    applyScale(readScale());
  }

  applyScale(readScale());
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectControls);
  } else {
    injectControls();
  }
})();
