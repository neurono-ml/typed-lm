// typed-lm — client-side Mermaid rendering for mdBook.
//
// mdBook does not render ```mermaid fenced blocks natively; it emits them as
// <pre><code class="language-mermaid">. This script loads the bundled
// mermaid.min.js (already on the page via book.toml additional-js) and renders
// every such block into an accessible <div class="mermaid"> as an SVG, with a
// role="img" and aria-label from the code's accTitle/accDescr (or a fallback).
//
// The theme uses mermaid "base" with a fixed violet/blue palette inspired by
// the IndieSync design tokens, so the diagrams are legible on both the light
// (#ffffff) and dark (#0b0b0f) page backgrounds.

(function () {
  var THEME_VARIABLES = {
    fontFamily: 'inherit',
    fontSize: '13px',
    textColor: '#18181b',
    primaryTextColor: '#3b0764',
    primaryColor: '#ede9fe',
    primaryBorderColor: '#7c3aed',
    secondaryTextColor: '#0c4a6e',
    secondaryColor: '#dbeafe',
    secondaryBorderColor: '#2563eb',
    tertiaryTextColor: '#064e3b',
    tertiaryColor: '#d1fae5',
    tertiaryBorderColor: '#059669',
    lineColor: '#52525c',
    edgeLabelBackground: '#f4f4f5',
    clusterBkg: '#f4f4f5',
    clusterBorder: '#a8a8b3',
    nodeBorder: '#7c3aed',
    mainBkg: '#ede9fe',
  };

  function renderAll() {
    if (typeof window.mermaid === 'undefined') {
      // mermaid.min.js not loaded yet — retry once after a tick.
      setTimeout(renderAll, 100);
      return;
    }
    var blocks = document.querySelectorAll('pre code.language-mermaid');
    if (!blocks.length) return;
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      themeVariables: THEME_VARIABLES,
      securityLevel: 'loose',
      fontFamily: 'inherit',
      flowchart: {
        htmlLabels: true,
        useMaxWidth: true,
        wrap: true,
        padding: 8,
      },
      sequence: {
        useMaxWidth: true,
      },
    });
    var index = 0;
    blocks.forEach(function (code) {
      var pre = code.parentElement;
      if (pre.dataset.tlMermaid) return;
      pre.dataset.tlMermaid = '1';
      var source = code.textContent;
      var id = 'tl-mermaid-' + index++;
      window.mermaid
        .render(id, source)
        .then(function (result) {
          var container = document.createElement('div');
          container.className = 'mermaid';
          container.setAttribute('role', 'figure');
          container.setAttribute('data-tl-source', source);
          container.innerHTML = result.svg;
          var svg = container.querySelector('svg');
          if (svg) {
            svg.setAttribute('role', 'img');
            svg.setAttribute('aria-label', extractTitle(source));
          }
          pre.replaceWith(container);
        })
        .catch(function (error) {
          var box = document.createElement('div');
          box.className = 'sk-box sk-box--danger';
          box.textContent = 'Mermaid diagram failed to render: ' + error.message;
          pre.replaceWith(box);
        });
    });
  }

  function extractTitle(source) {
    var accTitle = source.match(/\baccTitle:\s*(.+)/);
    if (accTitle) return accTitle[1].trim();
    var accDescr = source.match(/\baccDescr:\s*(.+)/);
    if (accDescr) return accDescr[1].trim();
    var first = source.split('\n').find(function (line) {
      return line.trim().length > 0;
    });
    return (first || 'Mermaid diagram').trim();
  }

  function boot() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', renderWhenFontsReady);
    } else {
      renderWhenFontsReady();
    }
  }

  // Mermaid measures node text with the font available at render time. The
  // book webfonts load asynchronously; rendering before they arrive makes
  // measurements too narrow and clips node text. Wait for the faces first,
  // with a timeout fallback so offline builds still render with fallbacks.
  function renderWhenFontsReady() {
    var rendered = false;
    function done() {
      if (rendered) return;
      rendered = true;
      renderAll();
    }
    setTimeout(done, 1500);
    if (document.fonts && document.fonts.load) {
      try {
        Promise.all([
          document.fonts.load('400 13px Inter'),
          document.fonts.load('600 13px "Space Grotesk"'),
          document.fonts.load('700 13px "Space Grotesk"'),
        ]).then(done, done);
      } catch (error) {
        done();
      }
    } else {
      done();
    }
  }

  // Re-render once all fonts settle, so a late-arriving face does not leave
  // clipped labels. Containers stash their source in data-tl-source.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () {
      var containers = document.querySelectorAll('div.mermaid');
      if (!containers.length) return;
      var rebuilt = false;
      containers.forEach(function (container) {
        var source = container.getAttribute('data-tl-source');
        if (!source) return;
        var pre = document.createElement('pre');
        var code = document.createElement('code');
        code.className = 'language-mermaid';
        code.textContent = source;
        pre.appendChild(code);
        container.replaceWith(pre);
        rebuilt = true;
      });
      if (rebuilt) renderAll();
    });
  }

  boot();
})();
