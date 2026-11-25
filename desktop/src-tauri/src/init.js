// Läuft in jeder Seite des App-Fensters (Ladebildschirm und Streamlit-App).
// Externe Links (z. B. zu Hugging Face) öffnen sich im normalen Browser statt im App-Fenster:
// Wir leiten sie auf eine normale Navigation um; die App (lib.rs, on_navigation) fängt sie ab und gibt sie
// an den Browser weiter. So geht auch target="_blank" und window.open nicht verloren.
(function () {
  if (window.__meridianLinks) return;
  window.__meridianLinks = true;
  var LOCAL = ["127.0.0.1", "localhost", "tauri.localhost", "[::1]"];
  function external(href) {
    try {
      var u = new URL(href, location.href);
      if (u.protocol === "mailto:") return true;
      return (u.protocol === "http:" || u.protocol === "https:") && LOCAL.indexOf(u.hostname) < 0;
    } catch (e) {
      return false;
    }
  }
  document.addEventListener("click", function (e) {
    var a = e.target && e.target.closest ? e.target.closest("a[href]") : null;
    if (!a || a.hasAttribute("download") || !external(a.href)) return;
    e.preventDefault();
    e.stopPropagation();
    location.assign(a.href);
  }, true);
  var open = window.open;
  window.open = function (url) {
    if (url && external(String(url))) {
      location.assign(new URL(String(url), location.href).href);
      return null;
    }
    return open.apply(window, arguments);
  };
})();
