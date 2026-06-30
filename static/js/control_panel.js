"use strict";

(function () {
  var KEYS = ["crop", "xshift", "k1", "k2", "sep", "gshift_x", "gshift_y"];
  var form = document.getElementById("params-form");
  var loading = document.getElementById("loading-msg");
  var msgEl = document.getElementById("apply-msg");

  // --- Load current params ---
  fetch("/api/params")
    .then(function (res) { return res.json(); })
    .then(function (params) {
      loading.hidden = true;
      form.hidden = false;
      populate(params);

      // Slider live feedback
      for (var i = 0; i < KEYS.length; i++) {
        var key = KEYS[i];
        var el = form.elements[key];
        var val = document.getElementById(key + "-val");
        if (!el || !val) continue;
        el.addEventListener("input", function () {
          val.textContent = this.value;
        });
      }

      // --- Apply on submit ---
      form.addEventListener("submit", function (e) {
        e.preventDefault();
        var update = {};
        for (var j = 0; j < KEYS.length; j++) {
          var k = KEYS[j];
          var el2 = form.elements[k];
          if (el2) update[k] = parseFloat(el2.value);
        }
        fetch("/api/params", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(update),
        })
          .then(function (r) { return r.json(); })
          .then(function (next) {
            populate(next);
            showMsg("Applied", "ok");
          })
          .catch(function () { return showMsg("Failed", "err"); });
      });

      // --- Reset ---
      document.getElementById("reset-btn").addEventListener("click", function () {
        fetch("/api/params/reset", { method: "POST" })
          .then(function (r) { return r.json(); })
          .then(function (def) {
            populate(def);
            showMsg("Reset", "ok");
          });
      });

      // --- Live preview: reload MJPEG every 2s so slider changes are visible ---
      var previewImg = document.getElementById("preview-img");
      if (previewImg) {
        setInterval(function () {
          previewImg.src = "/video_feed?ts=" + Date.now();
        }, 2000);
      }
    })
    .catch(function (e) {
      loading.textContent = "Failed to load parameters: " + e.message;
    });

  function populate(params) {
    for (var i = 0; i < KEYS.length; i++) {
      var k = KEYS[i];
      var el = form.elements[k];
      var val = document.getElementById(k + "-val");
      if (el) el.value = params[k] != null ? params[k] : 0;
      if (val) val.textContent = params[k] != null ? params[k] : 0;
    }
  }

  function showMsg(text, type) {
    if (!msgEl) return;
    msgEl.textContent = text;
    msgEl.className = "apply-msg " + type;
    msgEl.hidden = false;
    setTimeout(function () { msgEl.hidden = true; }, 2000);
  }
})();
