"use strict";

(function () {
  var canvas = document.getElementById("depth-canvas");
  var ctx = canvas.getContext("2d");
  var fpsEl = document.getElementById("fps-display");
  var statsEl = document.getElementById("stats-display");
  var lastTime = performance.now();
  var frameCount = 0;
  var lastFrameTime = 0;

  canvas.width = 640;
  canvas.height = 480;

  function drawFrame() {
    frameCount++;
    var now = performance.now();
    if (now - lastTime >= 1000) {
      if (fpsEl) fpsEl.textContent = "FPS: " + frameCount;
      frameCount = 0;
      lastTime = now;
    }
    requestAnimationFrame(drawFrame);
  }

  function loadFrame() {
    var url = "/depth/snapshot?_t=" + Date.now();
    var img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = function () {
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth || 640;
        canvas.height = img.naturalHeight || 480;
      }
      ctx.drawImage(img, 0, 0);
      lastFrameTime = performance.now();
    };
    img.onerror = function () {
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#666";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for depth feed...", canvas.width / 2, canvas.height / 2);
    };
    img.src = url;
  }

  // Poll stats every second
  (function pollStats() {
    fetch("/api/depth/stats")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (statsEl) {
          statsEl.textContent =
            "min " + data.min.toFixed(2) +
            "m  mean " + data.mean.toFixed(2) +
            "m  max " + data.max.toFixed(2) +
            "m  cov " + (data.coverage * 100).toFixed(0) + "%";
        }
      })
      .catch(function () {});
    setTimeout(pollStats, 1000);
  })();

  // Start rendering and loading frames
  drawFrame();
  loadFrame();
  setInterval(loadFrame, 100);
})();
