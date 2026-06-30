"use strict";

/* Stereo video: WebRTC or HTTP, with side-by-side SBS layout */

(function () {
  var params = new URLSearchParams(window.location.search);
  var transport = params.get("transport") || "http";
  var container = document.getElementById("stereo-container");
  var statusEl = document.getElementById("status-text");

  if (transport === "http") {
    initHTTP();
  } else {
    initWebRTC();
  }

  function initHTTP() {
    statusEl.textContent = "MJPEG streaming...";
    var img = document.createElement("img");
    img.src = "/video_feed";
    img.className = "stereo-img";
    img.alt = "Stereo video";
    container.appendChild(img);
  }

  async function initWebRTC() {
    statusEl.textContent = "WebRTC: connecting...";
    var pc = null;

    // Create SBS video elements
    var leftDiv = document.createElement("div");
    leftDiv.className = "eye left";
    var rightDiv = document.createElement("div");
    rightDiv.className = "eye right";
    var leftVid = document.createElement("video");
    leftVid.autoplay = true;
    leftVid.playsInline = true;
    leftVid.muted = true;
    var rightVid = document.createElement("video");
    rightVid.autoplay = true;
    rightVid.playsInline = true;
    rightVid.muted = true;
    leftDiv.appendChild(leftVid);
    rightDiv.appendChild(rightVid);
    container.appendChild(leftDiv);
    container.appendChild(rightDiv);

    try {
      var configRes = await fetch("/api/webrtc/config");
      var config = await configRes.json();
      pc = new RTCPeerConnection(config);

      var gotLeft = false;
      pc.ontrack = function (ev) {
        var stream = ev.streams[0];
        if (!stream) return;
        if (!gotLeft) {
          leftVid.srcObject = stream;
          gotLeft = true;
        } else {
          rightVid.srcObject = stream;
        }
      };

      var ws = new WebSocket(
        (location.protocol === "https:" ? "wss://" : "ws://") +
          location.host +
          "/ws/webrtc"
      );

      ws.onopen = async function () {
        statusEl.textContent = "WebRTC: creating offer...";
        var offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        ws.send(JSON.stringify({ type: "webrtc_offer", sdp: offer.sdp }));
      };

      ws.onmessage = async function (ev) {
        var msg = JSON.parse(ev.data);
        if (msg.type === "webrtc_answer") {
          await pc.setRemoteDescription(
            new RTCSessionDescription({ type: "answer", sdp: msg.sdp })
          );
          statusEl.textContent = "WebRTC: connected";
        } else if (msg.type === "error") {
          statusEl.textContent = "WebRTC error: " + msg.message;
        } else if (msg.type === "signaling_ready") {
          // waiting for offer
        }
      };

      ws.onerror = function () {
        statusEl.textContent = "WebRTC signaling error";
      };

      pc.onconnectionstatechange = function () {
        if (pc) {
          statusEl.textContent = "WebRTC: " + pc.connectionState;
        }
      };
    } catch (err) {
      statusEl.textContent = "WebRTC error: " + err.message;
    }
  }
})();
