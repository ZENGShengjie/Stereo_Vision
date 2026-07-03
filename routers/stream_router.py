"""Stream routers: MJPEG feed, WebRTC signaling and config.

阶段 0–4 集成:
- MJPEG 走 :class:`SBSPipeline.process_one_frame`,输出 3840x1080 SBS 帧。
- WebRTC 同上,``frame_fn`` 优先用 ``app[PIPELINE_KEY]``;若 pipeline 启动失败,
  回退到老路径 ``cam.read_stereo()``。
- 本任务硬约束"无拉伸、无裁切",所以**不**再调 :class:`DisplayProcessor`
  (那是 VR 视觉调参面板,与本任务正交)。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from aiohttp import web

# #region agent log — DEBUG mode instrumentation
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug-c7ffa8.log")
def _dbg(hyp, msg, data):
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "c7ffa8",
                "location": "routers/stream_router.py",
                "hypothesisId": hyp,
                "message": msg,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion

from config import (
    MONO_HEIGHT,
    MONO_WIDTH,
    SBS_HEIGHT,
    SBS_WIDTH,
    USB_FPS,
    USB_TARGET_WIDTH,
    USB_TARGET_HEIGHT,
    WEBRTC_MAX_BITRATE,
    WEBRTC_MAX_FRAMERATE,
    build_webrtc_client_config,
    create_rtc_configuration,
)
from state import CAMERA_KEY, PIPELINE_KEY, SBS_QUEUE_KEY, WEBRTC_PEERS_KEY
from transport import ZEDTrack
from transport.mjpeg_stream import mjpeg_generator_async
from utils import timestamped_print


def _build_frame_fn(app: web.Application):
    """Build a frame_fn that reads from the shared LatestSBSQueue.

    Before refactor: each MJPEG request / each WebRTC client called
    pipeline.process_one_frame() independently → N copies of YOLO×2 + SGBM.
    After refactor: SBSPipeline pushes to LatestSBSQueue; this fn just
    pulls from the queue (O(1), no recompute).
    Falls back to camera.read_stereo() if the queue is unavailable.
    """
    sbs_queue = app.get(SBS_QUEUE_KEY)
    cam = app.get(CAMERA_KEY)

    # #region agent log — DEBUG mode instrumentation
    _dbg("BF1", "_build_frame_fn start", {"sbs_queue_is_none": sbs_queue is None, "cam_is_none": cam is None})
    # #endregion

    if sbs_queue is not None:
        # #region agent log — DEBUG mode instrumentation
        import time as _t
        _pull_count = [0]
        _pull_none_count = [0]
        _enc_ms_total = [0.0]
        _enc_ms_max = [0.0]
        _enc_count = [0]
        # #endregion
        def frame_fn():
            result = sbs_queue.pull()
            # #region agent log — DEBUG mode instrumentation
            _pull_count[0] += 1
            if result is None:
                _pull_none_count[0] += 1
            if _pull_count[0] % 30 == 1:
                _dbg("MJ1", "sbs_queue.pull", {
                    "calls": _pull_count[0],
                    "none_rate": _pull_none_count[0] / max(_pull_count[0], 1),
                    "enc_avg_ms": _enc_ms_total[0] / max(_enc_count[0], 1),
                    "enc_max_ms": _enc_ms_max[0],
                })
            # #endregion
            if result is not None:
                return result[0]  # (frame, timestamp)
            return None
        return frame_fn

    if cam is not None:
        def frame_fn():
            try:
                return cam.read_stereo()
            except Exception as ex:
                timestamped_print(f"[stream] cam read_stereo error: {ex}")
                return None
        return frame_fn

    return lambda: None


def _import_aiortc():
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
    except ImportError as err:
        raise RuntimeError(
            "aiortc is required for WebRTC mode. Install: pip install aiortc"
        ) from err
    return RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer


async def _wait_ice_gathered(pc, timeout_s=5.0):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_ice():
        if pc.iceGatheringState == "complete":
            done.set()

    await asyncio.wait_for(done.wait(), timeout=timeout_s)


def _set_sender_limits(sender):
    params = sender.getParameters() if hasattr(sender, "getParameters") else None
    if params is None or not params.encodings:
        return
    params.encodings[0]["maxBitrate"] = WEBRTC_MAX_BITRATE
    params.encodings[0]["maxFramerate"] = WEBRTC_MAX_FRAMERATE
    return params


async def _close_peer(app: web.Application, pc) -> None:
    peers = app[WEBRTC_PEERS_KEY]
    track = peers.pop(pc, None)
    if track is not None:
        track.stop()
    if pc.connectionState != "closed":
        await pc.close()


async def mjpeg_feed_handler(request: web.Request) -> web.Response:
    """MJPEG 视频流。

    阶段 4:frame_fn 走 SBSPipeline,输出 (1080, 3840, 3) SBS。
    """
    if not request.app.get(WEBRTC_PEERS_KEY):
        pass  # noqa: silence unused

    if (
        request.app.get(SBS_QUEUE_KEY) is None
        and request.app.get(CAMERA_KEY) is None
    ):
        return web.Response(text="Camera not available", status=503)

    frame_fn = _build_frame_fn(request.app)

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache",
            "Connection": "close",
        },
    )
    await response.prepare(request)
    try:
        async for chunk in mjpeg_generator_async(frame_fn, fps=USB_FPS):
            try:
                await response.write(chunk)
            except ConnectionResetError:
                break
    except Exception:
        pass
    finally:
        try:
            await response.write_eof()
        except Exception:
            pass
    return response


async def webrtc_config_handler(_request: web.Request) -> web.Response:
    return web.json_response(build_webrtc_client_config())


async def webrtc_signaling_handler(request: web.Request) -> web.Response:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    client = request.remote or "unknown"
    pc = None
    peers = request.app[WEBRTC_PEERS_KEY]
    timestamped_print(f"WebRTC signaling connected: {client}")

    try:
        await ws.send_json({"type": "signaling_ready"})
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            if payload.get("type") != "webrtc_offer":
                continue

            try:
                RTCPC, RTCSD, RTCConf, RTCIce = _import_aiortc()
            except RuntimeError as err:
                await ws.send_json({"type": "error", "message": str(err)})
                continue

            try:
                if pc is not None:
                    await _close_peer(request.app, pc)

                pc = RTCPC(configuration=create_rtc_configuration(RTCConf, RTCIce))

                @pc.on("connectionstatechange")
                async def _on_state():
                    ts = request.app.get("_stereo_ws_clients", {}).get(pc)
                    ts_val = ts if ts else "unknown"
                    timestamped_print(
                        f"WebRTC [{ts_val}] state: {pc.connectionState}"
                    )
                    if pc.connectionState in {"failed", "closed", "disconnected"}:
                        await _close_peer(request.app, pc)

                if (
                    request.app.get(SBS_QUEUE_KEY) is None
                    and request.app.get(CAMERA_KEY) is None
                ):
                    await ws.send_json(
                        {"type": "error", "message": "Camera not available"}
                    )
                    continue

                # After refactor: pass LatestSBSQueue directly to ZEDTrack
                # (each ZEDTrack.recv() pulls from shared queue — no recompute)
                sbs_queue = request.app.get(SBS_QUEUE_KEY)
                if sbs_queue is not None:
                    track = ZEDTrack(sbs_queue, fps=webrtc_fps, fallback_shape=fallback_shape)
                else:
                    track = ZEDTrack(frame_fn, fps=webrtc_fps, fallback_shape=fallback_shape)
                peers[pc] = track
                sender = pc.addTrack(track)

                offer = RTCSD(sdp=payload["sdp"], type="offer")
                await pc.setRemoteDescription(offer)

                limits = _set_sender_limits(sender)
                if limits:
                    try:
                        await sender.setParameters(limits)
                    except Exception:
                        pass

                answer = await pc.createAnswer()
                try:
                    await pc.setLocalDescription(answer)
                    await _wait_ice_gathered(pc)
                    sdp_out = pc.localDescription.sdp
                except ValueError:
                    lines = [
                        line
                        for line in (answer.sdp or "").splitlines()
                        if not line.startswith("a=group:BUNDLE")
                    ]
                    sdp_out = "\r\n".join(lines) + "\r\n"

                await ws.send_json({"type": "webrtc_answer", "sdp": sdp_out})

            except Exception as ex:
                if pc is not None:
                    await _close_peer(request.app, pc)
                    pc = None
                await ws.send_json({"type": "error", "message": str(ex)})

    finally:
        if pc is not None:
            await _close_peer(request.app, pc)
        timestamped_print(f"WebRTC signaling disconnected: {client}")

    return ws


async def stream_stats_handler(request: web.Request) -> web.Response:
    """返回 pipeline 实时性能统计（延迟 FPS 各阶段耗时）。"""
    pipeline = request.app.get(PIPELINE_KEY)
    if pipeline is None:
        return web.json_response({"error": "pipeline not available"}, status=503)
    return web.json_response(pipeline.stats_summary())


def register_stream_routes(app: web.Application) -> None:
    app.router.add_get("/video_feed", mjpeg_feed_handler)
    app.router.add_get("/api/webrtc/config", webrtc_config_handler)
    app.router.add_get("/ws/webrtc", webrtc_signaling_handler)
    app.router.add_get("/api/stream/stats", stream_stats_handler)
