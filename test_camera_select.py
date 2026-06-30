"""Camera selector test: serve MJPEG from a specific camera index."""
import sys, argparse, asyncio
from aiohttp import web
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

INDEX = args.index
FPS = 10

async def mjpeg_handler(request):
    cap = cv2.VideoCapture(INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Test] Camera index {INDEX}: {w}x{h}")

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache",
        },
    )
    await response.prepare(request)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ret2, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret2:
                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
            await asyncio.sleep(1.0 / FPS)
    finally:
        cap.release()
        await response.write_eof()
    return response

async def index_handler(request):
    return web.Response(
        text=f"""<!doctype html>
<html><head><title>Camera {INDEX}</title></head>
<body style="margin:0;background:#000;text-align:center">
<h1 style="color:#fff">Camera Index {INDEX}</h1>
<img src="/cam" style="max-width:100%">
</body></html>""",
        content_type="text/html",
    )

app = web.Application()
app.router.add_get("/", index_handler)
app.router.add_get("/cam", mjpeg_handler)
print(f"Serving camera {INDEX} at http://localhost:9002/")
web.run_app(app, host="0.0.0.0", port=9002)
