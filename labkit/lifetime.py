"""Stop a local app's server once its last browser page is closed.

labkit.js holds one connection open to ``/__labkit/live`` while any page of
the app is open: one tab at a time holds it (a Web Lock), and when that tab
goes the next one takes over.  However a tab ends -- closed, reloaded, the
browser quit or crashed -- its connection drops.  Once none has been open for
``GRACE`` seconds and no request is in flight, the server interrupts itself the
way Ctrl-C would, so ``atexit`` handlers run and a save that was under way has
already finished.

    from labkit import lifetime
    lifetime.flask(app)                  # just before app.run(...)
    uvicorn.run(lifetime.asgi(app))      # FastAPI, or any ASGI app

``busy=`` takes a callable that says whether background work is still running
(a batch on a thread, say); the server waits for it as it waits for requests.

A reload or a second tab never stops it: the next page connects well inside the
grace period.  It is never armed until a page has connected, so a server
started without a browser runs as before.  And it stays up for good when
started with ``--keep-alive``, ``--state`` or ``--home`` (a scratch copy under
test), or with ``LABKIT_KEEP_ALIVE=1`` in the environment: automation must not
lose its server when a headless page closes.  (labkit.js also does not connect
in a WebDriver/Playwright-driven browser.)
"""

from __future__ import annotations

import os
import select
import signal
import socket
import sys
import threading
import time

GRACE = 10.0     # seconds with no page open before the server stops
TICK = 2.0       # how often a live connection is checked
PATH = "/__labkit/live"


class Pages:
    """How many pages are connected, and whether the server has been idle long enough."""

    def __init__(self, grace=None, busy=None, clock=time.monotonic):
        self.grace = GRACE if grace is None else grace
        self.clock = clock
        self.work = busy or (lambda: False)
        self.open = 0                        # connected pages
        self.inflight = 0                    # ordinary requests under way
        self.live_at = None                  # last moment a page was open; None = never armed
        self.lock = threading.Lock()

    def _add(self, field, n):
        with self.lock:
            setattr(self, field, getattr(self, field) + n)
            if field == "open":
                self.live_at = self.clock()

    def connect(self):
        self._add("open", 1)

    def disconnect(self):
        self._add("open", -1)

    def enter(self):
        self._add("inflight", 1)

    def leave(self):
        self._add("inflight", -1)

    def idle(self):
        """True once no page has been open for the whole grace period."""
        with self.lock:
            if self.open:
                self.live_at = self.clock()
            if self.live_at is None or self.open or self.inflight:
                return False
            if self.clock() - self.live_at < self.grace:
                return False
        return not self.work()


def keep_alive(argv=None, env=None) -> bool:
    """Whether this server was started to stay up (see the module docstring)."""
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env
    return (env.get("LABKIT_KEEP_ALIVE", "") not in ("", "0")
            or any(a == "--keep-alive" or a.split("=")[0] in ("--state", "--home")
                   for a in argv))


def _watch(pages, poll=1.0):
    while not pages.idle():
        time.sleep(poll)
    print("labkit: the last browser window was closed -- stopping the server.",
          file=sys.stderr, flush=True)
    # Ctrl-C, so the server unwinds and atexit runs; os._exit only if that hangs.
    threading.Thread(target=lambda: (time.sleep(10), os._exit(0)), daemon=True).start()
    os.kill(os.getpid(), signal.SIGINT)


def _start(busy):
    if keep_alive():
        return None
    pages = Pages(busy=busy)
    # A server started with `&` (the .command launchers) inherits SIGINT
    # ignored, and the Ctrl-C _watch sends would then do nothing.
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except ValueError:                   # not the main thread: os._exit will do
            pass
    threading.Thread(target=_watch, args=(pages,), daemon=True, name="labkit-lifetime").start()
    return pages


HEADERS = [("Content-Type", "text/event-stream"), ("Cache-Control", "no-store")]


def _closed(sock, wait):
    """Wait up to *wait* seconds; True if the page hung up meanwhile."""
    if sock is None:
        time.sleep(wait)                     # no socket to watch: the next write will fail
        return False
    try:
        if select.select([sock], [], [], wait)[0]:
            if sock.recv(1, socket.MSG_PEEK) == b"":
                return True
            time.sleep(wait)                 # it sent something; nothing an EventSource does
    except (OSError, ValueError):
        return True
    return False


def _stream(pages, sock=None):
    pages.connect()
    try:
        yield b"retry: 2000\n\n"
        while not _closed(sock, TICK):
            yield b": live\n\n"
    finally:
        pages.disconnect()


def wsgi(inner, pages):
    """*inner* with the live connection answered and its requests counted."""
    def app(environ, start_response):
        if environ.get("PATH_INFO") == PATH:
            start_response("200 OK", HEADERS)
            return _stream(pages, environ.get("werkzeug.socket"))
        pages.enter()
        try:
            return inner(environ, start_response)
        finally:
            pages.leave()
    return app


def flask(app, busy=None):
    """Wire a Flask app up (call it before ``app.run``)."""
    pages = _start(busy)
    if pages:
        app.wsgi_app = wsgi(app.wsgi_app, pages)
    return pages


def asgi(app, busy=None, pages=None):
    """*app* (FastAPI or any ASGI app) wired up; serve what this returns."""
    pages = pages or _start(busy)
    if not pages:
        return app

    async def wrapped(scope, receive, send):
        if scope["type"] != "http":
            return await app(scope, receive, send)
        if scope["path"] == PATH:
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(k.lower().encode(), v.encode()) for k, v in HEADERS]})
            await send({"type": "http.response.body", "body": b"retry: 2000\n\n", "more_body": True})
            pages.connect()
            try:                             # the server says when the page has gone
                while (await receive())["type"] != "http.disconnect":
                    pass
            finally:
                pages.disconnect()
            return
        pages.enter()
        try:
            await app(scope, receive, send)
        finally:
            pages.leave()
    return wrapped
