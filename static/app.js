/* ArtStitch -- the browser half.
 *
 * Seven pages, one per step.  The server holds the open set and everything decided about it; this file
 * holds only what the screen is showing, so a reload never leaves the two disagreeing.  The picture
 * itself is never held here: `Viewer` asks the server for the window it is looking at and draws that,
 * so a 300-megapixel result costs the browser what a screenful costs.
 *
 * Everything shared -- the file dialog, the path boxes, the palette -- comes from labkit.js.
 */
(function () {
  'use strict';

  const el = LK.el;
  const $ = (id) => document.getElementById(id);
  const req = (url, method, body) => LK.api(url, { method: method || 'GET', body: body });
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const int = (n) => Math.round(n).toLocaleString('en-US');
  const megabytes = (b) => (b / 1e6).toFixed(b < 1e7 ? 2 : 1) + ' MB';       // decimal, like Finder and the 10 MB cap
  const base = (p) => String(p).split(/[\\/]/).pop();
  const home = (p) => String(p).replace(/^\/Users\/[^/]+/, '~');
  const remember = (k, v) => { try { localStorage.setItem('artstitch.' + k, v); } catch (e) { /* private mode */ } };
  const recall = (k) => { try { return localStorage.getItem('artstitch.' + k) || ''; } catch (e) { return ''; } };

  const LV_ID = [0, 1, 1, 0, 1];                    // levels that change nothing: black, gamma, white, out black, out white
  const S = {
    consts: null,
    sess: { open: false },                          // /api/session: the open set, as the server sees it
    epoch: 0,                                       // bumps whenever a different Session object is opened
    sets: [],                                       // sets found on the Scans page
    presets: [], preset: null, presetIx: 0,         // export presets: the saved list, and the working copy
    colours: {},                                    // saved color settings, by name
    adj: {}, adjVer: 0, ch: 'rgb',                  // the color settings being edited
    hist: null, histKey: '', curveY: null, pickKind: null,
    level: false, crop: null,                       // straighten tools
    patches: [], shown: null, brush: 24, space: false,   // patch page: the patches in result px, the one whose source shows
    dust: { items: null, key: null, shown: [], out: new Set(), hover: null, min: 12, hmin: 12, thumb: 48, cols: 32 },   // specks found
    batch: { sets: [], status: [], job: null, root: '', running: false, open: new Set() },
    page: 'scans',
  };
  const views = {};                                 // stitch / plane / result -> Viewer

  // ============================================================== small helpers

  let toastTimer = null;
  function toast(message, kind) {
    const node = $('toast');
    node.textContent = message;
    node.className = 'status ' + (kind || 'ok');
    clearTimeout(toastTimer);
    if (kind !== 'err') toastTimer = setTimeout(() => node.classList.add('hidden'), 5000);   // errors stay put
  }
  function fail(err) { toast(String(err && err.message ? err.message : err), 'err'); }
  let busyCount = 0;
  function busy(on) { busyCount += on ? 1 : -1; $('busy').classList.toggle('hidden', busyCount <= 0); }

  async function guard(fn) {
    busy(true);
    try { return await fn(); } catch (err) { fail(err); } finally { busy(false); }
  }

  function put(hostId, node) {
    const host = $(hostId);
    host.textContent = '';
    if (node) host.appendChild(node);
    return host;
  }

  function options(select, items, value) {
    select.textContent = '';
    items.forEach((it) => {
      const o = typeof it === 'string' ? { value: it, label: it } : it;
      select.appendChild(el('option', { value: o.value, text: o.label, title: o.help || '' }));
    });
    if (value !== undefined && value !== null) select.value = value;
  }

  /** A plain table.  `cols` is [{key, label, num, fmt(row)}]. */
  function table(rows, cols) {
    const head = el('tr', {}, cols.map((c) => el('th', { class: c.num ? 'num' : '', text: c.label })));
    const body = rows.map((row) => el('tr', {}, cols.map((c) => {
      const v = c.fmt ? c.fmt(row) : row[c.key];
      const cls = [c.num ? 'num' : '', row['_' + c.key] || ''].join(' ').trim();
      return el('td', { class: cls, text: v === null || v === undefined || v === '' ? '—' : String(v) });
    })));
    return el('table', { class: 'grid' }, [el('thead', {}, [head]), el('tbody', {}, body)]);
  }

  /** Poll a job and draw a progress bar into *hostId*.  Resolves with its result. */
  function runJob(jobId, hostId, label) {
    const host = $(hostId);
    host.textContent = '';
    const fill = el('div', { class: 'bar' });
    const line = el('span', { text: (label || 'working') + '…' });
    host.appendChild(el('div', { class: 'progress' }, [fill, line]));
    return new Promise((resolve, reject) => {
      const tick = async () => {
        let job;
        try { job = await req('/api/job/' + jobId); } catch (err) { return reject(err); }
        fill.style.width = Math.round((job.progress || 0) * 100) + '%';
        line.textContent = job.message || '';
        if (!job.done) return setTimeout(tick, 350);
        if (job.error) { host.textContent = ''; return reject(new Error(job.error)); }
        line.textContent = (label || 'done') + ' — ' + job.seconds + ' s';
        resolve(job.result);
      };
      tick();
    });
  }

  /** A button that needs two clicks: the first arms it (for three seconds), the second does it. */
  function twoClick(btn, fn) {
    const text = btn.textContent;
    let timer = null;
    btn.onclick = () => {
      if (timer) { clearTimeout(timer); timer = null; btn.textContent = text; return fn(); }
      btn.textContent = 'Really?';
      timer = setTimeout(() => { timer = null; btn.textContent = text; }, 3000);
    };
  }

  // ================================================================== navigation

  const PAGES = ['scans', 'stitch', 'straighten', 'patch', 'colour', 'export', 'batch'];
  function showPage(name) {
    S.page = name;
    PAGES.forEach((p) => $('page-' + p).classList.toggle('active', p === name));
    LK.$$('#rail button').forEach((b) => b.classList.toggle('on', b.dataset.page === name));
    if (enter[name]) enter[name]();
  }
  const enter = {
    stitch() { syncView(views.stitch, false); },
    straighten() { syncView(views.plane, false); renderGeom(); },
    patch() { S.dust.shown = []; syncView(views.patch, false).then(loadPatches).then(loadDust); },
    colour() { syncView(views.result, false); loadHist(); },
    export() { planUpdate(); },
  };

  // ==================================================================== the viewer

  /* One canvas that shows a window of a very large image.  Positions are in *view pixels* -- the
     pixels of the full-size picture the server would save -- and the zoom `z` is device pixels per
     view pixel, so 100 % is one screen pixel per picture pixel.  The server draws whatever window is
     asked for at the scale asked for (never above 1:1); this class only places it.  While the
     next window is on its way the last one stays on screen, moved and scaled to where it now
     belongs, so panning and zooming are immediate and sharpen a moment later. */
  class Viewer {
    constructor(host, opts) {
      this.host = host;
      this.opts = opts;                             // {mode, params(), readout, after()}
      this.empty = host.querySelector('.empty');
      this.cv = el('canvas');
      this.ov = el('canvas');
      this.spin = el('div', { class: 'loading hidden' });
      host.append(this.cv, this.ov, this.spin);
      this.info = null; this.ver = null; this.epoch = null; this.tile = null;
      this.z = 1; this.zfit = 1; this.cx = 0; this.cy = 0; this.W = 0; this.H = 0; this.dpr = 1;
      this.gen = 0; this.inflight = false; this.again = false; this.timer = null; this.needFit = false;
      this.tool = null; this.drag = null;
      new ResizeObserver(() => this.resize()).observe(host);
      const ov = this.ov;
      ov.addEventListener('wheel', (e) => this.wheel(e), { passive: false });
      ov.addEventListener('pointerdown', (e) => this.down(e));
      ov.addEventListener('pointermove', (e) => this.move(e));
      ov.addEventListener('pointerup', (e) => this.up(e));
      ov.addEventListener('pointercancel', (e) => this.up(e));
      ov.addEventListener('dblclick', () => { if (this.info) { this.fit(); this.changed(); } });
    }

    // -- geometry
    px(n) { return n * this.dpr; }
    toScreen(x, y) { return [(x - this.cx) * this.z + this.W / 2, (y - this.cy) * this.z + this.H / 2]; }
    point(e) {
      const r = this.ov.getBoundingClientRect();
      const sx = (e.clientX - r.left) * (this.W / r.width), sy = (e.clientY - r.top) * (this.H / r.height);
      return { sx, sy, x: this.cx + (sx - this.W / 2) / this.z, y: this.cy + (sy - this.H / 2) / this.z };
    }
    fit() {
      const [w, h] = this.info.size;
      this.zfit = Math.min(this.W / w, this.H / h) * 0.97;
      this.z = this.zfit; this.cx = w / 2; this.cy = h / 2;
    }
    limit() {
      const [w, h] = this.info.size;
      this.cx = clamp(this.cx, 0, w); this.cy = clamp(this.cy, 0, h);
    }
    zoomAt(f, sx, sy) {
      const vx = this.cx + (sx - this.W / 2) / this.z, vy = this.cy + (sy - this.H / 2) / this.z;
      this.z = clamp(this.z * f, this.zfit * 0.4, 8);
      this.cx = vx - (sx - this.W / 2) / this.z; this.cy = vy - (sy - this.H / 2) / this.z;
      this.limit(); this.changed();
    }
    zoomOne() { if (this.info) this.zoomAt(1 / this.z, this.W / 2, this.H / 2); }
    zoomFit() { if (this.info) { this.fit(); this.changed(); } }

    // -- state
    resize() {
      const r = this.host.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return;      // a page that is not showing
      const dpr = window.devicePixelRatio || 1;
      const W = Math.round(r.width * dpr), H = Math.round(r.height * dpr);
      if (W === this.W && H === this.H) return;
      this.W = this.cv.width = this.ov.width = W;
      this.H = this.cv.height = this.ov.height = H;
      this.dpr = dpr;
      if (this.info && this.needFit) { this.fit(); this.needFit = false; }
      this.draw(); this.schedule();
    }
    setInfo(info, keep, message) {
      const old = this.info;
      this.info = info; this.gen += 1;
      this.empty.hidden = !!info;
      if (!info) {
        this.empty.textContent = message || ''; this.tile = null; this.draw();
        if (this.opts.readout) $(this.opts.readout).textContent = '';      // not the last set's size
        return;
      }
      this.resize();
      if (keep && old && this.W) {                  // the same place in the picture, though its origin moved
        this.cx += old.origin[0] - info.origin[0]; this.cy += old.origin[1] - info.origin[1];
        this.zfit = Math.min(this.W / info.size[0], this.H / info.size[1]) * 0.97;
      } else if (this.W) this.fit(); else this.needFit = true;
      if (!keep || !old || old.ver !== info.ver) this.tile = null;
      this.draw(); this.schedule();
    }
    invalidate() { this.gen += 1; this.schedule(); }   // the same picture, drawn differently
    changed() { this.draw(); this.schedule(); }

    // -- drawing
    draw() {
      const c = this.cv.getContext('2d');
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, this.W, this.H);
      if (this.info) {
        const t = this.tile;
        if (t) {
          const k = this.z / t.s, exact = Math.abs(k - 1) < 1e-6;
          let [dx, dy] = this.toScreen(t.x0, t.y0);
          if (exact) { dx = Math.round(dx); dy = Math.round(dy); }
          c.imageSmoothingEnabled = k < 1;          // magnified pixels stay square
          c.drawImage(t.img, dx, dy, t.img.naturalWidth * k, t.img.naturalHeight * k);
        }
        const [ax, ay] = this.toScreen(0, 0), [bx, by] = this.toScreen(this.info.size[0], this.info.size[1]);
        c.strokeStyle = css('--edge'); c.lineWidth = 1; c.strokeRect(ax + .5, ay + .5, bx - ax, by - ay);
      }
      this.drawOverlay();
      if (this.opts.readout && this.info) {
        $(this.opts.readout).textContent = (this.z * 100 < 10 ? (this.z * 100).toFixed(1) : Math.round(this.z * 100)) +
          ' %  ·  ' + int(this.info.size[0]) + ' × ' + int(this.info.size[1]) + ' px';
      }
    }
    drawOverlay() {
      const c = this.ov.getContext('2d');
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, this.W, this.H);
      if (this.info && this.tool && this.tool.draw) this.tool.draw(c, this);
    }

    // -- fetching
    schedule() {
      if (this.timer) return;
      this.timer = setTimeout(() => { this.timer = null; this.fetch(); }, 80);
    }
    fetch() {
      if (!this.info || !this.W) return;
      if (this.inflight) { this.again = true; return; }
      const [vw, vh] = this.info.size, z = this.z, s = Math.min(1, z), pad = 0.25;
      const hw = this.W / 2 / z * (1 + pad), hh = this.H / 2 / z * (1 + pad);
      const x0 = Math.max(0, Math.floor(this.cx - hw)), y0 = Math.max(0, Math.floor(this.cy - hh));
      const x1 = Math.min(vw, Math.ceil(this.cx + hw)), y1 = Math.min(vh, Math.ceil(this.cy + hh));
      if (x1 - x0 < 1 || y1 - y0 < 1) return;
      const q = new URLSearchParams({ mode: this.opts.mode, x0, y0, w: x1 - x0, h: y1 - y0, s: s.toPrecision(6),
        v: S.sess.ver, a: S.adjVer, e: S.epoch });
      const extra = this.opts.params ? this.opts.params() : {};
      Object.keys(extra).forEach((k) => extra[k] && q.set(k, extra[k]));
      const gen = this.gen, img = new Image();
      this.inflight = true; this.spin.classList.remove('hidden');
      const done = (ok) => {
        this.inflight = false;
        if (ok && gen === this.gen) { this.tile = { img, x0, y0, s }; this.draw(); }
        if (this.again || gen !== this.gen) { this.again = false; this.schedule(); } else this.spin.classList.add('hidden');
      };
      img.onload = () => done(true);
      img.onerror = () => { done(false); toast('The picture could not be drawn.', 'err'); };
      img.src = '/api/view.png?' + q;
    }

    // -- pointer
    wheel(e) {
      e.preventDefault();
      if (!this.info) return;
      if (!(e.ctrlKey || e.metaKey)) {      // two-finger pan; pinch/ctrl zooms
        const r = this.ov.getBoundingClientRect(), k = this.W / r.width;
        this.cx += e.deltaX * k / this.z; this.cy += e.deltaY * k / this.z;
        this.limit(); this.changed();
        return;
      }
      const p = this.point(e);
      this.zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.012 : 0.0016)), p.sx, p.sy);
    }
    down(e) {
      if (!this.info) return;
      const p = this.point(e);
      if (e.button === 0 && this.tool && this.tool.down && this.tool.down(p, e)) this.drag = 'tool';
      else if (e.button === 0 || e.button === 1) {
        this.drag = { sx: p.sx, sy: p.sy, cx: this.cx, cy: this.cy };
        this.ov.style.cursor = 'grabbing';
      } else return;
      this.ov.setPointerCapture(e.pointerId);
    }
    move(e) {
      if (!this.info) return;
      const p = this.point(e);
      if (this.drag === 'tool') { if (this.tool.move) this.tool.move(p, e); return; }
      if (this.drag) {
        this.cx = this.drag.cx - (p.sx - this.drag.sx) / this.z;
        this.cy = this.drag.cy - (p.sy - this.drag.sy) / this.z;
        this.limit(); this.changed(); return;
      }
      this.ov.style.cursor = (this.tool && this.tool.cursor && this.tool.cursor(p)) || 'grab';
    }
    up(e) {
      if (this.drag === 'tool' && this.tool.up) this.tool.up(this.point(e), e);
      this.drag = null;
      this.ov.style.cursor = 'grab';
      try { this.ov.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
    }
  }

  /** Bring a viewer up to date with the open set (a no-op when nothing changed since it was drawn). */
  async function syncView(v, keep) {
    const s = S.sess;
    if (!s.open) return v.setInfo(null, false, 'Open a set of scans on the first page.');
    if (!s.aligned || !s.seamed) return v.setInfo(null, false, 'Stitch the set first.');
    if (v.info && v.ver === s.ver && v.epoch === S.epoch) { v.resize(); return; }
    try {
      const info = await req('/api/view/info?mode=' + v.opts.mode);
      v.ver = s.ver; v.epoch = S.epoch;
      v.setInfo(info, keep && v.info !== null, '');
      if (v.opts.after) v.opts.after();
    } catch (err) { fail(err); }
  }

  // ==================================================================== 1. scans

  let scanPick = null;
  let batchPick = null;

  function fileRow(f, onchange) {
    const box = el('input', { type: 'checkbox', onchange: () => { f.include = box.checked; onchange(); } });
    box.checked = !!f.include;
    const meta = f.error ? f.error : [f.w + ' × ' + f.h, f.dpi ? Math.round(f.dpi[0]) + ' dpi' : '', megabytes(f.bytes)].filter(Boolean).join('  ·  ');
    return el('label', { class: 'file' }, [box, el('span', { class: 'name', text: f.name, title: f.path }),
      f.suspect ? el('span', { class: 'badge warn', title: 'Much larger than the other scans of this name: probably an earlier result',
        text: 'an earlier result?' }) : null,
      f.error ? el('span', { class: 'badge bad', text: 'unreadable' }) : null,
      el('span', { class: 'meta', text: meta })]);
  }

  function setCard(set) {
    const card = el('div', { class: 'panel setcard' });
    const openBtn = el('button', { class: 'primary', text: 'Open', onclick: () => {
      const files = set.files.filter((f) => f.include);
      if (!files.length) return toast('Check at least one scan.', 'warn');
      openPaths(files.map((f) => f.path), set.name);
    } });
    const count = el('span', { class: 'badge' });
    const paint = () => {
      const n = set.files.filter((f) => f.include).length;
      count.textContent = n + ' of ' + set.files.length + ' scans · ' + set.mp + ' MP';
      card.classList.toggle('off', n === 0);
    };
    card.appendChild(el('div', { class: 'head' }, [el('b', { text: set.name }),
      el('span', { class: 'dim mono', text: home(set.folder) }), count, el('span', { class: 'spacer' }), openBtn]));
    card.appendChild(el('div', { class: 'files' }, set.files.map((f) => fileRow(f, paint))));
    paint();
    return card;
  }

  function renderSets() {
    const host = put('setList', null);
    if (!S.sets.length) return;
    host.appendChild(el('p', { class: 'hint', text: S.sets.length + (S.sets.length === 1 ? ' set' : ' sets') +
      ' found. Open one to stitch it; to do them all at once use Batch.' }));
    S.sets.forEach((set) => host.appendChild(setCard(set)));
  }

  function openedSession(res) {
    S.epoch += 1;
    S.adj = clone(res.adj || {});
    S.adjVer += 1;
    S.hist = null; S.histKey = ''; S.curveY = null; S.level = false; S.pickKind = null; S.crop = null;
    S.patches = []; S.shown = null; S.dust.shown = [];
    Object.values(views).forEach((v) => { v.info = null; v.tile = null; });
    updateSession(res);
    syncColourUI();
  }

  /** Take a fresh summary from the server and redraw everything that is text. */
  function updateSession(res) {
    S.sess = res;
    $('setLabel').textContent = res.open ? res.name + '  ·  ' + res.paths.length + (res.paths.length === 1 ? ' scan' : ' scans') : 'no set open';
    if (!res.open) return;
    $('setName').value = res.name;
    $('maxDeg').value = res.max_deg;
    $('dpiIn').value = res.dpi || '';
    renderStitchSide(); renderGeom(); syncCropFields(); planUpdate();
  }

  function openPaths(paths, name) {
    return guard(async () => {
      openedSession(await req('/api/open', 'POST', { paths, name }));
      showPage('stitch');
    });
  }

  function buildScans() {
    scanPick = LK.pathPicker($('scanPick'), { mode: 'dir', value: recall('folder') || LK.recentFolders()[0] || '',
      placeholder: '/folder/with/the/scans' });
    $('btnScan').onclick = () => guard(async () => {
      const folder = scanPick.value.trim();
      if (!folder) return toast('Give a folder first.', 'warn');
      remember('folder', folder);
      const r = await req('/api/scan', 'POST', { folder, recursive: $('scanRecursive').checked, all_images: $('scanAll').checked });
      S.sets = r.sets;
      renderSets();
      if (!r.sets.length) toast('No scans found in that folder.', 'warn');
    });
    $('btnScanFiles').onclick = async () => {
      const paths = await LK.browse({ mode: 'files', path: scanPick.value, title: 'Choose the scans of one artwork',
        exts: $('scanAll').checked ? S.consts.extensions.image : S.consts.extensions.tiff });
      if (paths && paths.length) openPaths(paths);
    };
    $('btnLoadSession').onclick = async () => {
      const path = await LK.browse({ mode: 'file', exts: ['.json'], path: scanPick.value, title: 'Open a saved session' });
      if (path) guard(async () => { openedSession(await req('/api/session/load', 'POST', { path })); showPage('stitch'); });
    };
  }

  // =================================================================== 2. stitch

  function renderStitchSide() {
    const s = S.sess, qc = put('qcList', null), scans = put('poseTable', null);
    if (!s.open || !s.aligned) { qc.appendChild(el('p', { class: 'dim', text: 'Nothing yet.' })); return; }
    if (!s.qc.length) qc.appendChild(el('div', { class: 'qc-item ok', text: 'No problems found.' }));
    s.qc.forEach((q) => qc.appendChild(el('div', { class: 'qc-item ' + q.level, text: q.msg })));
    const res = (s.report || {}).residual || {};
    const worst = Object.values(res).reduce((m, v) => Math.max(m, v.rms), 0);
    if (worst) qc.appendChild(el('p', { class: 'hint', text: 'Neighboring scans agree to ' + worst.toFixed(2) + ' px (RMS, worst pair) once matched.' }));
    const rows = s.tiles.map((t, k) => ({ n: k + 1, scan: t.name, size: int(t.w) + ' × ' + int(t.h),
      turned: s.poses[k] ? s.poses[k].theta.toFixed(2) + '°' : '', fixed: s.poses[k] && s.poses[k].anchor ? 'reference' : '' }));
    scans.appendChild(table(rows, [{ key: 'n', label: '#', num: true }, { key: 'scan', label: 'scan' },
      { key: 'turned', label: 'turned', num: true }, { key: 'fixed', label: '' }]));
  }

  async function saveSettings() {
    const dpi = parseFloat($('dpiIn').value);
    updateSession(await req('/api/session/settings', 'POST',
      { name: $('setName').value, dpi: Number.isFinite(dpi) ? dpi : 0, max_deg: parseFloat($('maxDeg').value) || 0 }));
  }

  function doStitch(poses) {
    if (!S.sess.open) return toast('Open a set of scans on the first page.', 'warn');
    return guard(async () => {
      await saveSettings();
      const had = S.sess.patched || 0;
      const { job } = await req('/api/align', 'POST', poses ? { poses } : {});
      S.epoch += 1;
      updateSession(await runJob(job, 'stitchProgress', 'stitched'));
      S.patches = []; S.shown = null; S.dust.shown = [];
      Object.values(views).forEach((v) => { v.info = null; });
      await syncView(views.stitch, false);
      toast((S.sess.qc.length ? 'Stitched — see the checks on the right.' : 'Stitched.') +
        (had ? (had === 1 ? ' The patch was' : ' The ' + had + ' patches were') + ' removed: they were made on the old stitch.' : ''),
        S.sess.qc.length || had ? 'warn' : 'ok');
    });
  }

  // ------------------------------------------------------ placing scans by hand

  function arrange() {
    const s = S.sess;
    if (!s.open) return toast('Open a set of scans on the first page.', 'warn');
    const T = s.tiles, n = T.length;
    const row = () => { let x = 0; return T.map((t, k) => { const p = { x: x + t.w / 2, y: t.h / 2, theta: 0, anchor: k === 0 }; x += t.w + 600; return p; }); };
    const P = s.poses && s.poses.length === n ? clone(s.poses) : row();
    let sel = 0, z = 0.01, cx = 0, cy = 0, drag = null;

    const cv = el('canvas');
    const draw = () => {
      const c = cv.getContext('2d'), W = cv.width, H = cv.height;
      c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, W, H);
      const dpr = W / cv.getBoundingClientRect().width;
      [...P.keys()].filter((k) => k !== sel).concat([sel]).forEach((k) => {
        const t = T[k], p = P[k];
        c.save();
        c.translate((p.x - cx) * z + W / 2, (p.y - cy) * z + H / 2);
        c.rotate(p.theta * Math.PI / 180);
        if (imgs[k].complete && imgs[k].naturalWidth) { c.globalAlpha = 0.72; c.drawImage(imgs[k], -t.w * z / 2, -t.h * z / 2, t.w * z, t.h * z); c.globalAlpha = 1; }
        c.lineWidth = (k === sel ? 3 : 1.5) * dpr;
        c.strokeStyle = k === sel ? css('--accent') : css('--sel');
        c.strokeRect(-t.w * z / 2, -t.h * z / 2, t.w * z, t.h * z);
        const label = (k + 1) + '  ' + t.name, lx = -t.w * z / 2 + 8 * dpr, ly = -t.h * z / 2 + 22 * dpr;
        c.font = (16 * dpr) + 'px sans-serif';
        c.lineWidth = 4 * dpr; c.lineJoin = 'round'; c.strokeStyle = 'rgba(0,0,0,.65)'; c.strokeText(label, lx, ly);   // readable on any picture
        c.fillStyle = k === sel ? css('--accent') : css('--sel');
        c.fillText(label, lx, ly);
        c.restore();
      });
    };
    const imgs = T.map((t, k) => { const im = new Image(); im.onload = draw; im.src = '/api/tile/' + k + '.jpg?f=32&s=' + s.key; return im; });
    const corners = (k) => {
      const t = T[k], p = P[k], a = p.theta * Math.PI / 180, co = Math.cos(a), si = Math.sin(a);
      return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([i, j]) => { const u = i * t.w / 2, v = j * t.h / 2; return [p.x + u * co - v * si, p.y + u * si + v * co]; });
    };
    const fit = () => {
      const pts = P.flatMap((_, k) => corners(k)), xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      z = Math.min(cv.width / (x1 - x0), cv.height / (y1 - y0)) * 0.92; cx = (x0 + x1) / 2; cy = (y0 + y1) / 2;
    };
    const at = (e) => {
      const r = cv.getBoundingClientRect(), k = cv.width / r.width;
      const sx = (e.clientX - r.left) * k, sy = (e.clientY - r.top) * k;
      return { sx, sy, x: cx + (sx - cv.width / 2) / z, y: cy + (sy - cv.height / 2) / z };
    };
    const hit = (m) => [sel, ...[...P.keys()].reverse().filter((k) => k !== sel)].find((k) => {
      const p = P[k], a = -p.theta * Math.PI / 180, dx = m.x - p.x, dy = m.y - p.y;
      return Math.abs(dx * Math.cos(a) - dy * Math.sin(a)) <= T[k].w / 2 && Math.abs(dx * Math.sin(a) + dy * Math.cos(a)) <= T[k].h / 2;
    });
    cv.addEventListener('pointerdown', (e) => {
      const m = at(e), k = hit(m);
      if (k !== undefined) { sel = k; drag = { k, last: m }; } else drag = { pan: true, sx: m.sx, sy: m.sy, cx, cy };
      cv.classList.add('moving'); cv.setPointerCapture(e.pointerId); draw();
    });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const m = at(e);
      if (drag.pan) { cx = drag.cx - (m.sx - drag.sx) / z; cy = drag.cy - (m.sy - drag.sy) / z; }
      else { P[drag.k].x += m.x - drag.last.x; P[drag.k].y += m.y - drag.last.y; drag.last = at(e); }
      draw();
    });
    cv.addEventListener('pointerup', () => { drag = null; cv.classList.remove('moving'); });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (!(e.ctrlKey || e.metaKey)) {      // two-finger pan; pinch/ctrl zooms
        const r = cv.getBoundingClientRect(), k = cv.width / r.width;
        cx += e.deltaX * k / z; cy += e.deltaY * k / z; draw();
        return;
      }
      const m = at(e), f = Math.exp(-e.deltaY * 0.0016);
      z = clamp(z * f, 0.0005, 0.2); cx = m.x - (m.sx - cv.width / 2) / z; cy = m.y - (m.sy - cv.height / 2) / z; draw();
    }, { passive: false });

    const turn = (deg) => () => { P[sel].theta += deg; draw(); };
    const btn = (text, fn, title) => el('button', { class: 'small', text, title: title || '', onclick: fn });
    const close = () => { document.removeEventListener('keydown', onKey); modal.remove(); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    const modal = el('div', { class: 'modal', onclick: (e) => { if (e.target === modal) close(); } }, [
      el('div', { class: 'modal-box arrange' }, [
        el('div', { class: 'modal-head' }, [el('h2', { text: 'Place the scans' }),
          el('span', { class: 'dim', text: 'Drag a scan onto where it belongs, turn it upright, and stitch. Roughly is enough — the app finds the exact position.' }),
          el('span', { class: 'spacer' }), el('button', { class: 'ghost small', text: 'Cancel', onclick: close })]),
        el('div', { class: 'modal-body' }, [cv]),
        el('div', { class: 'modal-foot' }, [
          el('div', { class: 'row' }, [el('span', { text: 'Turn the picked scan' }), btn('↺ 90°', turn(-90)), btn('↻ 90°', turn(90)),
            btn('−1°', turn(-1)), btn('+1°', turn(1)), btn('−0.1°', turn(-0.1)), btn('+0.1°', turn(0.1))]),
          el('span', { class: 'spacer' }),
          btn('Line them up', () => { const r = row(); r.forEach((p, k) => Object.assign(P[k], p)); fit(); draw(); }, 'Start over: every scan upright, side by side'),
          el('button', { class: 'primary', text: 'Stitch from this', onclick: () => { close(); doStitch(P.map((p) => ({ x: p.x, y: p.y, theta: p.theta, anchor: !!p.anchor }))); } })]),
      ])]);
    document.addEventListener('keydown', onKey);
    document.body.appendChild(modal);
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    fit(); draw();
  }

  // ====================================================== 3. straighten & crop

  const METHODS = { edges: 'from the paper edges', lines: 'from the strongest straight lines (no paper edge found)',
    manual: 'set by hand', none: 'not straightened' };

  function renderGeom() {
    const s = S.sess, g = s.geom, info = put('geomInfo', null), edgesHost = put('edgeTable', null);
    if (!s.open || !g) { info.appendChild(el('p', { class: 'dim', text: 'Nothing yet.' })); return; }
    const [w, h] = s.result_size || [0, 0], dpi = dpiOf();
    const kv = el('dl', { class: 'kv' });
    const add = (k, v) => kv.append(el('dt', { text: k }), el('dd', { text: v }));
    add('straightened', METHODS[g.method] || g.method);
    add('angle', g.angle_deg.toFixed(2) + '°');
    if (g.quarter) add('turned', ['', '90° clockwise', '180°', '90° counterclockwise'][g.quarter]);
    if (g.method === 'edges') add('edges agree to', g.spread_deg.toFixed(2) + '°  (' + g.confidence + ')');
    add('result', int(w) + ' × ' + int(h) + ' px');
    add('', (w * h / 1e6).toFixed(0) + ' MP');
    if (dpi) add('at ' + Math.round(dpi) + ' dpi', (w / dpi * 2.54).toFixed(1) + ' × ' + (h / dpi * 2.54).toFixed(1) + ' cm');
    info.appendChild(kv);
    const ring = ['top', 'right', 'bottom', 'left'];     // the edges were found upright; a clockwise turn moves each one place on
    const rows = ['left', 'right', 'top', 'bottom'].map((side) => {
      const e = (g.sides || {})[ring[(ring.indexOf(side) + 4 - (g.quarter || 0)) % 4]];
      return e && e.found ? { side, tilt: e.tilt_deg.toFixed(2) + '°', scatter: e.rms_px.toFixed(1) + ' px', length: int(e.span_px) + ' px' }
        : { side, tilt: '', scatter: '', length: '' };
    });
    if (g.method === 'edges' || g.method === 'manual') {
      edgesHost.appendChild(table(rows, [{ key: 'side', label: 'edge' }, { key: 'tilt', label: 'tilt', num: true },
        { key: 'scatter', label: 'scatter', num: true }, { key: 'length', label: 'length', num: true }]));
    } else edgesHost.appendChild(el('p', { class: 'dim', text: 'None found.' }));
    $('angleIn').value = g.angle_deg.toFixed(2);
    $('angleRange').value = clamp(g.angle_deg, -3, 3);
    $('insetIn').value = g.inset === undefined ? 12 : g.inset;
  }

  function dpiOf() {
    const s = S.sess;
    if (s.dpi) return s.dpi;
    const t = (s.tiles || [])[0];
    return t && t.dpi ? t.dpi[0] : null;
  }

  /** The crop as numbers: pixels of the straightened sheet, from its top-left corner. */
  function syncCropFields() {
    const c = S.crop;
    ['X0', 'Y0', 'X1', 'Y1'].forEach((k, i) => { $('crop' + k).value = c ? Math.round([c.x0, c.y0, c.x1, c.y1][i]) : ''; });
  }

  function cropFromGeom() {
    const g = S.sess.geom, v = views.plane;
    if (!g || !v.info) { S.crop = null; return; }
    const [ox, oy] = v.info.origin;
    S.crop = { x0: g.crop[0] - ox, y0: g.crop[1] - oy, x1: g.crop[2] - ox, y1: g.crop[3] - oy };
    syncCropFields();
    v.drawOverlay();
  }

  async function commitCrop() {
    const c = S.crop, [ox, oy] = views.plane.info.origin;
    await guard(async () => {
      updateSession(await req('/api/crop', 'POST', { crop: [c.x0 + ox, c.y0 + oy, c.x1 + ox, c.y1 + oy] }));
      views.plane.ver = S.sess.ver;                 // the crop does not change the plane, only what is inside it
      cropFromGeom();
    });
  }

  /* The crop rectangle: drag an edge or corner to move it, drag inside to move all of it. */
  const cropTool = {
    drag: null,
    hit(p) {
      const v = views.plane, c = S.crop;
      if (!c) return null;
      const [a, b] = v.toScreen(c.x0, c.y0), [d, e] = v.toScreen(c.x1, c.y1), t = v.px(9);
      const inX = p.sx > a - t && p.sx < d + t, inY = p.sy > b - t && p.sy < e + t;
      const m = { x0: Math.abs(p.sx - a) < t && inY, x1: Math.abs(p.sx - d) < t && inY,
        y0: Math.abs(p.sy - b) < t && inX, y1: Math.abs(p.sy - e) < t && inX };
      if (m.x0 || m.x1 || m.y0 || m.y1) return m;
      return p.sx > a && p.sx < d && p.sy > b && p.sy < e ? { move: true } : null;
    },
    cursor(p) {
      const m = this.hit(p);
      if (!m) return null;
      if (m.move) return 'move';
      return (m.x0 || m.x1) && (m.y0 || m.y1) ? ((m.x0 && m.y0) || (m.x1 && m.y1) ? 'nwse-resize' : 'nesw-resize')
        : (m.x0 || m.x1) ? 'ew-resize' : 'ns-resize';
    },
    down(p) {
      const m = this.hit(p);
      if (!m) return false;
      this.drag = { m, p, c: { ...S.crop } };
      return true;
    },
    move(p) {
      const d = this.drag, [w, h] = views.plane.info.size;
      if (!d) return;
      const dx = p.x - d.p.x, dy = p.y - d.p.y, o = d.c, c = S.crop, min = 16;
      if (d.m.move) {
        const mx = clamp(dx, -o.x0, w - o.x1), my = clamp(dy, -o.y0, h - o.y1);
        Object.assign(c, { x0: o.x0 + mx, x1: o.x1 + mx, y0: o.y0 + my, y1: o.y1 + my });
      } else {
        if (d.m.x0) c.x0 = clamp(o.x0 + dx, 0, c.x1 - min);
        if (d.m.x1) c.x1 = clamp(o.x1 + dx, c.x0 + min, w);
        if (d.m.y0) c.y0 = clamp(o.y0 + dy, 0, c.y1 - min);
        if (d.m.y1) c.y1 = clamp(o.y1 + dy, c.y0 + min, h);
      }
      syncCropFields();
      views.plane.drawOverlay();
    },
    up() { if (this.drag) { this.drag = null; commitCrop(); } },
    draw(ctx, v) {
      const c = S.crop;
      if (!c) return;
      const [a, b] = v.toScreen(c.x0, c.y0), [d, e] = v.toScreen(c.x1, c.y1), W = v.W, H = v.H;
      ctx.fillStyle = 'rgba(0,0,0,.5)';
      ctx.fillRect(0, 0, W, Math.max(0, b)); ctx.fillRect(0, e, W, Math.max(0, H - e));
      ctx.fillRect(0, b, Math.max(0, a), e - b); ctx.fillRect(d, b, Math.max(0, W - d), e - b);
      ctx.strokeStyle = css('--accent'); ctx.lineWidth = v.px(1.5); ctx.strokeRect(a, b, d - a, e - b);
      const hs = v.px(4);
      ctx.fillStyle = css('--accent');
      [[a, b], [d, b], [a, e], [d, e], [(a + d) / 2, b], [(a + d) / 2, e], [a, (b + e) / 2], [d, (b + e) / 2]]
        .forEach(([x, y]) => ctx.fillRect(x - hs, y - hs, 2 * hs, 2 * hs));
    },
  };

  /* Draw a line along something that ought to be level (or upright): the sheet is turned until it is. */
  const levelTool = {
    a: null, b: null,
    cursor() { return 'crosshair'; },
    down(p) { this.a = p; this.b = p; return true; },
    move(p) { this.b = p; views.plane.drawOverlay(); },
    up(p) {
      const a = this.a; this.a = null;
      if (!a) return;
      views.plane.drawOverlay();
      straighten({ line: [[a.x, a.y], [p.x, p.y]] }, 'leveled');
      setLevel(false);
    },
    draw(ctx, v) {
      if (!this.a) return;
      const [x0, y0] = v.toScreen(this.a.x, this.a.y), [x1, y1] = v.toScreen(this.b.x, this.b.y);
      ctx.strokeStyle = '#ffd54a'; ctx.lineWidth = v.px(2);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    },
  };

  function gridTool(inner) {                        // the tool with an optional guide grid under it
    return {
      __proto__: inner,
      draw(ctx, v) {
        if ($('ovGrid').checked) {
          const span = Math.max(v.W, v.H) / v.z / 10, mag = Math.pow(10, Math.floor(Math.log10(span))),
            step = [1, 2, 5, 10].map((m) => m * mag).find((m) => m >= span) || span;
          ctx.strokeStyle = 'rgba(255,213,74,.35)'; ctx.lineWidth = 1; ctx.beginPath();
          for (let x = Math.ceil((v.cx - v.W / 2 / v.z) / step) * step; x < v.cx + v.W / 2 / v.z; x += step) {
            const [sx] = v.toScreen(x, 0); ctx.moveTo(Math.round(sx) + .5, 0); ctx.lineTo(Math.round(sx) + .5, v.H);
          }
          for (let y = Math.ceil((v.cy - v.H / 2 / v.z) / step) * step; y < v.cy + v.H / 2 / v.z; y += step) {
            const [, sy] = v.toScreen(0, y); ctx.moveTo(0, Math.round(sy) + .5); ctx.lineTo(v.W, Math.round(sy) + .5);
          }
          ctx.stroke();
        }
        if (inner.draw) inner.draw(ctx, v);
      },
    };
  }

  function setLevel(on) {
    S.level = on;
    $('btnLevel').classList.toggle('armed', on);
    views.plane.tool = gridTool(on ? levelTool : cropTool);
    views.plane.drawOverlay();
  }

  function insetValue() { return Math.max(0, parseFloat($('insetIn').value) || 0); }

  function straighten(body, label) {
    return guard(async () => {
      const { job } = await req('/api/straighten', 'POST', body);
      updateSession(await runJob(job, 'straightenProgress', label));
      await syncView(views.plane, true);
      toast('Angle ' + S.sess.geom.angle_deg.toFixed(2) + '°  ·  ' + int(S.sess.result_size[0]) + ' × ' + int(S.sess.result_size[1]) + ' px');
    });
  }

  function buildStraighten() {
    $('btnEdges').onclick = () => straighten({ inset: insetValue() }, 'edges found');
    $('btnAngle').onclick = () => {
      const a = parseFloat($('angleIn').value);
      if (!Number.isFinite(a)) return toast('Type an angle in degrees.', 'warn');
      straighten({ angle_deg: a, inset: insetValue() }, 'angle set');
    };
    $('angleIn').onkeydown = (e) => { if (e.key === 'Enter') $('btnAngle').click(); };
    $('angleRange').oninput = () => { $('angleIn').value = parseFloat($('angleRange').value).toFixed(2); };
    $('angleRange').onchange = () => $('btnAngle').click();
    $('btnLevel').onclick = () => setLevel(!S.level);
    const turn = (by) => () => S.sess.aligned && guard(async () => {
      updateSession(await req('/api/turn', 'POST', { by }));
      await syncView(views.plane, false);           // another shape: fit it again
    });
    $('btnTurnL').onclick = turn(-1);
    $('btnTurnR').onclick = turn(1);
    $('btnAutoCrop').onclick = () => S.sess.geom && straighten({ keep: true, inset: insetValue() }, 'crop found');
    $('btnCropAll').onclick = () => {
      const v = views.plane;
      if (!v.info) return;
      S.crop = { x0: 0, y0: 0, x1: v.info.size[0], y1: v.info.size[1] };
      commitCrop();
    };
    $('btnCropSet').onclick = () => {
      const v = views.plane;
      if (!v.info) return;
      const f = ['X0', 'Y0', 'X1', 'Y1'].map((k) => parseFloat($('crop' + k).value));
      if (f.some((x) => !Number.isFinite(x))) return toast('Fill in all four numbers.', 'warn');
      S.crop = { x0: f[0], y0: f[1], x1: f[2], y1: f[3] };
      commitCrop();
    };
  }

  // =============================================================== 4. patch

  /* Click on a speck, drag along a hair: the server fills the stroke with a copy of the picture from close by (a long
     stroke in stretches, each copied from a place of its own) and draws the picture patched.  This draws only the
     outlines: white round each patch, yellow round where the copies of the one under the pointer came from (drag one
     to copy that stretch from elsewhere).  Option-click takes a patch out. */

  /** Is view point p within the stroke of patch q, moved by (dx, dy)? */
  function onStroke(p, q, dx, dy, slack) {
    const pts = q.pts;
    for (let i = 0; i < pts.length; i += 1) {
      const [ax, ay] = pts[i], [bx, by] = pts[Math.min(i + 1, pts.length - 1)];
      const ux = bx - ax, uy = by - ay, L = ux * ux + uy * uy;
      const t = L ? clamp(((p.x - dx - ax) * ux + (p.y - dy - ay) * uy) / L, 0, 1) : 0;
      if (Math.hypot(p.x - dx - ax - t * ux, p.y - dy - ay - t * uy) <= q.r + slack) return true;
    }
    return false;
  }

  function strokePath(c, v, pts, dx, dy) {
    c.beginPath();
    pts.forEach(([x, y], i) => { const [sx, sy] = v.toScreen(x + dx, y + dy); if (i) c.lineTo(sx, sy); else c.moveTo(sx, sy); });
    if (pts.length === 1) { const [sx, sy] = v.toScreen(pts[0][0] + dx, pts[0][1] + dy); c.lineTo(sx + 0.01, sy); }   // a dot
  }

  let outlineCanvas = null;
  /** One outline round all of `shapes` ({pts, r, dx, dy}): the strokes drawn wide, then their insides cut out. */
  function outlines(ctx, v, shapes, colour) {
    if (!shapes.length) return;
    const off = outlineCanvas || (outlineCanvas = document.createElement('canvas'));
    if (off.width !== v.W || off.height !== v.H) { off.width = v.W; off.height = v.H; }
    const o = off.getContext('2d'), w = v.px(1.25);
    o.clearRect(0, 0, v.W, v.H);
    o.lineCap = o.lineJoin = 'round';
    [[4 * w, 'rgba(0,0,0,.55)', 'source-over'], [2 * w, colour, 'source-over'], [0, '#000', 'destination-out']].forEach(([extra, style, op]) => {
      o.globalCompositeOperation = op; o.strokeStyle = style;
      shapes.forEach((s) => { o.lineWidth = 2 * s.r * v.z + extra; strokePath(o, v, s.pts, s.dx, s.dy); o.stroke(); });
    });
    o.globalCompositeOperation = 'source-over';
    ctx.drawImage(off, 0, 0);
  }

  /** An arrow from a copy (the stretch's points moved by d) to where it went, between their outlines. */
  function arrow(ctx, v, pts, d, r) {
    const c = [0, 1].map((k) => pts.reduce((a, pt) => a + pt[k], 0) / pts.length);
    const [x0, y0] = v.toScreen(c[0] + d[0], c[1] + d[1]), [x1, y1] = v.toScreen(c[0], c[1]);
    const len = Math.hypot(x1 - x0, y1 - y0), gap = r * v.z + v.px(3), head = v.px(7);
    if (len <= 2 * gap + head) return;
    const ux = (x1 - x0) / len, uy = (y1 - y0) / len, ex = x1 - ux * gap, ey = y1 - uy * gap;
    ctx.strokeStyle = ctx.fillStyle = '#ffd54a'; ctx.lineWidth = v.px(1.5);
    ctx.beginPath(); ctx.moveTo(x0 + ux * gap, y0 + uy * gap); ctx.lineTo(ex - ux * head, ey - uy * head); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ux * head - uy * head * 0.45, ey - uy * head + ux * head * 0.45);
    ctx.lineTo(ex - ux * head + uy * head * 0.45, ey - uy * head - ux * head * 0.45); ctx.fill();
  }

  const patchTool = {
    mode: null, hover: null, pending: null,
    shown() { return S.patches.find((q) => q.id === S.shown) || S.patches[S.patches.length - 1] || null; },
    source(p) {                                     // {q, k}: stretch k of the shown patch, when p is on its copy's outline
      const q = this.shown();
      if (!q || !$('patchOutlines').checked) return null;
      const k = q.parts.findIndex((s) => s.d && onStroke(p, { pts: s.pts, r: q.r }, s.d[0], s.d[1], 3 / views.patch.z));
      return k < 0 ? null : { q, k };
    },
    under(p) {
      for (let i = S.patches.length - 1; i >= 0; i -= 1) if (onStroke(p, S.patches[i], 0, 0, 0)) return S.patches[i];
      return null;
    },
    cursor(p) {
      this.hover = p;
      const src = this.source(p), q = !src && this.under(p);
      if (q) S.shown = q.id;                        // a patch shows where its copy came from until another is hovered
      views.patch.drawOverlay();
      return S.space ? 'grab' : src ? 'move' : 'crosshair';
    },
    down(p, e) {
      if (S.space) return false;                    // the space bar held: move the picture
      if (e.altKey) {
        const q = this.under(p);
        if (q) patchOp({ op: 'remove', id: q.id });
        return true;
      }
      const hit = this.source(p);
      this.mode = hit ? { ...hit, p, d: hit.q.parts[hit.k].d.slice() } : { pts: [[p.x, p.y]] };
      views.patch.drawOverlay();
      return true;
    },
    move(p) {
      const m = this.mode;
      this.hover = p;
      if (!m) return;
      if (m.q) m.q.parts[m.k].d = [Math.round(m.d[0] + p.x - m.p.x), Math.round(m.d[1] + p.y - m.p.y)];
      else {
        const [lx, ly] = m.pts[m.pts.length - 1];
        if (Math.hypot(p.x - lx, p.y - ly) >= Math.max(0.5, S.brush / 8)) m.pts.push([p.x, p.y]);
      }
      views.patch.drawOverlay();
    },
    up() {
      const m = this.mode;
      this.mode = null;
      if (!m) return;
      if (!m.q) { this.pending = m.pts; patchOp({ op: 'add', pts: m.pts, r: S.brush / 2 }); return; }
      const d = m.q.parts[m.k].d;
      if (d[0] !== m.d[0] || d[1] !== m.d[1]) patchOp({ op: 'source', id: m.q.id, part: m.k, d });
    },
    draw(ctx, v) {
      const D = S.dust;
      if ($('patchOutlines').checked && D.shown.length) {
        const dot = (it) => ({ pts: it.pts, r: it.r, dx: 0, dy: 0 });
        outlines(ctx, v, D.shown.filter((it) => !D.out.has(it.i)).map(dot), '#3ddcff');
        if (D.hover) outlines(ctx, v, [dot(D.hover)], '#ff9d2e');
      }
      if ($('patchOutlines').checked && S.patches.length) {
        const shape = (pts, r, d) => ({ pts, r, dx: d ? d[0] : 0, dy: d ? d[1] : 0 });
        outlines(ctx, v, S.patches.map((q) => shape(q.pts, q.r)), '#fff');
        outlines(ctx, v, S.patches.flatMap((q) => q.parts.filter((s) => !s.d).map((s) => shape(s.pts, q.r))), '#ff5a5a');   // nowhere to copy from
        const q = this.shown();
        if (q) {
          outlines(ctx, v, q.parts.filter((s) => s.d).map((s) => shape(s.pts, q.r, s.d)), '#ffd54a');
          const same = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];
          for (let i = 0, j; i < q.parts.length; i = j) {   // one arrow for stretches in a row copied from one place
            for (j = i + 1; j < q.parts.length && same(q.parts[j].d, q.parts[i].d);) j += 1;
            if (q.parts[i].d) arrow(ctx, v, q.parts.slice(i, j).flatMap((s) => s.pts), q.parts[i].d, q.r);
          }
        }
      }
      const pts = (this.mode && this.mode.pts) || this.pending;   // the stroke being painted, until its patch is drawn
      if (pts) {
        ctx.strokeStyle = 'rgba(255,64,64,.45)'; ctx.lineWidth = Math.max(S.brush * v.z, 2);
        ctx.lineCap = ctx.lineJoin = 'round';
        strokePath(ctx, v, pts, 0, 0); ctx.stroke();
      }
      if (this.hover && !S.space && !(this.mode && this.mode.q)) {     // the brush
        const [x, y] = v.toScreen(this.hover.x, this.hover.y), rad = Math.max(S.brush / 2 * v.z, v.px(2));
        ctx.lineWidth = v.px(1);
        ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.beginPath(); ctx.arc(x, y, rad + v.px(1), 0, 2 * Math.PI); ctx.stroke();
        ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI); ctx.stroke();
      }
    },
  };

  function renderPatchSide() {
    const n = S.patches.length, lost = S.patches.filter((q) => q.parts.some((s) => !s.d)).length, host = put('patchInfo', null);
    host.appendChild(el('p', { class: n ? '' : 'dim', text: n ? n + (n === 1 ? ' patch' : ' patches') : 'None yet.' }));
    if (lost) host.appendChild(el('p', { class: 'hint warn', text: (lost === 1 ? 'One (red) found' : lost + ' (red) found') + ' nowhere to copy from: too near the edge of the picture?' }));
    $('btnPatchUndo').disabled = !S.sess.undo;
    renderDust();
  }

  async function loadPatches() {
    S.patches = [];
    if (views.patch.info) {
      try { S.patches = (await req('/api/patches')).patches; } catch (err) { fail(err); }
    }
    renderPatchSide();
    views.patch.drawOverlay();
  }

  /* The server looks for hairs and specks standing out from the grain round them (dust.py) and hands them back hairs
     first, each strongest first, each with the stroke that would patch it and its place in a sheet of pictures.  Shown
     are those from the slider's score not yet under a patch; those not left out are circled and patched in one go
     (one step to undo). */

  const centre = (pts) => [0, 1].map((k) => pts.reduce((a, pt) => a + pt[k], 0) / pts.length);

  function renderDust() {
    const D = S.dust, grid = put('dustGrid', null);
    $('dustMinOut').textContent = D.min;
    $('dustHairOut').textContent = D.hmin;
    $('dustMinRow').hidden = !D.items;
    const from = (D.items || []).filter((it) => it.score >= (it.shape === 'hair' ? D.hmin : D.min));
    D.shown = from.filter((it) => {                 // a point of its own stroke: a curled hair's center is off it
      const [x, y] = it.pts[it.pts.length >> 1];
      return !S.patches.some((q) => onStroke({ x, y }, q, 0, 0, 0));
    });
    D.below = (D.items || []).length - from.length; D.done = from.length - D.shown.length;
    $('dustPick').hidden = !D.shown.length;
    dustCount();
    if (!D.items) return;
    const url = 'url("/api/dust/sheet.jpg?' + new URLSearchParams({ k: D.key, a: S.adjVer }) + '")';
    const rows = Math.ceil(D.items.length / D.cols), at = (k, n) => (n > 1 ? k / (n - 1) * 100 : 0) + '%';
    D.shown.forEach((it) => {                       // a sprite: the sheet as wide as `cols` tiles, placed in %
      const t = el('div', { class: D.out.has(it.i) ? 'out' : '', title: it.shape + ' · ' + it.kind + ' · ' + it.score + '× the grain' });
      Object.assign(t.style, { backgroundImage: url, backgroundSize: D.cols * 100 + '% auto',
        backgroundPosition: at(it.i % D.cols, D.cols) + ' ' + at(Math.floor(it.i / D.cols), rows) });
      t.onclick = () => {                           // a double-click toggles twice: no change, and goes there
        if (D.out.has(it.i)) D.out.delete(it.i); else D.out.add(it.i);
        t.classList.toggle('out', D.out.has(it.i));
        dustCount(); views.patch.drawOverlay();
      };
      t.ondblclick = () => {
        const v = views.patch;
        [v.cx, v.cy] = centre(it.pts); v.z = Math.max(v.z, 2); v.limit(); v.changed();
      };
      t.onmouseenter = () => { D.hover = it; views.patch.drawOverlay(); };
      t.onmouseleave = () => { D.hover = null; views.patch.drawOverlay(); };
      grid.appendChild(t);
    });
  }

  function dustCount() {
    const D = S.dust, info = put('dustInfo', null);
    const todo = D.shown.filter((it) => !D.out.has(it.i)).length, out = D.shown.length - todo;
    $('btnDustPatch').disabled = !todo;
    $('btnDustPatch').textContent = todo ? 'Patch ' + todo : 'Patch them';
    if (!D.items) return;
    const hairs = D.shown.filter((it) => it.shape === 'hair').length, specks = D.shown.length - hairs;
    const n = (k, one) => k + ' ' + one + (k === 1 ? '' : 's');
    info.appendChild(el('p', { class: D.shown.length ? '' : 'dim', text: !D.items.length ? 'No dust found.' :
      ([hairs && n(hairs, 'hair'), specks && n(specks, 'speck')].filter(Boolean).join(' and ') || 'None') +
      (out ? ', ' + out + ' left out' : '') + (D.done ? '; ' + D.done + ' patched' : '') +
      (D.below ? ' (' + D.below + ' fainter)' : '') }));
  }

  async function loadDust() {
    const D = S.dust;
    let got = { items: null, key: null };
    if (views.patch.info) {
      try { got = await req('/api/dust'); } catch (err) { fail(err); }
    }
    if (got.key !== D.key) {                        // another search: large dots start left out, mostly paint
      D.out = new Set((got.items || []).filter((it) => it.pts.length === 1 && it.r > got.large).map((it) => it.i));
    }
    Object.assign(D, got, { hover: null });
    renderDust();
    views.patch.drawOverlay();
  }

  async function patchOp(body) {
    busy(true);
    try {
      const res = await req('/api/patch', 'POST', body);
      S.patches = res.patches;
      updateSession(res);
      views.patch.ver = S.sess.ver;                 // the same view with the patches in it: no need to fit it again
      views.patch.invalidate();
      renderPatchSide();
    } catch (err) { fail(err); await loadPatches(); } finally {
      busy(false);
      patchTool.pending = null;
      views.patch.drawOverlay();
    }
  }

  function setBrush(size) {
    S.brush = clamp(Math.round(size), 1, 300) || 24;
    $('brushIn').value = S.brush;
    remember('brush', S.brush);
    views.patch.drawOverlay();
  }

  function buildPatch() {
    setBrush(parseFloat(recall('brush')) || 24);
    $('brushIn').onchange = () => setBrush(parseFloat($('brushIn').value));
    // blurred, or the space bar would toggle it back instead of panning
    $('patchOutlines').onchange = (e) => { e.target.blur(); views.patch.drawOverlay(); };
    $('btnPatchUndo').onclick = () => S.sess.undo && patchOp({ op: 'undo' });
    $('btnDust').onclick = () => guard(async () => {
      const { job } = await req('/api/dust', 'POST', {});
      await runJob(job, 'dustProgress', 'looked for dust');
      await loadDust();
    });
    [['dustMin', 'min'], ['dustHair', 'hmin']].forEach(([id, key]) => {
      $(id).oninput = (e) => { S.dust[key] = parseInt(e.target.value, 10); renderDust(); views.patch.drawOverlay(); };
      $(id).onchange = (e) => e.target.blur();      // or the space bar would move it instead of the picture
    });
    document.querySelectorAll('[data-dust]').forEach((b) => {
      b.onclick = () => {                           // of those shown: all, none, or one shape only
        const D = S.dust, k = b.dataset.dust;
        D.shown.forEach((it) => { if (k === 'all' || k === it.shape) D.out.delete(it.i); else D.out.add(it.i); });
        b.blur(); renderDust(); views.patch.drawOverlay();
      };
    });
    $('btnDustPatch').onclick = () => {
      const todo = S.dust.shown.filter((it) => !S.dust.out.has(it.i));
      if (todo.length) patchOp({ op: 'add', strokes: todo.map((it) => ({ pts: it.pts, r: it.r })) });
    };
    twoClick($('btnPatchClear'), () => S.patches.length && patchOp({ op: 'clear' }));
    views.patch.ov.addEventListener('pointerleave', () => { patchTool.hover = null; views.patch.drawOverlay(); });
    document.addEventListener('keydown', (e) => {
      if (S.page !== 'patch' || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      if (e.key === ' ') {
        e.preventDefault();
        if (document.activeElement && document.activeElement.tagName === 'BUTTON') document.activeElement.blur();
        if (!S.space) { S.space = true; views.patch.ov.style.cursor = 'grab'; views.patch.drawOverlay(); }
      } else if (e.key === '[') setBrush(Math.min(S.brush - 1, S.brush * 0.8));
      else if (e.key === ']') setBrush(Math.max(S.brush + 1, S.brush * 1.25));
      else if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (S.sess.undo) patchOp({ op: 'undo' });
      }
    });
    document.addEventListener('keyup', (e) => {
      if (e.key === ' ') { S.space = false; views.patch.drawOverlay(); }
    });
  }

  // ============================================================== 5. color

  const lv = (ch) => (S.adj.levels && S.adj.levels[ch]) || LV_ID;
  const same = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
  function setLv(ch, a) {
    S.adj.levels = S.adj.levels || {};
    if (same(a, LV_ID)) delete S.adj.levels[ch]; else S.adj.levels[ch] = a;
    if (!Object.keys(S.adj.levels).length) delete S.adj.levels;
  }
  const IDENTITY = [[0, 0], [1, 1]];
  const pts = (ch) => (S.adj.curves && S.adj.curves[ch]) || clone(IDENTITY);
  function setPts(ch, p) {
    S.adj.curves = S.adj.curves || {};
    if (same(p.flat(), IDENTITY.flat())) delete S.adj.curves[ch]; else S.adj.curves[ch] = p;
    if (!Object.keys(S.adj.curves).length) delete S.adj.curves;
  }
  const CHAN_COLOUR = { rgb: '#c8ccd2', r: '#ff6262', g: '#5fd48a', b: '#5b9dff' };

  const pushAdj = LK.debounce(async () => {
    try {
      await req('/api/adjust', 'POST', { adj: S.adj });
      S.adjVer += 1;
      views.result.invalidate();
    } catch (err) { fail(err); }
  }, 120);

  function changedAdj() { syncColourUI(); pushAdj(); }

  const SLIDERS = [['temp', 'Temperature', -100, 100], ['tint', 'Tint', -100, 100], ['light', 'Lightness', -100, 100],
    ['contrast', 'Contrast', -100, 100], ['sat', 'Saturation', -100, 100], ['hue', 'Hue (°)', -180, 180]];

  function buildSliders() {
    const host = $('sliders');
    SLIDERS.forEach(([key, label, lo, hi]) => {
      const range = el('input', { type: 'range', id: 'sl-' + key, min: lo, max: hi, step: 1, value: 0 });
      const num = el('input', { type: 'number', id: 'sn-' + key, min: lo, max: hi, step: 1, value: 0 });
      const set = (v) => {
        v = clamp(Math.round(parseFloat(v) || 0), lo, hi);
        range.value = num.value = v;
        if (v) S.adj[key] = v; else delete S.adj[key];
        pushAdj();
      };
      range.oninput = () => set(range.value);
      num.onchange = () => set(num.value);
      host.appendChild(el('div', { class: 'slider' }, [
        el('label', { text: label, title: 'Double-click to reset', ondblclick: () => set(0) }), range, num]));
    });
  }

  /** Put the current settings on every color control. */
  function syncColourUI() {
    SLIDERS.forEach(([key]) => { $('sl-' + key).value = $('sn-' + key).value = S.adj[key] || 0; });
    const [b, g, w] = lv(S.ch);
    $('lvBlack').value = Math.round(b * 255); $('lvGamma').value = g.toFixed(2); $('lvWhite').value = Math.round(w * 255);
    LK.$$('#chanSeg button').forEach((x) => x.classList.toggle('active', x.dataset.ch === S.ch));
    const own = ['r', 'g', 'b'].filter((c) => (S.adj.levels && S.adj.levels[c]) || (S.adj.curves && S.adj.curves[c]));
    $('lvNote').hidden = S.ch !== 'rgb' || !own.length;
    $('lvNote').textContent = own.map((c) => c.toUpperCase()).join(', ') + ' also ' + (own.length > 1 ? 'have' : 'has') +
      ' levels or a curve of ' + (own.length > 1 ? 'their' : 'its') + ' own, applied first. Choose the channel to see ' +
      (own.length > 1 ? 'them' : 'it') + '.';
    drawHist(); drawCurve();
  }

  function sizeCanvas(cv) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const W = Math.max(2, Math.round(r.width * dpr)), H = Math.max(2, Math.round(r.height * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    return { c: cv.getContext('2d'), W, H, dpr };
  }

  async function loadHist() {
    const s = S.sess, key = S.epoch + ':' + s.ver;
    if (!s.open || !s.aligned || S.histKey === key) { drawHist(); drawCurve(); return; }
    try {
      S.hist = (await req('/api/histogram')).hist;
      S.histKey = key;
    } catch (err) { fail(err); }
    drawHist(); drawCurve();
  }

  const CH_INDEX = { rgb: 3, r: 0, g: 1, b: 2 };
  function paintBins(c, bins, W, H, colour, alpha) {
    const sorted = bins.slice().sort((a, b) => a - b), top = Math.sqrt(sorted[Math.floor(255 * 0.995)] || sorted[255] || 1);
    c.fillStyle = colour; c.globalAlpha = alpha;
    for (let i = 0; i < 256; i += 1) {
      const h = Math.min(1, Math.sqrt(bins[i]) / top) * H;
      c.fillRect(Math.floor(i * W / 256), H - h, Math.ceil(W / 256), h);
    }
    c.globalAlpha = 1;
  }

  function drawHist() {
    const { c, W, H, dpr } = sizeCanvas($('histCanvas')), band = Math.round(16 * dpr), plot = H - band;
    c.clearRect(0, 0, W, H);
    if (S.hist) {
      if (S.ch === 'rgb') {
        paintBins(c, S.hist[3], W, plot, '#8a919b', 0.9);
        ['r', 'g', 'b'].forEach((k) => paintBins(c, S.hist[CH_INDEX[k]], W, plot, CHAN_COLOUR[k], 0.28));
      } else paintBins(c, S.hist[CH_INDEX[S.ch]], W, plot, CHAN_COLOUR[S.ch], 0.85);
    }
    const [b, g, w] = lv(S.ch), mid = b + (w - b) * Math.pow(0.5, g);
    [[b, '#000', '#e2e6eb'], [mid, '#8a919b', '#e2e6eb'], [w, '#fff', '#555']].forEach(([x, fill, line]) => {
      const px = x * W, s = 6 * dpr;
      c.beginPath(); c.moveTo(px, plot + 2 * dpr); c.lineTo(px - s, H - 1); c.lineTo(px + s, H - 1); c.closePath();
      c.fillStyle = fill; c.fill(); c.strokeStyle = line; c.lineWidth = dpr; c.stroke();
    });
  }

  function buildHist() {
    const cv = $('histCanvas');
    let drag = null;
    const xOf = (e) => { const r = cv.getBoundingClientRect(); return clamp((e.clientX - r.left) / r.width, 0, 1); };
    cv.addEventListener('pointerdown', (e) => {
      const [b, g, w] = lv(S.ch), x = xOf(e), mid = b + (w - b) * Math.pow(0.5, g), tol = 9 / cv.getBoundingClientRect().width;
      const cand = [['b', b], ['m', mid], ['w', w]].map(([k, v]) => [k, Math.abs(v - x)]).sort((p, q) => p[1] - q[1])[0];
      if (cand[1] > tol) return;
      drag = cand[0]; cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const a = lv(S.ch).slice(), x = xOf(e), eps = 1 / 255;
      if (drag === 'b') a[0] = clamp(x, 0, a[2] - eps);
      else if (drag === 'w') a[2] = clamp(x, a[0] + eps, 1);
      else a[1] = clamp(Math.log(clamp((x - a[0]) / (a[2] - a[0]), 0.02, 0.98)) / Math.log(0.5), 0.1, 9.99);
      setLv(S.ch, a); changedAdj();
    });
    cv.addEventListener('pointerup', () => { drag = null; });
    const fromInputs = () => {
      const a = lv(S.ch).slice();
      a[0] = clamp((parseFloat($('lvBlack').value) || 0) / 255, 0, 254 / 255);
      a[2] = clamp((parseFloat($('lvWhite').value) || 255) / 255, a[0] + 1 / 255, 1);
      a[1] = clamp(parseFloat($('lvGamma').value) || 1, 0.1, 9.99);
      setLv(S.ch, a); changedAdj();
    };
    ['lvBlack', 'lvGamma', 'lvWhite'].forEach((id) => { $(id).onchange = fromInputs; });
    LK.$$('#chanSeg button').forEach((b) => { b.onclick = () => { S.ch = b.dataset.ch; S.curveY = null; syncColourUI(); fetchCurve(); }; });
    const auto = (perChannel) => guard(async () => {
      const r = await req('/api/autolevels', 'POST', { clip: 0.001, per_channel: perChannel });
      S.adj.levels = r.levels; changedAdj();
    });
    $('btnAutoAll').onclick = () => auto(false);
    $('btnAutoEach').onclick = () => auto(true);
  }

  // --- the curve
  async function fetchCurve() {
    const p = pts(S.ch);
    if (same(p.flat(), IDENTITY.flat())) { S.curveY = null; drawCurve(); return; }
    try { S.curveY = (await req('/api/curve', 'POST', { pts: p })).y; } catch (err) { return; }
    drawCurve();
  }

  function drawCurve() {
    const { c, W, H, dpr } = sizeCanvas($('curveCanvas')), p = pts(S.ch);
    c.clearRect(0, 0, W, H);
    if (S.hist) paintBins(c, S.hist[CH_INDEX[S.ch]], W, H, CHAN_COLOUR[S.ch], 0.16);
    c.strokeStyle = css('--line'); c.lineWidth = dpr; c.beginPath();
    for (let i = 1; i < 4; i += 1) { c.moveTo(i * W / 4, 0); c.lineTo(i * W / 4, H); c.moveTo(0, i * H / 4); c.lineTo(W, i * H / 4); }
    c.stroke();
    c.strokeStyle = css('--edge'); c.beginPath(); c.moveTo(0, H); c.lineTo(W, 0); c.stroke();
    c.strokeStyle = CHAN_COLOUR[S.ch]; c.lineWidth = 2 * dpr; c.beginPath();
    const y = S.curveY || [0, 1].map((v) => v);
    y.forEach((v, i) => { const x = i / (y.length - 1) * W; if (i) c.lineTo(x, (1 - v) * H); else c.moveTo(x, (1 - v) * H); });
    c.stroke();
    c.fillStyle = css('--ink'); c.strokeStyle = CHAN_COLOUR[S.ch];
    p.forEach(([x, v]) => { c.beginPath(); c.arc(x * W, (1 - v) * H, 4.5 * dpr, 0, 6.2832); c.fill(); c.stroke(); });
  }

  function buildCurve() {
    const cv = $('curveCanvas');
    let drag = null, busyCurve = false;
    const at = (e) => { const r = cv.getBoundingClientRect(); return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp(1 - (e.clientY - r.top) / r.height, 0, 1), r]; };
    const nearest = (p, x, y, r) => {
      let best = -1, d = 1e9;
      p.forEach(([px, py], i) => { const dd = Math.hypot((px - x) * r.width, (py - y) * r.height); if (dd < d) { d = dd; best = i; } });
      return d < 11 ? best : -1;
    };
    const edit = (p) => { setPts(S.ch, p); pushAdj(); if (!busyCurve) { busyCurve = true; fetchCurve().finally(() => { busyCurve = false; }); } drawCurve(); };
    cv.addEventListener('pointerdown', (e) => {
      const [x, y, r] = at(e), p = pts(S.ch).map((q) => q.slice());
      let i = nearest(p, x, y, r);
      if (i < 0) { p.push([x, y]); p.sort((a, b) => a[0] - b[0]); i = p.findIndex((q) => q[0] === x && q[1] === y); edit(p); }
      drag = i; cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', (e) => {
      if (drag === null) return;
      const [x, y] = at(e), p = pts(S.ch).map((q) => q.slice());
      if (!p[drag]) return;
      p[drag][1] = y;
      p[drag][0] = clamp(x, drag > 0 ? p[drag - 1][0] + 0.005 : 0, drag < p.length - 1 ? p[drag + 1][0] - 0.005 : 1);
      edit(p);
    });
    cv.addEventListener('pointerup', () => { drag = null; });
    cv.addEventListener('dblclick', (e) => {
      const [x, y, r] = at(e), p = pts(S.ch).map((q) => q.slice()), i = nearest(p, x, y, r);
      if (i > 0 && i < p.length - 1) { p.splice(i, 1); edit(p); }
    });
  }

  // --- eyedroppers
  const pickTool = {
    cursor() { return 'crosshair'; },
    down() { return true; },
    up(p) {
      const kind = S.pickKind;
      setPick(null);
      guard(async () => applyPick(kind, (await req('/api/pick', 'POST', { x: p.x, y: p.y })).rgb));
    },
  };

  function setPick(kind) {
    S.pickKind = kind;
    LK.$$('[data-pick]').forEach((b) => b.classList.toggle('armed', b.dataset.pick === kind));
    views.result.tool = kind ? pickTool : null;
    views.result.ov.style.cursor = kind ? 'crosshair' : 'grab';
  }

  function applyPick(kind, rgb) {
    const chs = ['r', 'g', 'b'], eps = 1 / 255;
    if (kind === 'grey') {                          // the spot becomes neutral: one gamma per channel, same brightness overall
      const t = chs.map((c, i) => { const [b, , w] = lv(c); return (rgb[i] / 255 - b) / (w - b); });
      if (t.some((v) => v <= 0.02 || v >= 0.98)) return toast('That spot is too dark or too bright to be the gray point.', 'warn');
      const m = t.reduce((a, b) => a + b, 0) / 3;
      chs.forEach((c, i) => { const a = lv(c).slice(); a[1] = clamp(Math.log(t[i]) / Math.log(m), 0.1, 9.99); setLv(c, a); });
    } else {
      chs.forEach((c, i) => {
        const a = lv(c).slice();
        if (kind === 'black') a[0] = clamp(rgb[i] / 255, 0, a[2] - eps); else a[2] = clamp(rgb[i] / 255, a[0] + eps, 1);
        setLv(c, a);
      });
    }
    changedAdj();
  }

  // --- saved color settings
  function refreshColourSelects() {
    const names = Object.keys(S.colours).sort();
    options($('colourSel'), [{ value: '', label: '— saved settings —' }].concat(names), $('colourSel').value);
    options($('batchColour'), [{ value: '', label: 'as scanned' }, { value: 'auto', label: 'auto levels' }]
      .concat(names.map((n) => ({ value: 'set:' + n, label: n }))), $('batchColour').value);
  }

  function buildColour() {
    buildSliders(); buildHist(); buildCurve();
    LK.$$('[data-pick]').forEach((b) => { b.onclick = () => {
      if (!views.result.info) return toast('Stitch the set first.', 'warn');
      setPick(S.pickKind === b.dataset.pick ? null : b.dataset.pick);
    }; });
    $('showRaw').onchange = () => views.result.invalidate();
    $('btnColourLoad').onclick = () => {
      const a = S.colours[$('colourSel').value];
      if (!a) return toast('Choose saved settings first.', 'warn');
      S.adj = clone(a); S.curveY = null; changedAdj(); fetchCurve();
    };
    $('btnColourSave').onclick = () => guard(async () => {
      const name = $('colourName').value.trim();
      if (!name) return toast('Give the settings a name.', 'warn');
      S.colours = (await req('/api/colours', 'POST', { name, adj: S.adj })).colours;
      refreshColourSelects(); $('colourSel').value = name;
      toast('Saved “' + name + '”.');
    });
    twoClick($('btnColourDel'), () => guard(async () => {
      const name = $('colourSel').value;
      if (!name) return toast('Choose saved settings first.', 'warn');
      S.colours = (await req('/api/colours', 'DELETE', { name })).colours;
      refreshColourSelects();
    }));
    $('btnColourReset').onclick = () => { S.adj = {}; S.curveY = null; changedAdj(); };
  }

  // ============================================================== 6. export

  function fillPresetSelects() {
    options($('presetSel'), S.presets.map((p) => p.name), S.presets[S.presetIx] && S.presets[S.presetIx].name);
    options($('batchPreset'), S.presets.map((p) => p.name), $('batchPreset').value || (S.presets[0] && S.presets[0].name));
  }

  function loadPreset(ix) {
    S.presetIx = clamp(ix, 0, S.presets.length - 1);
    S.preset = clone(S.presets[S.presetIx]);
    $('presetSel').value = S.preset.name;
    $('presetName').value = S.preset.name;
    $('outFolder').value = S.preset.folder || '';
    options($('overwriteSel'), S.consts.overwrite, S.preset.overwrite || 'number');
    renderOutputs();
    planUpdate();
  }

  function newOutput() {
    return { kind: 'png', name: '{name}_' + (S.preset.outputs.length + 1), size: 'dpi', dpi: 300 };
  }

  const UNITS = { dpi: 'dpi', long: 'px', mb: 'MB' };
  const EXT = { tiff: '.tiff', psd: '.psd', png: '.png', jpg: '.jpg', pdf: '.pdf' };
  const FULL = ['tiff', 'psd'];                     // always full size
  const SIZE_DEFAULT = { dpi: 300, long: 2500, mb: 10 };

  function outputRow(o, i) {
    const C = S.consts;
    const kind = el('select', { class: 'kind' });
    options(kind, C.kinds, o.kind);
    const name = el('input', { type: 'text', class: 'fname', value: o.name || '', spellcheck: 'false', placeholder: '{name}' });
    const ext = el('span', { class: 'dim mono', text: EXT[o.kind] });
    const rest = el('div', { class: 'row wrap' });
    const note = el('div', { class: 'hint' });
    const number = (key, min, max, step) => {
      const box = el('input', { type: 'number', class: 'val', min, max, step, value: o[key] === undefined ? '' : o[key] });
      box.oninput = () => { o[key] = parseFloat(box.value); planUpdate(); };
      return box;
    };
    const fill = () => {
      rest.textContent = ''; note.textContent = '';
      const help = (list, v) => (list.find((x) => x.value === v) || {}).help || '';
      const say = () => {
        note.textContent = [o.kind === 'tiff' ? help(C.compressions, o.compression || 'zopfli') : '', help(C.depths, String(o.bits || 'auto'))].filter(Boolean).join('  ·  ');
      };
      const bits = el('select', { class: 'kind', title: 'Bits a channel of the file (PNG, JPEG and PDF are always 8)' });
      options(bits, C.depths, String(o.bits || 'auto'));
      bits.onchange = () => { o.bits = bits.value === 'auto' ? 'auto' : +bits.value; say(); planUpdate(); };
      if (o.kind === 'tiff') {
        const comp = el('select', { class: 'kind' });
        options(comp, C.compressions, o.compression || 'zopfli');
        comp.onchange = () => { o.compression = comp.value; say(); };
        say();
        const layers = el('input', { type: 'checkbox', onchange: () => { o.layers = layers.checked; } });
        layers.checked = !!o.layers;
        rest.append(el('span', { class: 'dim', text: 'full size, lossless —' }), bits, el('span', { class: 'dim', text: 'compression' }), comp,
          el('label', { class: 'chk', title: 'Each scan as a layer, whole, with a mask that cuts it to its part of the stitch; the patches on a layer above them; ' +
            'each color adjustment as a Photoshop adjustment layer on top (at 32 bits the color is in the layers instead). Every other program sees the finished picture. About three times the size.' },
          [layers, ' Photoshop layers: scans, patches, color']));
      } else if (o.kind === 'psd') {
        say();
        rest.append(el('span', { class: 'dim', text: 'full size, lossless —' }), bits,
          el('span', { class: 'dim', text: 'each scan as a layer with a mask, the patches above them, the color adjustments as adjustment layers on top' }));
      } else {
        const size = el('select', { class: 'kind' });
        options(size, C.sizes, o.size || 'dpi');
        size.onchange = () => { o.size = size.value; if (SIZE_DEFAULT[o.size] && !(o[o.size] > 0)) o[o.size] = SIZE_DEFAULT[o.size]; fill(); planUpdate(); };
        rest.append(size);
        const key = o.size || 'dpi';
        if (UNITS[key]) rest.append(number(key, 1, 100000, key === 'mb' ? 0.5 : 1), el('span', { class: 'dim', text: UNITS[key] }));
        if (o.kind === 'jpg') {
          if (!(o.quality > 0)) o.quality = 92;
          rest.append(el('span', { class: 'spacer' }), el('span', { class: 'dim', text: 'quality' }), number('quality', 1, 100, 1));
        }
        if (o.kind === 'pdf') {
          const how = el('select', { class: 'kind', title: 'Lossless keeps every pixel (as PNG does); JPEG makes a far smaller file' });
          options(how, [{ value: 'lossless', label: 'lossless' }, { value: 'jpeg', label: 'JPEG' }], o.image || 'lossless');
          how.onchange = () => { o.image = how.value; if (o.image === 'jpeg' && !(o.quality > 0)) o.quality = 92; fill(); planUpdate(); };
          rest.append(el('span', { class: 'spacer' }), el('span', { class: 'dim', text: 'picture' }), how);
          if (o.image === 'jpeg') rest.append(el('span', { class: 'dim', text: 'quality' }), number('quality', 1, 100, 1));
        }
      }
    };
    kind.onchange = () => {
      o.kind = kind.value; ext.textContent = EXT[o.kind];
      if (!FULL.includes(o.kind) && !o.size) { o.size = 'dpi'; o.dpi = o.dpi || 300; }
      fill(); planUpdate();
    };
    name.oninput = () => { o.name = name.value; planUpdate(); };
    fill();
    return el('div', { class: 'output' }, [
      el('div', { class: 'row' }, [kind, name, ext,
        el('button', { class: 'drop', title: 'Remove this file from the preset', text: '×', onclick: () => { S.preset.outputs.splice(i, 1); renderOutputs(); planUpdate(); } })]),
      rest, note]);
  }

  function renderOutputs() {
    const host = put('outputs', null);
    S.preset.outputs.forEach((o, i) => host.appendChild(outputRow(o, i)));
    $('placeholderHelp').textContent = S.consts.placeholders.map((p) => '{' + p.key + '}').join(' ');
    $('placeholderHelp').title = S.consts.placeholders.map((p) => '{' + p.key + '} — ' + p.help).join('\n');
  }

  const planUpdate = LK.debounce(async () => {
    const head = $('planHead'), list = put('planList', null);
    const s = S.sess;
    if (!s.open || !s.aligned || !s.seamed) { head.textContent = s.open ? 'Stitch the set first.' : 'Open a set of scans first.'; return; }
    try {
      const r = await req('/api/plan', 'POST', { preset: S.preset });
      const [w, h] = r.size, dpi = r.dpi;
      head.textContent = int(w) + ' × ' + int(h) + ' px  ·  ' + (w * h / 1e6).toFixed(0) + ' MP  ·  ' + Math.round(dpi) + ' dpi  ·  ' +
        (w / dpi * 2.54).toFixed(1) + ' × ' + (h / dpi * 2.54).toFixed(1) + ' cm';
      r.problems.forEach((p) => list.appendChild(el('div', { class: 'qc-item error', text: p })));
      const grid = el('div', { class: 'plan' });
      r.outputs.forEach((o) => grid.append(el('span', { class: 'badge', text: o.kind }), el('span', { class: 'file', text: o.file }),
        el('span', { class: 'dim', text: o.note || (int(o.w) + ' × ' + int(o.h) + ' px · ' + Math.round(o.dpi) + ' dpi' + (o.bits ? ' · ' + o.bits + ' bits' : '')) })));
      if (r.outputs.length) list.appendChild(grid);
    } catch (err) { head.textContent = err.message; }
  }, 250);

  async function savePresets(list, select) {
    S.presets = (await req('/api/presets', 'POST', { presets: list })).presets;
    fillPresetSelects();
    loadPreset(Math.max(0, S.presets.findIndex((p) => p.name === select)));
  }

  function buildExport() {
    $('presetSel').onchange = () => loadPreset(S.presets.findIndex((p) => p.name === $('presetSel').value));
    $('outFolder').oninput = () => { S.preset.folder = $('outFolder').value; };
    $('overwriteSel').onchange = () => { S.preset.overwrite = $('overwriteSel').value; };
    $('btnAddOutput').onclick = () => { S.preset.outputs.push(newOutput()); renderOutputs(); planUpdate(); };
    $('btnPresetSave').onclick = () => guard(async () => {
      const name = $('presetName').value.trim(), list = clone(S.presets);
      list[S.presetIx] = Object.assign(clone(S.preset), { name });
      await savePresets(list, name);
      toast('Saved preset “' + name + '”.');
    });
    $('btnPresetNew').onclick = () => guard(async () => {
      const name = $('presetName').value.trim(), list = clone(S.presets);
      list.push(Object.assign(clone(S.preset), { name }));
      await savePresets(list, name);
      toast('Saved preset “' + name + '”.');
    });
    twoClick($('btnPresetDel'), () => guard(async () => {
      if (S.presets.length < 2) return toast('The last preset cannot be deleted.', 'warn');
      const list = clone(S.presets);
      list.splice(S.presetIx, 1);
      await savePresets(list, list[0].name);
    }));
    $('btnExport').onclick = () => {
      if (!S.sess.open || !S.sess.seamed) return toast('Stitch the set first.', 'warn');
      guard(async () => {
        put('exportResult', null);
        const { job } = await req('/api/export', 'POST', { preset: S.preset });
        const r = await runJob(job, 'exportProgress', 'saved');
        renderExports(r);
        toast('Saved ' + r.outputs.filter((o) => !o.skipped).length + ' file(s).');
      });
    };
    $('btnSaveSession').onclick = () => guard(async () => {
      if (!S.sess.open) return toast('Open a set of scans first.', 'warn');
      // Asked, not assumed: the session is a file the person names and puts
      // where they want it, the way the open dialog already reads one back.
      const path = await LK.save({
        ext: '.artstitch.json', exts: ['.artstitch.json'],
        name: (S.sess.name || 'session') + '.artstitch.json',
        path: S.sess.folder || '', title: 'Save this session',
        hint: 'Give the session a file name',
      });
      if (!path) return;
      const r = await req('/api/session/save', 'POST', { path });
      toast('Session saved: ' + home(r.path));
    });
  }

  function renderExports(r) {
    const host = put('exportResult', null);
    r.outputs.forEach((o) => host.appendChild(el('div', { class: 'row' }, [
      el('span', { class: 'badge ' + (o.skipped ? 'warn' : 'ok'), text: o.skipped ? 'skipped' : o.kind }),
      el('span', { class: 'mono', style: { flex: '1 1 auto', wordBreak: 'break-all' }, text: base(o.path), title: o.path }),
      el('span', { class: 'dim mono', text: o.skipped ? 'already there' : int(o.w) + ' × ' + int(o.h) + (o.bits ? '  ·  ' + o.bits + ' bits' : '') + '  ·  ' + megabytes(o.bytes) + (o.layers ? '  ·  ' + o.layers + ' layers' : '') }),
      el('button', { class: 'small ghost', text: 'Show', onclick: () => req('/api/reveal', 'POST', { path: o.path }).catch(fail) })])));
    host.appendChild(el('p', { class: 'hint', text: 'Session kept as ' + home(r.session) + ' — open it from the first page to change anything and save again.' }));
  }

  // ================================================================ 7. batch

  function batchRows() {
    const host = $('batchTable'), B = S.batch;
    host.classList.toggle('hidden', !B.sets.length);
    if (!B.sets.length) return;
    const locked = B.running;
    // the folders the sets sit in, without what they all share: nothing at all when it is one folder
    const dirs = B.sets.map((x) => x.folder), same = dirs.every((d) => d === dirs[0]);
    const shared = same ? dirs[0] : dirs.reduce((a, d) => { let i = 0; while (i < a.length && a[i] === d[i]) i++; return a.slice(0, i); }, dirs[0]).replace(/[^/\\]*$/, '');
    const head = el('tr', {}, ['', 'set', 'scans', 'state', 'seconds', 'notes', ''].map((t) => el('th', { text: t })));
    const body = [];
    B.sets.forEach((set, i) => {
      const st = B.status[i] || {};
      const box = el('input', { type: 'checkbox', disabled: locked || null, onchange: () => { set.include = box.checked; batchNote(); batchRows(); } });
      box.checked = !!set.include;
      const n = set.files ? set.files.filter((f) => f.include).length : set.n_files;
      const state = st.state && (st.state !== 'skipped' || set.include) ? st.state : '';
      const kind = { ok: 'ok', review: 'warn', failed: 'bad', running: 'accent', canceled: 'warn' }[state] || '';
      const notes = st.error || st.message && state !== 'ok' && state !== 'review' ? (st.error || st.message)
        : (st.qc || []).map((q) => q.msg).join('; ');
      const more = set.files ? el('button', { class: 'small ghost', text: B.open.has(i) ? '▾' : '▸', title: 'Which scans',
        onclick: () => { if (B.open.has(i)) B.open.delete(i); else B.open.add(i); batchRows(); } }) : null;
      body.push(el('tr', {}, [el('td', {}, [box]), el('td', {}, [el('b', { text: set.name }), ' ', el('span', { class: 'dim mono', text: set.folder.slice(shared.length).replace(/^[/\\]/, '') })]),
        el('td', { class: 'num', text: n + (set.files ? ' of ' + set.files.length : '') }),
        el('td', {}, [state ? el('span', { class: 'badge state ' + kind, text: state }) : '']),
        el('td', { class: 'num', text: st.seconds ? st.seconds.toFixed(0) : '' }),
        el('td', { text: notes }),
        el('td', {}, [more, ' ', st.session ? el('button', { class: 'small', text: 'Open', title: 'Open in the editor, as the batch left it',
          onclick: () => guard(async () => { openedSession(await req('/api/batch/open', 'POST', { index: i })); showPage('stitch'); }) }) : null,
          ' ', st.outputs && st.outputs.length ? el('button', { class: 'small ghost', text: 'Show', onclick: () => req('/api/reveal', 'POST', { path: st.outputs[0].path }).catch(fail) }) : null])]));
      if (set.files && B.open.has(i)) {
        body.push(el('tr', { class: 'sub' }, [el('td'), el('td', { colspan: 6 }, [el('div', { class: 'setcard' }, [el('div', { class: 'files' },
          set.files.map((f) => fileRow(f, () => { if (locked) return; batchRows(); })))])])]));
      }
    });
    put('batchTable', el('table', { class: 'grid' }, [el('thead', {}, [head]), el('tbody', {}, body)]));
  }

  function batchNote() {
    const sets = S.batch.sets, n = sets.filter((x) => x.include).length;
    $('batchNote').textContent = sets.length ? sets.length + ' sets found, ' + n + ' checked. Sets with one scan, and scans that look like an earlier result, start unchecked.' : '';
  }

  function batchColour() {
    const v = $('batchColour').value;
    if (v === 'auto') return { mode: 'auto', clip: 0.001 };
    if (v.startsWith('set:')) return { mode: 'adj', adj: S.colours[v.slice(4)] || {} };
    return {};
  }

  function setBatchRunning(on) {
    S.batch.running = on;
    $('btnBatchRun').disabled = on;
    $('btnBatchStop').classList.toggle('hidden', !on);
    $('btnBatchStop').disabled = false; $('btnBatchStop').textContent = 'Stop after this set';
    batchRows();
  }

  async function pollBatch() {
    const host = $('batchProgress'), B = S.batch;
    const fill = el('div', { class: 'bar' }), line = el('span', { text: 'starting…' });
    put('batchProgress', el('div', { class: 'progress' }, [fill, line]));
    setBatchRunning(true);
    try {
      for (;;) {
        const st = await req('/api/batch/status');
        B.status = st.rows; batchRows();
        fill.style.width = Math.round(st.progress * 100) + '%'; line.textContent = st.message;
        if (!st.running) break;
        await sleep(1000);
      }
      const job = await req('/api/job/' + B.job);
      if (job.error) throw new Error(job.error);
      const t = job.result.tally, said = Object.entries(t).filter(([k, n]) => n && k !== 'queued').map(([k, n]) => n + ' ' + k).join(', ');
      line.textContent = 'finished — ' + said + (job.result.report ? '  ·  report: ' + home(job.result.report) : '');
      toast('Batch finished: ' + said, t.failed ? 'warn' : 'ok');
    } catch (err) { host.textContent = ''; fail(err); }
    setBatchRunning(false);
  }

  function buildBatch() {
    batchPick = LK.pathPicker($('batchPick'), { mode: 'dir', value: recall('folder') || LK.recentFolders()[0] || '', placeholder: '/folder/with many artworks' });
    $('btnBatchScan').onclick = () => guard(async () => {
      const folder = batchPick.value.trim();
      if (!folder) return toast('Give a folder first.', 'warn');
      remember('folder', folder);
      const r = await req('/api/scan', 'POST', { folder, recursive: $('batchRecursive').checked, all_images: $('batchAll').checked });
      Object.assign(S.batch, { sets: r.sets, status: [], root: folder, open: new Set() });
      $('batchSetup').classList.toggle('hidden', !r.sets.length);
      put('batchProgress', null);
      batchNote();
      batchRows();
      if (!r.sets.length) toast('No scans found in that folder.', 'warn');
    });
    $('btnBatchRun').onclick = () => {
      const B = S.batch, preset = S.presets.find((p) => p.name === $('batchPreset').value);
      if (!B.sets.some((s) => s.include)) return toast('Check at least one set.', 'warn');
      if (!preset) return toast('Choose a preset to save with.', 'warn');
      guard(async () => {
        B.status = [];
        B.job = (await req('/api/batch/run', 'POST', { sets: B.sets, preset, colour: batchColour(), root: B.root })).job;
        pollBatch();
      });
    };
    $('btnBatchStop').onclick = () => S.batch.job && req('/api/job/' + S.batch.job + '/cancel', 'POST').then(() => {
      $('btnBatchStop').disabled = true; $('btnBatchStop').textContent = 'Stopping…';
      toast('The set in progress is finished, the rest are not run.', 'warn');
    }).catch(fail);
  }

  /** A batch that is still running (or finished) on the server when the page was opened. */
  async function restoreBatch() {
    const st = await req('/api/batch/status');
    if (!st.rows.length) return;
    Object.assign(S.batch, { sets: st.rows, status: st.rows, job: st.job });
    $('batchSetup').classList.remove('hidden');
    batchNote();
    batchRows();
    if (st.running) pollBatch();
  }

  // ================================================================== boot

  function wireViews() {
    views.stitch = new Viewer($('viewStitch'), { mode: 'mosaic', readout: 'zoomStitch',
      params: () => ({ overlay: [$('ovSeams').checked ? 'seams' : '', $('ovOutlines').checked ? 'outlines' : ''].filter(Boolean).join(',') }) });
    views.plane = new Viewer($('viewPlane'), { mode: 'plane', readout: 'zoomPlane', after: cropFromGeom });
    views.result = new Viewer($('viewResult'), { mode: 'result', readout: 'zoomResult', params: () => ({ raw: $('showRaw').checked ? '1' : '' }) });
    views.patch = new Viewer($('viewPatch'), { mode: 'result', readout: 'zoomPatch' });
    views.plane.tool = gridTool(cropTool);
    views.patch.tool = patchTool;
    ['ovSeams', 'ovOutlines'].forEach((id) => { $(id).onchange = () => views.stitch.invalidate(); });
    $('ovGrid').onchange = () => views.plane.drawOverlay();
    LK.$$('[data-zoom]').forEach((b) => { b.onclick = () => { const v = views[b.dataset.view]; if (b.dataset.zoom === 'fit') v.zoomFit(); else v.zoomOne(); }; });
    // the buttons say stitch / plane / result; the viewers are keyed the same way
  }

  async function boot() {
    S.consts = await req('/api/constants');
    wireViews();
    LK.$$('#rail button').forEach((b) => { b.onclick = () => showPage(b.dataset.page); });
    buildScans(); buildStraighten(); buildPatch(); buildColour(); buildExport(); buildBatch();
    $('btnStitch').onclick = () => doStitch();
    $('btnArrange').onclick = arrange;
    ['setName', 'maxDeg', 'dpiIn'].forEach((id) => { $(id).onchange = () => S.sess.open && guard(saveSettings); });
    LK.mountThemeToggle('.topbar');
    document.addEventListener('labkit:theme', () => { Object.values(views).forEach((v) => v.draw()); drawHist(); drawCurve(); });

    S.presets = (await req('/api/presets')).presets;
    S.colours = (await req('/api/colours')).colours;
    fillPresetSelects(); refreshColourSelects(); loadPreset(0);

    const s = await req('/api/session');
    if (s.open) { openedSession(s); showPage('stitch'); } else showPage('scans');
    restoreBatch().catch(fail);
  }

  boot().catch(fail);
}());
