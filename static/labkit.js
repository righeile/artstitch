/* labkit -- the browser half of the shared toolkit.
 *
 * Exposes one global, `LK`.  No build step, no framework, no dependencies:
 * these apps are opened by double-clicking a .command file on a lab machine,
 * and anything that needs `npm install` first will eventually be the reason
 * one of them stops working.
 *
 *   LK.theme.toggle()                 dark <-> light, remembered
 *   LK.browse({...})                  the file dialog, in all of them
 *   LK.pathPicker(host, {...})        a path box with a Browse button
 *   LK.hexBox(colorInput)             the well's hex code, typed or pasted
 *   LK.PlotControls(host, {...})      the plotting panel, in all of them
 *   LK.plateEditor(host, {...})       a plate map: which wells got what
 *   LK.api(url, opts)                 fetch that throws the server's message
 *   LK.nameMatch(query)              a filter box, as a predicate over a name
 */
(function (global) {
  'use strict';

  const LK = {};

  // ---------------------------------------------------------------- DOM ---

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const key in (attrs || {})) {
      const value = attrs[key];
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'html') node.innerHTML = value;
      else if (key.slice(0, 2) === 'on') node.addEventListener(key.slice(2), value);
      else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
      else if (value === true) node.setAttribute(key, '');
      else node.setAttribute(key, value);
    }
    (Array.isArray(children) ? children : children ? [children] : [])
      .forEach((child) => child != null &&
        node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child));
    return node;
  }
  LK.el = el;
  LK.$ = (selector, root) => (root || document).querySelector(selector);

  /** Coalesce a burst of calls into one, *ms* after the last.  Typing "180"
      into a number box is three input events and should be one redraw. */
  LK.debounce = function (fn, ms) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms == null ? 220 : ms);
    };
  };
  LK.$$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));

  // -------------------------------------------------------------- names ---

  /** What someone typed in a filter box, as a predicate over a name.
   *
   * A box that only knows "contains" cannot say "every dish except the two
   * dead mutants", which is the filter a batch of constructs actually needs.
   * So these read the way they are written:
   *
   *     NOT E247A            NOT ('E247A' OR 'E115A')       dish2 AND NOT E247A
   *
   * `not`/`!`, `and`/`&&`, `or`/`||` and parentheses are the operators, `and`
   * is implied between adjacent terms, quotes protect a term that contains one
   * of those words, and a term still matches as a case-insensitive substring.
   *
   * An operator only counts where it stands on its own -- `not`/`!` at the
   * front of a term, `and`/`or` with space either side, a query that opens
   * with a bracket -- so "R&D" and "Image (2)" keep meaning the characters
   * someone typed rather than quietly becoming two terms and matching more
   * files than they name.  That matters here because "select shown" acts on
   * whatever the box left on screen.
   *
   * Never throws: a half-typed query is matched as far as it goes, because
   * this runs on every keystroke.
   */
  const FILTER_OPS = /(^|\s)(not\b|!)|\s(and|or)\s|\s(&&?|\|\|?)\s|^\s*\(/i;
  const FILTER_TOKENS = /\(|\)|&&|\|\||[!&|]|"[^"]*"|'[^']*'|[^\s()!&|"']+/g;

  LK.nameMatch = function (query) {
    const text = String(query == null ? '' : query).trim();
    if (!text) return () => true;
    const has = (needle) => (name) => String(name).toLowerCase().includes(needle);
    // Quotes are how a term says "these characters, operator words and all",
    // so they come off whether or not anything else in the box is an operator.
    const bare = (t) => ((t.length > 1 && (t[0] === '"' || t[0] === "'") && t[t.length - 1] === t[0])
      ? t.slice(1, -1) : t);
    if (!FILTER_OPS.test(text)) return has(bare(text).toLowerCase());

    const tokens = text.match(FILTER_TOKENS) || [];
    let at = 0;
    const peek = () => tokens[at];
    const is = (token, ...words) => token != null && words.indexOf(token.toLowerCase()) >= 0;
    const ends = (token) => token == null || is(token, 'or', '||', '|', ')');

    function factor() {
      const token = peek();
      if (is(token, 'not', '!')) {
        at += 1;
        if (peek() == null) return () => true;   // mid-word: someone typing "notes"
        const inner = factor();
        return (n) => !inner(n);
      }
      if (token === '(') {
        at += 1;
        const inner = any();
        if (peek() === ')') at += 1;
        return inner;
      }
      at += 1;
      if (token == null) return () => true;
      return has(bare(token).toLowerCase());
    }
    function all() {
      let left = factor();
      while (!ends(peek())) {
        if (is(peek(), 'and', '&&', '&')) { at += 1; if (ends(peek())) break; }
        const right = factor();
        const prev = left;
        left = (n) => prev(n) && right(n);
      }
      return left;
    }
    function any() {
      let left = all();
      while (is(peek(), 'or', '||', '|')) {
        at += 1;
        const right = all();
        const prev = left;
        left = (n) => prev(n) || right(n);
      }
      return left;
    }
    return any();
  };

  /** The sentence every filter box hangs on its own tooltip. */
  LK.FILTER_HINT = 'Matches anywhere in the name. NOT, AND, OR and parentheses '
    + "work too: NOT ('E247A' OR 'E115A').";

  // --------------------------------------------------------------- fetch ---

  /** fetch that surfaces the server's own error text instead of "500". */
  LK.api = async function (url, options) {
    const opts = Object.assign({ headers: {} }, options || {});
    if (opts.body && typeof opts.body !== 'string') {
      opts.body = JSON.stringify(opts.body);
      opts.headers['Content-Type'] = 'application/json';
      opts.method = opts.method || 'POST';
    }
    const response = await fetch(url, opts);
    const text = await response.text();
    let payload;
    try { payload = text ? JSON.parse(text) : {}; } catch (e) { payload = { error: text }; }
    if (!response.ok) {
      throw new Error(payload.error || payload.detail || text || response.statusText);
    }
    return payload;
  };

  // --------------------------------------------------------------- theme ---
  /* Dark is the default because five of the six apps run dark and a microscope
     room is not a bright place.  The choice is written to <html data-theme> so
     it beats the system setting in both directions, and remembered per
     machine rather than per app -- switching tools should not switch look. */

  const THEME_KEY = 'labkit.theme';
  LK.theme = {
    get() { return document.documentElement.getAttribute('data-theme') || 'dark'; },
    set(name) {
      document.documentElement.setAttribute('data-theme', name);
      try { localStorage.setItem(THEME_KEY, name); } catch (e) { /* private mode */ }
      document.dispatchEvent(new CustomEvent('labkit:theme', { detail: name }));
    },
    toggle() { this.set(this.get() === 'dark' ? 'light' : 'dark'); },
    restore() {
      let saved = null;
      try { saved = localStorage.getItem(THEME_KEY); } catch (e) { /* ignore */ }
      document.documentElement.setAttribute('data-theme', saved || 'dark');
    },
    /** A toggle button, for the right-hand end of a toolbar. */
    button() {
      const node = el('button', {
        class: 'ghost icon', title: 'Light / dark',
        onclick: () => { LK.theme.toggle(); paint(); },
      });
      const paint = () => { node.textContent = LK.theme.get() === 'dark' ? '☀' : '☾'; };
      paint();
      return node;
    },
  };
  LK.theme.restore();

  // ------------------------------------------------------------ browsing ---

  const RECENTS_KEY = 'labkit.recentFolders';
  function recents() {
    try { return JSON.parse(localStorage.getItem(RECENTS_KEY) || '[]'); } catch (e) { return []; }
  }
  function remember(path) {
    if (!path) return;
    const list = [path].concat(recents().filter((p) => p !== path)).slice(0, 8);
    try { localStorage.setItem(RECENTS_KEY, JSON.stringify(list)); } catch (e) { /* ignore */ }
  }
  LK.recentFolders = recents;

  function humanSize(bytes) {
    if (!bytes) return '';
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + ' MB';
    return Math.max(Math.round(bytes / 1024), 1) + ' KB';
  }

  /**
   * The file dialog, identical in every app.
   *
   *   mode      'dir' | 'file' | 'files'
   *   exts      ['.czi', '.tif']  -- which files are worth showing
   *   path      where to open
   *   endpoint  defaults to '/api/browse'
   *   title, hint
   *
   * Resolves to a path (or array of paths for 'files'), or null if canceled.
   *
   * Folders on the left, the files actually in the folder on the right.
   * Showing the files is the point: choosing a folder from a tree of names
   * alone means confirming blind, and the folder you want is usually
   * recognisable by what is in it rather than by what it is called.
   */
  LK.browse = function (options) {
    const opts = Object.assign({
      mode: 'dir', exts: null, path: '', endpoint: '/api/browse',
      title: '', hint: '', startAtRecent: true,
    }, options || {});
    if (!opts.path && opts.startAtRecent) opts.path = recents()[0] || '';

    return new Promise((resolve) => {
      let here = { path: opts.path, dirs: [], files: [] };
      const chosenFiles = new Set();
      let lastIndex = null;      // anchor for a shift-click run
      let chosenDir = '';

      // --- chrome ---------------------------------------------------------
      const crumbs = el('div', { class: 'lk-crumbs' });
      const pathBox = el('input', {
        type: 'text', class: 'lk-path', spellcheck: 'false',
        placeholder: '/path/to/folder',
        onkeydown: (e) => { if (e.key === 'Enter') go(pathBox.value); },
      });
      const dirsCol = el('div', { class: 'lk-dirs' });
      const filesCol = el('div', { class: 'lk-files' });
      const footNote = el('div', { class: 'dim', style: { flex: '1 1 auto', fontSize: '11.5px' } });
      const useButton = el('button', {
        class: 'primary',
        text: opts.mode === 'dir' ? 'Use this folder' : 'Open',
        onclick: () => finish(),
      });

      const filter = el('input', {
        type: 'search', placeholder: 'filter…', title: LK.FILTER_HINT,
        style: { flex: '1 1 auto', fontSize: '11.5px', padding: '3px 6px' },
        oninput: () => paint(),
      });

      // Select-all earns its place in a folder of 40 channels from one
      // acquisition: the alternative is 40 clicks to say "all of it".
      const selectAll = el('input', {
        type: 'checkbox',
        onchange: (e) => {
          chosenFiles.clear();
          if (e.target.checked) (here.files || []).forEach((f) => chosenFiles.add(f.path));
          paint();
        },
      });
      const selectAllLabel = el('label', { class: 'check', style: { fontSize: '11.5px' } },
        [selectAll, ' select all']);

      const recentRow = el('div', { class: 'lk-recents' });
      const box = el('div', { class: 'modal-box' }, [
        el('div', { class: 'modal-head' }, [
          el('h2', { text: opts.title || (opts.mode === 'dir' ? 'Choose a folder' : 'Choose a file') }),
          el('span', { class: 'spacer' }),
          el('button', { class: 'ghost small', text: 'Cancel', onclick: () => close(null) }),
        ]),
        el('div', { class: 'modal-head', style: { borderBottom: '1px solid var(--line)' } }, [
          el('button', { class: 'ghost small', text: '↑ Up', title: 'Parent folder',
            onclick: () => here.parent && go(here.parent) }),
          pathBox,
          el('button', { class: 'ghost small', text: 'Go', onclick: () => go(pathBox.value) }),
          el('button', { class: 'ghost small', text: '+ Folder',
            title: 'Create a subfolder of the folder shown', onclick: () => newFolder() }),
        ]),
        el('div', { class: 'lk-recents', style: { paddingTop: '8px' } }, [crumbs]),
        recentRow,
        el('div', { class: 'modal-body' }, [
          el('div', { class: 'lk-browser' }, [dirsCol, filesCol]),
        ]),
        el('div', { class: 'modal-foot' }, [
          opts.mode === 'files' ? selectAllLabel : null,
          footNote,
          useButton,
        ]),
      ]);
      const modal = el('div', { class: 'modal', onclick: (e) => { if (e.target === modal) close(null); } }, [box]);

      function close(result) {
        document.removeEventListener('keydown', onKey);
        modal.remove();
        resolve(result);
      }
      function onKey(e) { if (e.key === 'Escape') close(null); }
      document.addEventListener('keydown', onKey);

      function finish() {
        if (opts.mode === 'dir') {
          const target = chosenDir || here.path;
          remember(target);
          return close(target);
        }
        const picked = Array.from(chosenFiles);
        if (!picked.length) return;
        remember(here.path);
        close(opts.mode === 'files' ? picked : picked[0]);
      }

      /* Create a subfolder of the folder on screen.
         An input in the folder list, not prompt(): the names it must not
         collide with are the ones already listed beside it, and a modal on
         top of a modal hides them.  (It also keeps working in the embedded
         browsers that answer prompt() with null without asking anyone.)
         Anything that repaints the column -- a click, the filter -- drops the
         row, which is the same gesture as giving up on it. */
      function newFolder() {
        const open = dirsCol.querySelector('.lk-newname');
        if (open) return open.focus();
        const nameBox = el('input', {
          type: 'text', class: 'lk-newname', spellcheck: 'false',
          placeholder: 'name of the new folder',
          onkeydown: (e) => {
            if (e.key === 'Enter') create(nameBox.value);
            // Escape dismisses the row, not the whole dialog, so it must not
            // reach the document listener that closes it.
            if (e.key === 'Escape') { e.stopPropagation(); row.remove(); footer(); }
          },
        });
        const row = el('div', { class: 'lk-entry' }, [
          el('span', { class: 'lk-glyph', text: '▸' }), nameBox,
        ]);
        dirsCol.insertBefore(row, dirsCol.children[1] || null);
        nameBox.focus();
        footNote.style.color = '';
        footNote.textContent = 'Enter to create it in ' + here.path;
      }

      async function create(name) {
        let made;
        try {
          made = await LK.api(opts.endpoint + '/mkdir', { body: { path: here.path, name } });
        } catch (err) {
          made = { error: err.message };
        }
        if (made.error) {
          footNote.textContent = made.error;
          footNote.style.color = 'var(--bad)';
          return;
        }
        remember(made.path);
        go(made.path);        // land in it: the point was to put something here
      }

      async function go(path) {
        dirsCol.innerHTML = '';
        dirsCol.appendChild(el('div', { class: 'lk-empty', text: 'Reading…' }));
        const query = new URLSearchParams({ path: path || '' });
        if (opts.exts && opts.exts.length) query.set('exts', opts.exts.join(','));
        try {
          here = await LK.api(opts.endpoint + '?' + query.toString());
        } catch (err) {
          dirsCol.innerHTML = '';
          dirsCol.appendChild(el('div', { class: 'lk-empty', text: err.message }));
          return;
        }
        chosenDir = '';
        chosenFiles.clear();
        lastIndex = null;
        pathBox.value = here.path;
        paint();
      }

      function paint() {
        const needle = filter.value.trim().toLowerCase();
        const matches = LK.nameMatch(needle);

        // breadcrumbs -- clicking a component is faster than Up three times
        crumbs.innerHTML = '';
        const parts = String(here.path || '').split('/').filter(Boolean);
        let accumulated = '';
        crumbs.appendChild(el('button', { text: '/', onclick: () => go('/') }));
        parts.forEach((part, i) => {
          accumulated += '/' + part;
          const target = accumulated;
          if (i) crumbs.appendChild(el('span', { class: 'sep', text: '›' }));
          crumbs.appendChild(el('button', { text: part, onclick: () => go(target) }));
        });

        recentRow.innerHTML = '';
        (here.shortcuts || []).forEach((place) => {
          recentRow.appendChild(el('button', {
            class: 'ghost small', text: (place.glyph || '') + ' ' + place.label,
            title: place.path, onclick: () => go(place.path),
          }));
        });
        recents().slice(0, 3).forEach((path) => {
          if (path === here.path) return;
          recentRow.appendChild(el('button', {
            class: 'ghost small', title: path,
            text: '↺ ' + (path.split('/').filter(Boolean).pop() || path),
            onclick: () => go(path),
          }));
        });

        // folders
        dirsCol.innerHTML = '';
        dirsCol.appendChild(el('div', { class: 'lk-col-head' }, [
          el('span', { text: 'Folders' }),
          el('span', { class: 'spacer' }),
          filter,
        ]));
        const dirs = (here.dirs || []).filter((d) => matches(d.name));
        if (!dirs.length) {
          dirsCol.appendChild(el('div', { class: 'lk-empty', text: needle ? 'Nothing matches.' : 'No subfolders.' }));
        }
        dirs.forEach((dir) => {
          const row = el('div', {
            class: 'lk-entry' + (chosenDir === dir.path ? ' sel' : ''),
            ondblclick: () => go(dir.path),
            onclick: () => {
              if (opts.mode === 'dir') { chosenDir = dir.path; paint(); footer(); }
              else go(dir.path);
            },
          }, [
            el('span', { class: 'lk-glyph', text: '▸' }),
            el('span', { class: 'lk-name', text: dir.name, title: dir.path }),
            dir.n_files ? el('span', { class: 'lk-meta', text: dir.n_files }) : null,
          ]);
          dirsCol.appendChild(row);
        });

        // files
        filesCol.innerHTML = '';
        // The server may be filtering by the app's own list of formats even
        // when the caller passed none, so read the filter back off the listing.
        const shown = (opts.exts && opts.exts.length) ? opts.exts : (here.extensions || []);
        filesCol.appendChild(el('div', { class: 'lk-col-head' }, [
          el('span', { text: shown.length ? 'Files here' : 'Contents' }),
          el('span', { class: 'spacer' }),
          el('span', { text: String((here.files || []).length) }),
        ]));
        const files = (here.files || []).filter((f) => matches(f.name));
        if (!files.length) {
          filesCol.appendChild(el('div', {
            class: 'lk-empty',
            text: shown.length
              ? 'Nothing here matching ' + shown.slice(0, 6).join(', ')
                + (shown.length > 6 ? ', …' : '')
              : 'Empty.',
          }));
        }
        files.forEach((file, index) => {
          const row = el('div', {
            class: 'lk-entry' + (chosenFiles.has(file.path) ? ' sel' : ''),
            ondblclick: () => { if (opts.mode !== 'dir') { chosenFiles.add(file.path); finish(); } },
            onclick: (e) => {
              if (opts.mode === 'dir') return;
              if (e.shiftKey && opts.mode === 'files' && lastIndex != null) {
                const from = Math.min(lastIndex, index), to = Math.max(lastIndex, index);
                for (let i = from; i <= to; i++) chosenFiles.add(files[i].path);
              } else {
                if (opts.mode === 'file' || !(e.metaKey || e.ctrlKey)) chosenFiles.clear();
                if (chosenFiles.has(file.path)) chosenFiles.delete(file.path);
                else chosenFiles.add(file.path);
                lastIndex = index;
              }
              paint();
            },
          }, [
            el('span', { class: 'lk-glyph', text: '·' }),
            el('span', { class: 'lk-name', text: file.name, title: file.path }),
            el('span', { class: 'lk-meta', text: humanSize(file.size) }),
          ]);
          filesCol.appendChild(row);
        });
        if (here.truncated) {
          filesCol.appendChild(el('div', {
            class: 'lk-empty',
            text: '…and ' + here.truncated + ' more, not listed. Use the filter.',
          }));
        }
        footer();
      }

      function footer() {
        if (here.error) {
          footNote.textContent = here.error;
          footNote.style.color = 'var(--bad)';
        } else if (opts.mode === 'dir') {
          const target = chosenDir || here.path;
          const n = chosenDir
            ? (here.dirs.find((d) => d.path === chosenDir) || {}).n_files
            : (here.files || []).length;
          footNote.style.color = '';
          footNote.textContent = target + (n ? '  ·  ' + n + ' file(s)' : '');
          useButton.disabled = false;
        } else {
          footNote.style.color = '';
          footNote.textContent = chosenFiles.size
            ? chosenFiles.size + ' selected'
            : (opts.hint || 'Pick a file' + (opts.mode === 'files' ? ' — ⌘-click for several, shift for a run' : ''));
          useButton.disabled = !chosenFiles.size;
          const all = (here.files || []).length;
          selectAll.checked = all > 0 && chosenFiles.size === all;
          selectAll.indeterminate = chosenFiles.size > 0 && chosenFiles.size < all;
        }
      }

      document.body.appendChild(modal);
      go(opts.path);
    });
  };

  /**
   * A path box with a Browse button beside it, for a side panel that wants a
   * folder without a dialog of its own.  Returns { value, set, input }.
   */
  LK.pathPicker = function (host, options) {
    const opts = Object.assign({ mode: 'dir', exts: null, placeholder: '', value: '',
      endpoint: '/api/browse', onchange: null }, options || {});
    const input = el('input', {
      type: 'text', spellcheck: 'false',
      placeholder: opts.placeholder || (opts.mode === 'dir' ? 'folder…' : 'file…'),
      value: opts.value,
    });
    input.addEventListener('change', () => opts.onchange && opts.onchange(input.value));
    const wrap = el('div', { class: 'lk-pathpick' }, [
      input,
      el('button', {
        class: 'ghost small', text: 'Browse…',
        onclick: async () => {
          const picked = await LK.browse({
            mode: opts.mode, exts: opts.exts, endpoint: opts.endpoint,
            path: input.value,
          });
          if (picked) {
            input.value = Array.isArray(picked) ? picked[0] : picked;
            opts.onchange && opts.onchange(input.value);
          }
        },
      }),
    ]);
    if (host) host.appendChild(wrap);
    return {
      node: wrap, input,
      get value() { return input.value; },
      set(value) { input.value = value; },
    };
  };

  /* The hex code of a color well, in a box beside it: on screen without
     opening the picker, and typed or pasted where the browser's own picker
     opens on RGB.  `#12A396`, `12a396` and `#1a9` are all read; anything else
     puts the well's code back.  The well stays the one value -- the box
     follows it however it is set, from the picker or from code. */
  LK.hexBox = function (input) {
    const box = el('input', { type: 'text', class: 'lk-hex', spellcheck: 'false',
      maxlength: '7', title: 'hex color — type or paste one, e.g. #12A396' });
    const show = () => { box.value = input.value.toUpperCase(); };
    const own = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
    Object.defineProperty(input, 'value', {
      configurable: true,
      get() { return own.get.call(input); },
      set(value) { own.set.call(input, value); show(); },
    });
    input.addEventListener('input', show);
    input.addEventListener('change', show);
    // Typing is not a choice yet, so it stays in the box: a form listening
    // further up would redraw from a half-written code.
    box.addEventListener('input', (e) => e.stopPropagation());
    box.addEventListener('change', (e) => {
      e.stopPropagation();
      const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(box.value.trim());
      if (!m) { show(); return; }
      const hex = m[1].length === 3 ? m[1].replace(/./g, '$&$&') : m[1];
      if ('#' + hex.toLowerCase() === input.value) { show(); return; }
      input.value = '#' + hex;
      // What picking it in the well would have sent.
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    show();
    return box;
  };

  // ------------------------------------------------------- plot controls ---

  /* "min, max" as a pair, or null for anything else -- the same reading as
     labkit.plots._limits, so an app's own canvas preview and the figure it
     exports crop to the same window instead of disagreeing on screen. */
  LK.limits = function (text) {
    const values = String(text || '').split(/[,\s]+/)
      .filter((part) => part !== '').map(Number).filter(Number.isFinite);
    return values.length >= 2 ? [values[0], values[1]] : null;
  };

  /* labkit.plots.PALETTES, so an app's own canvas draws a group in the color
     the exported figure will give it. Gray first and red second is the point:
     control against mutant needs no legend. */
  LK.PALETTES = {
    'red-grey': ['#5f6772', '#c62f34', '#2c7fb8', '#c8935f',
                 '#7d5ba6', '#3f8f6b', '#1f2933', '#d1495b'],
    greys: ['#2b2f36', '#5f6772', '#8d95a0', '#b7bdc5', '#d9dde1'],
    reds: ['#7a1c20', '#a8272c', '#c62f34', '#dd6266', '#eb9c9e'],
    colourblind: ['#5f6772', '#c62f34', '#0072b2', '#e69f00',
                  '#009e73', '#cc79a7', '#56b4e9', '#111111'],
  };

  /* theme.PRINT.ink. Not pure black: 100 % K beside a gray series reads as a
     hole on paper. */
  LK.INK = '#1f2933';

  /** The color a palette gives the *index*-th group, cycling. */
  LK.seriesColour = function (index, palette) {
    const wheel = LK.PALETTES[palette] || LK.PALETTES['red-grey'];
    return wheel[((index % wheel.length) + wheel.length) % wheel.length];
  };

  const PALETTE_SWATCHES = {};   // four is enough to tell them apart in a row
  Object.keys(LK.PALETTES).forEach((key) => {
    PALETTE_SWATCHES[key] = LK.PALETTES[key].slice(0, 4);
  });

  /* What the journals actually ask for, from each publisher's own figure
     guidelines, read 2026-09-29. The `source` is kept beside the numbers so a
     figure that has moved can be checked rather than guessed at.

     The widths are printed column widths. Setting the *figure* to one of them
     is what makes a panel arrive at the size it will be printed, which is the
     only size at which "is this label readable" has an answer -- a panel drawn
     at 120 mm and dropped into an 89 mm column loses a quarter of its type. */
  LK.JOURNALS = {
    nature: {
      label: 'Springer Nature',
      widths: { '1 column': 89, '1\u00bd columns': 128, '2 columns': 183 },
      font_pt: 7, dpi: 300, max_height_mm: 247,
      note: 'Text 5\u20137 pt, panel letters 8 pt bold, lines 0.25\u20131 pt, 300 dpi. '
        + 'Page depth 247 mm. The 1\u00bd column is a 120\u2013136 mm range; 128 is its middle.',
      source: 'nature.com/nature/for-authors/final-submission',
    },
    cell: {
      label: 'Cell Press',
      widths: { '1 column': 85, '1\u00bd columns': 114, '2 columns': 174 },
      font_pt: 7, dpi: 300, max_height_mm: 200,
      note: 'Arial only, text 6\u20138 pt, lines 0.5\u20131.5 pt. 300 dpi color, '
        + '1000 dpi line art. Whole figure within 165 \u00d7 200 mm.',
      source: 'cell.com/information-for-authors/figure-guidelines',
    },
    science: {
      label: 'Science (AAAS)',
      widths: { '1 column': 57, '2 columns': 121, '3 columns': 184 },
      font_pt: 7, dpi: 300,
      note: 'Three-column page. Helvetica, axis labels 6\u20139 pt, nothing under 5 pt, '
        + 'panel letters 10 pt bold. Their own page is behind a bot check, so these '
        + 'come from the AAAS figure preparation guide \u2014 worth confirming.',
      source: 'science.org author figure prep guide',
    },
    embo: {
      label: 'EMBO Press',
      widths: { '1 column': 87, '2 columns': 180 },
      font_pt: 8, dpi: 300,
      note: 'Helvetica, panel letters bold; lettering kept on its own layer so '
        + 'production can restyle it \u2014 which needs live text, not outlines. '
        + 'EMBO publishes the 87\u2013180 mm range but no text size; 8 pt is a safe one.',
      source: 'link.springer.com/journal/44318/submission-guidelines',
    },
    pnas: {
      label: 'PNAS',
      widths: { '1 column': 87, '1\u00bd columns': 114, '2 columns': 178 },
      font_pt: 7, dpi: 300,
      note: 'Arial or Helvetica, text 6\u20138 pt and never under 6 pt after reduction.',
      source: 'pnas.org digital art guidelines',
    },
    jgp: {
      label: 'JGP / Rockefeller',
      widths: { '1 column': 85 },
      font_pt: 8, dpi: 300,
      note: 'Arial, 8 pt text (5 pt sub- and superscript), lines 0.5\u20131 pt, '
        + '300 dpi (600 where a photo and labels share a panel). Only the single '
        + 'column is published: they reduce figures to 85 mm wherever they can.',
      source: 'rupress.org/jgp/pages/fig-vid-guidelines',
    },
  };

  /* What a preset sets on top of the journal's own numbers.

     Journals publish a column width, a type size and a line-weight range.
     Nobody publishes how tall a panel should be, so it falls to the default
     -- 0.72 of the width -- which makes a two-column figure 135 mm tall: one
     panel over half of Nature's 247 mm page. These two are the house answer,
     and both stay editable like everything else a preset writes.

     The pinned number is the *data area*, not the canvas, so a panel is the
     size asked for whatever the labels turn out to be, and the canvas grows
     round it by the 10 mm the ticks and the axis label need. Two panels set by
     the same preset then match on the page, which is the whole point of
     pinning it. It goes on the axis the *values* are on -- the height of an
     upright panel, the width of a turned one -- because that is the axis a bar
     is read along and the one bar_mm is not already sizing.

     The width is not set at all. It is calculated, from one bar: 3.94 mm each,
     so a bar is the same size on the page whether a panel holds two of them
     or seven -- which is what a reader compares across a figure, and what
     goes wrong the moment a panel is stretched to fill its column instead.

     3.94 mm makes a category 6.35 mm wide -- 0.25 in, the pitch the lab's
     own ephys notebook draws at (figsize=(0.25*18, 1.2)), so a panel from
     here and a panel from that notebook sit side by side at one scale. It
     also clears the (n = 5) printed under a group name, which at 7 pt is
     5.33 mm. Names past about four characters -- R201C, rescue -- still want
     the x rotation or the horizontal box; no width fits them and stays this
     tight.
     The column width becomes a ceiling: past about twenty groups the bars give
     way rather than the figure running off the page.

     32 mm is the compact end of what still reads: a single column comes out
     43 mm tall, five rows to a page, and a p bracket still clears the tallest
     point. Drawn down the ladder it is 28 mm where the bracket starts sitting
     on the data, so this is one step back from the edge rather than a guess.
     A two-column figure keeps the same 32 mm and so comes out a long strip --
     which is right, because a figure that wide is meant to be a row of panels
     and not one panel stretched across the page.

     0.75 pt is inside every line-weight range these journals publish (Nature
     0.25-1, Cell 0.5-1.5, JGP 0.5-1) and under matplotlib's 0.8 default, so
     the frame stops being heavier than the trace inside it. */
  const JOURNAL_PANEL = { bar_mm: 3.5, axes_mm: 32, line_width: 0.75 };

  /** Mirrors labkit.plots.PlotSpec -- keep the two in step. */
  /* `names` in the sequence `order` asks for, keeping the strangers.
   *
   * The twin of labkit.plots.order_series, and the same reasoning: the order a
   * reader needs is almost never alphabetical -- WT, P716L, Y715C is the order
   * along the protein, and sorting puts the wild type last. A name the order
   * does not mention keeps its place after the ones it does, so a saved order
   * outlives the dataset it was typed for and a new condition appears at the
   * end instead of vanishing. */
  LK.orderSeries = function (names, order) {
    const have = (names || []).map(String);
    if (!order || !order.length) return have;
    const seen = new Set();
    const first = [];
    order.map(String).forEach((name) => {
      if (have.includes(name) && !seen.has(name)) { seen.add(name); first.push(name); }
    });
    return first.concat(have.filter((n) => !seen.has(n)));
  };

  LK.defaultSpec = function () {
    return {
      kind: 'bar', centre: 'mean', error: 'sem', points: 'jitter',
      palette: 'red-grey', colours: {}, order: [],
      width_mm: 85, height_mm: 0, axes_width_mm: 0, axes_height_mm: 0, bar_mm: 3.5,
      horizontal: false,
      font_pt: 8, font: 'arial',
      log_y: false, show_n: true, legend: true,
      title: '', ylabel: '', xlabel: '',
      show_p: true, p_style: 'auto', p_test: 'mann-whitney',
      p_correction: 'fdr_bh', p_pairs: 'all', p_only_significant: false,
      p_use_corrected: true, p_align: 'stagger', p_alpha: 0.05, p_pair_on: '',
      size_unit: 'mm', xlim: '', ylim: '', xticks: '', yticks: '',
      xtick_labels: '', ytick_labels: '', xtick_rotation: 0, ytick_rotation: 0,
      tick_direction: 'out', minor_ticks: false, grid: 'none', spines: 'auto',
      log_x: false, ref_lines: '', legend_loc: 'best', dpi: 300,
      error_style: 'band', markers: 'o', marker_size: 2.6, line_width: 1.0,
      line_styles: '-', join: true, point_size: 3.2,
      panel_width_mm: 25.4, panel_height_mm: 38.1, panel_cols: 0,   // 1 x 1.5 in
      scale_x_s: 0, scale_y: 0, family_spread: false, family_colour: 'voltage',
    };
  };

  function selectRow(label, id, choices, title) {
    const select = el('select', { id });
    choices.forEach(([value, text, hint]) =>
      select.appendChild(el('option', { value, text, title: hint || '' })));
    return { row: el('label', { class: 'row', title: title || '' },
      [el('span', { text: label }), select]), field: select };
  }

  function numberRow(label, id, attrs, title) {
    const input = el('input', Object.assign({ type: 'number', id }, attrs));
    return { row: el('label', { class: 'row', title: title || '' },
      [el('span', { text: label }), input]), field: input };
  }

  function textRow(label, id, placeholder, title) {
    const input = el('input', { type: 'text', id, placeholder: placeholder || '' });
    return { row: el('label', { class: 'row', title: title || '' },
      [el('span', { text: label }), input]), field: input };
  }

  function checkRow(label, id, checked, title) {
    const input = el('input', { type: 'checkbox', id, checked: checked || null });
    return { row: el('label', { class: 'row check', title: title || '' },
      [el('span', {}, [input, ' ' + label])]), field: input };
  }

  /**
   * The plotting panel, built into `host`.
   *
   * Every app gets the same controls in the same order, and `spec()` returns
   * exactly what labkit.plots.PlotSpec expects, so a control added here shows
   * up everywhere and means the same thing.
   *
   *   opts.groups   labels available to the "compare against" control
   *   opts.onchange fired on any edit (for a live preview)
   *   opts.extra    a node to drop in above the p-value block
   *   opts.kinds    restrict the plot kinds offered
   */
  LK.PlotControls = function (host, options) {
    const opts = Object.assign({ groups: [], onchange: null, extra: null,
      kinds: null, spec: null }, options || {});
    const state = Object.assign(LK.defaultSpec(), opts.spec || {});
    const fields = {};
    const root = el('div', { class: 'plotctl' });

    const allKinds = [
      ['bar', 'Bars', 'Group means. Degrades to "points only" under n = 3.'],
      ['box', 'Box', 'Median, quartiles, whiskers.'],
      ['violin', 'Violin', 'The whole distribution. Needs n of about 8 to mean anything.'],
      ['none', 'Points only', 'The center as a line, nothing else claimed.'],
    ];
    const kinds = opts.kinds ? allKinds.filter((k) => opts.kinds.includes(k[0])) : allKinds;

    const rows = [
      ['kind', selectRow('Plot', 'lk-kind', kinds, 'What the summary shape is')],
      ['centre', selectRow('Center', 'lk-centre',
        [['mean', 'Mean'], ['median', 'Median']],
        'Median is the honest one for skewed data or small n')],
      ['error', selectRow('Error bars', 'lk-error',
        [['sem', 'SEM'], ['sd', 'SD'], ['ci', '95% CI'], ['none', 'None']],
        'SEM describes the mean, SD describes the data. Say which in the legend.')],
      ['points', selectRow('Points', 'lk-points',
        [['jitter', 'Jittered'], ['swarm', 'Beeswarm'], ['strip', 'In a line'], ['', 'Hidden']],
        'Every data point on top of the summary. Hiding them is rarely the right call.')],
      ['palette', selectRow('Colors', 'lk-palette',
        [['red-grey', 'Red / gray'], ['greys', 'Grays'], ['reds', 'Reds'],
         ['colourblind', 'Color-blind safe']], 'Gray first, red second')],
      ['font', selectRow('Font', 'lk-font',
        [['arial', 'Arial'], ['pt-sans', 'PT Sans'], ['pt-mono', 'PT Mono']],
        'Arial is what the journals ask for. PT Sans is the one the app itself '
        + 'is set in, for a figure going next to it in a talk; PT Mono is for '
        + 'labels that are sequences, well numbers or filenames.')],
    ];
    rows.forEach(([key, built]) => { fields[key] = built.field; root.appendChild(built.row); });

    const swatches = el('div', { class: 'swatches' });
    /* The way back out of a typed order, which is not the same as alphabetical:
       it hands the sequence back to whatever the data arrived in. */
    const orderReset = el('button', { type: 'button', class: 'small ghost', hidden: true,
      text: 'Default order',
      title: 'Forget the order typed here and draw the conditions in the order '
        + 'the data arrived in' });
    root.appendChild(el('label', { class: 'row' },
      [el('span', { text: '' }), swatches]));
    root.appendChild(el('label', { class: 'row' },
      [el('span', { text: '' }), orderReset]));

    // Saved schemes. Hidden unless the app serves the store, so an app
    // without labkit.schemes wired up shows no button that cannot work.
    const schemeField = el('select', { style: { flex: '1 1 auto', width: 'auto' } });
    const saveScheme = el('button', { type: 'button', class: 'small', text: 'Save' });
    const dropScheme = el('button', { type: 'button', class: 'small ghost', text: '\u00d7',
      title: 'Delete the saved scheme' });
    const schemeRow = el('div', { class: 'row', hidden: true,
      title: 'A saved set of colors, by series name — shared with the other apps, '
        + 'so one decision about what the KO looks like covers the whole paper' },
      [el('span', { text: 'Scheme' }),
       el('div', { class: 'pair' }, [schemeField, saveScheme, dropScheme])]);
    root.appendChild(schemeRow);

    /* Journal presets. A menu, not a setting: picking one writes the width,
       the type size, the font and the dpi into the boxes below, where they can
       then be edited, along with a panel height and a line weight of our own
       (see JOURNAL_PANEL). Nothing remembers which journal it was, because the
       moment one of those boxes is touched the label would be a lie about the
       figure. What the journal asks for beyond those four numbers -- minimum
       type, line weights, page depth -- is printed underneath, since it is the
       part no control can enforce. */
    const journalField = el('select', { style: { flex: '1 1 auto', width: 'auto' } });
    const columnsField = el('select', { style: { flex: '0 1 auto', width: 'auto' } });
    const journalNote = el('div', { class: 'hint', hidden: true });
    journalField.appendChild(el('option', { value: '', text: '\u2014' }));
    Object.keys(LK.JOURNALS).forEach((key) => journalField.appendChild(
      el('option', { value: key, text: LK.JOURNALS[key].label })));
    root.appendChild(el('label', { class: 'row',
      title: 'Sets the figure to a printed column width, with the type size, font and '
        + 'resolution that journal asks for. Everything stays editable afterwards. '
        + 'The point size sets a whole 7/8/9 ladder, so tick labels come out a little '
        + 'under the number and a panel title a little over it \u2014 which is one more '
        + 'reason a title belongs in the caption. It also pins a compact plotting area '
        + 'and a thinner line, so the figure is the shape a journal prints rather than '
        + 'half a page tall; both are in the boxes below and both can be changed.' },
      [el('span', { text: 'Journal' }),
       el('div', { class: 'pair' }, [journalField, columnsField])]));
    root.appendChild(journalNote);

    function applyJournal(resetWidth) {
      const journal = LK.JOURNALS[journalField.value];
      journalNote.hidden = !journal;
      if (!journal) { columnsField.innerHTML = ''; return; }
      const names = Object.keys(journal.widths);
      if (resetWidth || !names.includes(columnsField.value)) {
        columnsField.innerHTML = '';
        names.forEach((name) => columnsField.appendChild(
          el('option', { value: name, text: name })));
      }
      state.width_mm = journal.widths[columnsField.value] || journal.widths[names[0]];
      state.height_mm = 0;                 // the pinned axes size sets it
      state.bar_mm = JOURNAL_PANEL.bar_mm;
      // The pin goes on the value axis and bar_mm sizes the other one, so a
      // turned panel gets the same two numbers the other way round rather than
      // a 32 mm pin on the axis the groups sit on, which would squash every
      // bar to fit it.
      state.axes_width_mm = state.horizontal ? JOURNAL_PANEL.axes_mm : 0;
      state.axes_height_mm = state.horizontal ? 0 : JOURNAL_PANEL.axes_mm;
      state.line_width = JOURNAL_PANEL.line_width;
      state.font_pt = journal.font_pt;
      state.font = 'arial';
      state.dpi = journal.dpi;
      journalNote.textContent = journal.note + ' \u2014 ' + journal.source;
      show(fields.font, state.font);
      show(fields.line_width, state.line_width);
      fontField.value = state.font_pt;
      writeSize();
      refresh();
    }
    journalField.addEventListener('change', () => applyJournal(true));
    columnsField.addEventListener('change', () => applyJournal(false));

    // Size. Stored in millimeters whatever the boxes are typed in, so a
    // figure specified in inches and one in mm are the same figure.
    const size = el('div', { class: 'pair' });
    const widthField = el('input', { type: 'number', min: '0.5', step: '0.1' });
    const heightField = el('input', { type: 'number', min: '0', step: '0.1',
      title: 'Height. 0 lets it follow the width \u2014 or, on a horizontal panel, '
        + 'lets the figure grow down the page as groups are added, which is the '
        + 'reason to turn it. Type one to cap that.' });
    const unitField = el('select', {});
    [['mm', 'mm'], ['in', 'in']].forEach(([value, text]) =>
      unitField.appendChild(el('option', { value, text })));
    const fontField = el('input', { type: 'number', min: '5', max: '18', step: '0.5' });
    size.appendChild(el('label', {}, [el('span', { class: 'dim', text: 'w' }), widthField]));
    size.appendChild(el('label', {}, [el('span', { class: 'dim', text: 'h' }), heightField]));
    size.appendChild(unitField);
    size.appendChild(el('label', {}, [el('span', { class: 'dim', text: 'pt' }), fontField]));
    fields.font_pt = fontField;
    root.appendChild(el('label', { class: 'row', title: 'The whole canvas, labels included, and the base font size in points — 85 mm (3.35 in) is one journal column. Leave it to the axes size below to have the canvas grow around the plot instead.' },
      [el('span', { text: 'Figure' }), size]));

    /* The plotting area itself. matplotlib only sizes the canvas, and the
       title and tick labels then eat into it from a fixed number of inches
       away -- so the same figure width buys a different amount of plot every
       time the labels change, and two panels meant to sit side by side on a
       page do not match. Set these and the canvas grows around them instead. */
    const axesSize = el('div', { class: 'pair' });
    const axesWField = el('input', { type: 'number', min: '0', step: '0.1' });
    const axesHField = el('input', { type: 'number', min: '0', step: '0.1' });
    const barField = el('input', { type: 'number', min: '0', step: '0.5',
      title: 'The printed thickness of one bar, box or violin. The panel is sized '
        + 'from it and the number of groups, so a bar stays the same size on the page '
        + 'as groups come and go, and the figure size becomes a ceiling rather than '
        + 'the size. Across the panel upright, down it when horizontal. 3.5 mm is the '
        + 'tightest that still clears an (n = 5) under the group name; names longer '
        + 'than about four characters want 5 mm, a rotation, or the horizontal box. '
        + 'Ignored if that axes box is typed in. 0 hands the size back to the figure.' });
    axesSize.appendChild(el('label', {}, [el('span', { class: 'dim', text: 'w' }), axesWField]));
    axesSize.appendChild(el('label', {}, [el('span', { class: 'dim', text: 'h' }), axesHField]));
    axesSize.appendChild(el('label', {}, [el('span', { class: 'dim', text: 'bar' }), barField]));
    root.appendChild(el('label', { class: 'row', title: 'One panel\u2019s plotting area, in the unit chosen above. The title, the axis labels and the ticks are added outside it, so two figures set to the same number have the same-sized data area whatever their titles say. 0 leaves the size to the figure \u2014 or, on the axis the groups sit on, to the bar box, which sizes the panel from the number of groups. Turning the panel swaps w and h with it.' },
      [el('span', { text: 'Axes' }), axesSize]));

    [['horizontal', 'Horizontal',
      'Turn the panel a quarter turn: groups down the left, bars reaching '
      + 'right, first group at the top. The sizing turns with it \u2014 the bar '
      + 'box still sets one bar\u2019s thickness and the axes box now pins the '
      + 'width. Worth it for long group names, which get a whole column to '
      + 'themselves instead of 6.35 mm of tick.'],
     ['show_n', 'n on the axis', 'Put (n = …) beside each group label'],
     ['log_y', 'Log value axis', 'The values, wherever they are: x on a turned panel.'],
     ['legend', 'Legend', '']].forEach(([key, label, title]) => {
      const built = checkRow(label, 'lk-' + key, state[key], title);
      fields[key] = built.field; root.appendChild(built.row);
    });

    /* Turning a panel turns its geometry with it: whichever axes box pinned
       the value axis moves to the axis the values are now on, and bar_mm sizes
       the one the groups are on. Without this the 32 mm pin would land on the
       group axis and every bar would be squashed to fit it.

       On the checkbox rather than on `root`, so it runs before the delegated
       change handler reads the boxes back -- the other way round, the figure
       would be redrawn once from the un-turned numbers and then sit there. */
    fields.horizontal.addEventListener('change', () => {
      const width = state.axes_width_mm;
      state.axes_width_mm = state.axes_height_mm;
      state.axes_height_mm = width;
      writeSize();
    });

    function section(title, rows) {
      const box = el('details', { class: 'more' });
      box.appendChild(el('summary', { text: title }));
      rows.forEach(([key, built]) => { fields[key] = built.field; box.appendChild(built.row); });
      root.appendChild(box);
      return box;
    }

    // --- labels ------------------------------------------------------------
    // Filled in from the data by the host; typing here replaces them.
    section('Labels', [
      ['title', textRow('Title', 'lk-title', '', 'Usually better in the caption than on the axes')],
      ['xlabel', textRow('x label', 'lk-xlabel', 'Voltage (mV)', '')],
      ['ylabel', textRow('y label', 'lk-ylabel', 'Current density (pA/pF)', '')],
    ]);

    // --- axes and curves -------------------------------------------------
    // Folded away: the defaults are publication-ready, and these are for the
    // figure that has to match one already drawn.
    section('Axes', [
      ['xlim', textRow('x limits', 'lk-xlim', 'min, max', 'Empty leaves the range to the data')],
      ['ylim', textRow('y limits', 'lk-ylim', 'min, max', 'Empty leaves the range to the data')],
      ['xticks', textRow('x ticks', 'lk-xticks', '0, 20, 40  ·  step 20',
        'Where the ticks go: a list, or "step 20" for regular ones')],
      ['xtick_labels', textRow('x tick text', 'lk-xticklabels', 'WT, KO',
        'Replaces the tick numbers, in order. Ignored unless it matches the number of ticks.')],
      ['xtick_rotation', numberRow('x rotation', 'lk-xrot', { min: '-90', max: '90', step: '15' }, 'Degrees')],
      ['yticks', textRow('y ticks', 'lk-yticks', '0, 50, 100  ·  step 50', '')],
      ['ytick_labels', textRow('y tick text', 'lk-yticklabels', '', '')],
      ['ytick_rotation', numberRow('y rotation', 'lk-yrot', { min: '-90', max: '90', step: '15' }, 'Degrees')],
      ['tick_direction', selectRow('Ticks point', 'lk-tickdir',
        [['out', 'Outwards'], ['in', 'Inwards'], ['inout', 'Both ways']], '')],
      ['minor_ticks', checkRow('Minor ticks', 'lk-minorticks', false, '')],
      ['grid', selectRow('Grid', 'lk-grid',
        [['none', 'None'], ['y', 'Horizontal'], ['x', 'Vertical'], ['both', 'Both']],
        'A grid competes with the data; none is usually right')],
      ['spines', selectRow('Axis lines', 'lk-spines',
        [['auto', 'As the plot kind wants'], ['lb', 'Left and bottom'], ['box', 'Full box'],
         ['zero', 'Crossing at zero'], ['none', 'None (scale bars)']],
        'Crossing at zero is the IV convention: the origin is a real place and the reversal potential is read off it')],
      ['log_x', checkRow('Log x axis', 'lk-logx', false, '')],
      ['ref_lines', textRow('Reference lines', 'lk-reflines', '0, -30',
        'Dotted horizontal lines, e.g. a control level')],
      ['legend_loc', selectRow('Legend at', 'lk-legendloc',
        [['best', 'Wherever it fits'], ['upper right', 'Top right'], ['upper left', 'Top left'],
         ['lower right', 'Bottom right'], ['lower left', 'Bottom left'],
         ['center right', 'Right'], ['center left', 'Left']], '')],
      ['dpi', numberRow('Export dpi', 'lk-dpi', { min: '72', max: '1200', step: '50' },
        'For the PNG; the SVG is vector whatever this says')],
    ]);

    section('Curves', [
      ['error_style', selectRow('Spread as', 'lk-errorstyle',
        [['band', 'Shaded band'], ['bars', 'Error bars'], ['both', 'Both'], ['none', 'Nothing']],
        'What the SEM (or SD, or CI) looks like on a curve')],
      ['markers', textRow('Markers', 'lk-markers', 'o, s, ^',
        'Cycled across the curves, so they stay apart in print: o s ^ v D < > p * x +. Empty draws none.')],
      ['marker_size', numberRow('Marker size', 'lk-markersize', { min: '0', max: '12', step: '0.2' }, '')],
      ['line_styles', textRow('Line styles', 'lk-linestyles', '-, --, :',
        'Cycled across the curves: - -- -. :')],
      ['line_width', numberRow('Line width', 'lk-linewidth', { min: '0.2', max: '4', step: '0.1' }, '')],
      ['join', checkRow('Join the points', 'lk-join', true, 'Off leaves the markers alone')],
      ['point_size', numberRow('Point size', 'lk-pointsize', { min: '0.5', max: '10', step: '0.2' },
        'The dots on a bar or box plot')],
    ]);

    // Step families: one panel per group, every step overlaid. Panel size is
    // typed in the unit chosen under Size, like the figure itself.
    const panelWField = el('input', { type: 'number', min: '0.2', step: '0.1' });
    const panelHField = el('input', { type: 'number', min: '0.2', step: '0.1' });
    const familyBox = section('Step families', [
      ['family_colour', selectRow('Color by', 'lk-familycolour',
        [['voltage', 'Step voltage'], ['black', 'Black only']],
        'A ramp across the steps, black, or one color per group — '
        + 'the grouping fields appear here once something is plotted')],
      ['panel_cols', numberRow('Panels per row', 'lk-panelcols', { min: '0', max: '8', step: '1' },
        '0 picks a near-square grid')],
      ['scale_x_s', numberRow('Scale bar, time (s)', 'lk-scalex', { min: '0', step: '0.05' },
        '0 picks a round length')],
      ['scale_y', numberRow('Scale bar, current', 'lk-scaley', { min: '0', step: '5' },
        'In the plotted unit, e.g. 50 for 50 pA/pF. 0 picks a round length.')],
      ['family_spread', checkRow("Shade each step's SEM", 'lk-familyspread', false,
        'Off by default: fourteen overlapping bands hide the currents they describe')],
    ]);
    familyBox.insertBefore(
      el('label', { class: 'row', title: 'One panel, in the unit chosen under Size — your old subplotsize' },
        [el('span', { text: 'Panel size' }), el('div', { class: 'pair' }, [
          el('label', {}, [el('span', { class: 'dim', text: 'w' }), panelWField]),
          el('label', {}, [el('span', { class: 'dim', text: 'h' }), panelHField])])]),
      familyBox.children[1]);   // straight after <summary>, above "Color by"

    if (opts.extra) root.appendChild(opts.extra);

    // --- p-values -------------------------------------------------------
    const pBlock = el('div', { class: 'pblock' });
    const pOn = el('input', { type: 'checkbox', checked: true });
    fields.show_p = pOn;
    pBlock.appendChild(el('label', { class: 'phead' }, [pOn, ' p-values on the plot']));

    const pRows = [
      ['p_style', selectRow('Show as', 'lk-pstyle',
        [['auto', 'p = 0.031'], ['exact', 'p = 0.031 (always)'],
         ['stars', '✱ ✱✱ ✱✱✱'], ['both', 'p = 0.031 ✱']],
        'A number is the honest form: with four animals a side, p = 0.04 and p = 0.06 are the same result')],
      ['p_test', selectRow('Test', 'lk-ptest',
        [['mann-whitney', 'Mann-Whitney'], ['welch', 'Welch t'],
         ['student', 'Student t'], ['wilcoxon', 'Wilcoxon (paired)'],
         ['paired-t', 'Paired t']],
        'Rank tests make no normality assumption; the paired ones need a pairing column')],
      ['p_correction', selectRow('Correct', 'lk-pcorr',
        [['fdr_bh', 'FDR (Benjamini–Hochberg)'], ['holm', 'Holm'],
         ['bonferroni', 'Bonferroni'], ['none', 'None']],
        'Across every comparison drawn on this figure')],
      ['p_pairs', selectRow('Compare', 'lk-ppairs',
        [['all', 'Every pair'], ['adjacent', 'Neighbors only']],
        'Which pairs get a bracket')],
      ['p_align', selectRow('Brackets', 'lk-palign',
        [['stagger', 'Close to the data'], ['level', 'All at one height']], '')],
    ];
    pRows.forEach(([key, built]) => { fields[key] = built.field; pBlock.appendChild(built.row); });

    // Everything against one control is the common design, so offer the
    // group labels themselves as a "Compare" option once they are known.
    const pairsField = fields.p_pairs;

    [['p_only_significant', 'Only significant ones', 'Hide the n.s. brackets entirely'],
     ['p_use_corrected', 'Use corrected p', 'Report the multiplicity-corrected value']]
      .forEach(([key, label, title]) => {
        const built = checkRow(label, 'lk-' + key, state[key], title);
        fields[key] = built.field; pBlock.appendChild(built.row);
      });

    const pairOn = el('input', { type: 'text', placeholder: 'cell_id', style: { flex: '0 0 122px' } });
    fields.p_pair_on = pairOn;
    const pairOnRow = el('label', { class: 'row', title: 'The column identifying what is paired. A paired test without it is not paired.' },
      [el('span', { text: 'Paired by' }), pairOn]);
    pBlock.appendChild(pairOnRow);

    const preview = el('div', { class: 'preview' });
    pBlock.appendChild(preview);
    root.appendChild(pBlock);

    /* Reading the widgets and telling the caller about it are deliberately
       two functions.  They were one, and `spec()` called it -- so a host whose
       onchange redrew the figure got: change -> onchange -> read the spec ->
       onchange -> ... Each turn of that fired another request, and the browser
       gave up with ERR_INSUFFICIENT_RESOURCES. `spec()` reads; only a real
       edit notifies. */
    // Size is kept in millimeters and shown in whichever unit is chosen, so
    // switching the unit re-labels the same figure rather than resizing it.
    const MM_PER_IN = 25.4;
    let shownUnit = state.size_unit || 'mm';

    function writeSize() {
      const factor = (state.size_unit === 'in') ? MM_PER_IN : 1;
      // Three decimals in inches, two in mm: both are a thousandth of the
      // figure, and at two an inch is only quarter-millimeter resolution --
      // enough that switching the unit and switching back used to move a
      // 6 mm bar to 6.1 and a 114 mm column to 114.05.
      const places = (factor === 1) ? 100 : 1000;
      const round = (v) => Math.round(v * places) / places;
      widthField.value = round((state.width_mm || 0) / factor);
      heightField.value = state.height_mm ? round(state.height_mm / factor) : 0;
      panelWField.value = round((state.panel_width_mm || 25.4) / factor);
      panelHField.value = round((state.panel_height_mm || 38.1) / factor);
      axesWField.value = state.axes_width_mm ? round(state.axes_width_mm / factor) : 0;
      axesHField.value = state.axes_height_mm ? round(state.axes_height_mm / factor) : 0;
      barField.value = state.bar_mm ? round(state.bar_mm / factor) : 0;
      shownUnit = state.size_unit;
    }

    function readSize() {
      if (unitField.value !== shownUnit) {       // the unit changed: relabel
        state.size_unit = unitField.value;
        writeSize();
        return;
      }
      const factor = (unitField.value === 'in') ? MM_PER_IN : 1;
      const width = parseFloat(widthField.value);
      const height = parseFloat(heightField.value);
      state.size_unit = unitField.value;
      if (Number.isFinite(width) && width > 0) state.width_mm = width * factor;
      state.height_mm = Number.isFinite(height) && height > 0 ? height * factor : 0;
      const panelW = parseFloat(panelWField.value);
      const panelH = parseFloat(panelHField.value);
      if (Number.isFinite(panelW) && panelW > 0) state.panel_width_mm = panelW * factor;
      if (Number.isFinite(panelH) && panelH > 0) state.panel_height_mm = panelH * factor;
      // 0 or empty is a real answer here: it hands the size back to the figure.
      const axesW = parseFloat(axesWField.value);
      const axesH = parseFloat(axesHField.value);
      state.axes_width_mm = Number.isFinite(axesW) && axesW > 0 ? axesW * factor : 0;
      state.axes_height_mm = Number.isFinite(axesH) && axesH > 0 ? axesH * factor : 0;
      const bar = parseFloat(barField.value);
      state.bar_mm = Number.isFinite(bar) && bar > 0 ? bar * factor : 0;
    }

    /* The color boxes: one per series, once the host has said what the
       series are, and the palette's first four swatches until then -- a row of
       pickers labeled with nothing is worse than no row at all.  Rebuilt only
       when the palette or the list of names changes, because rebuilding it
       under an open picker closes the picker. */
    let colourKeys = [];
    /* The names as the data handed them over, kept so that "Default order"
       has something to go back to: `colourKeys` is the *drawn* sequence and a
       move rewrites it, so clearing `order` without this would leave the panel
       showing a sequence the figure no longer draws -- the two disagreeing is
       the exact failure the one-sequence rule exists to prevent. */
    let arrivedKeys = [];
    let colourSig = null;
    const colourInputs = {};

    function writeColours() {
      colourKeys.forEach((key, i) => {
        const input = colourInputs[key];
        if (input) input.value = state.colours[key] || LK.seriesColour(i, state.palette);
      });
    }

    function buildSwatches() {
      const sig = state.palette + '\u0000' + colourKeys.join('\u0000');
      if (sig === colourSig) return;
      colourSig = sig;
      swatches.innerHTML = '';
      Object.keys(colourInputs).forEach((key) => delete colourInputs[key]);
      if (!colourKeys.length) {
        (PALETTE_SWATCHES[state.palette] || []).forEach((colour) =>
          swatches.appendChild(el('span', { class: 'swatch', style: { background: colour } })));
        return;
      }
      colourKeys.forEach((key, at) => {
        /* Keyed by name: an override has to mean "the KO is this red", not
           "the second group is", or a figure that gains a genotype in the
           middle of the list silently recolors the ones either side. */
        const input = el('input', { type: 'color', title: key, oninput: () => {
          state.colours = Object.assign({}, state.colours, { [key]: input.value });
        } });
        colourInputs[key] = input;
        /* This list is already the conditions in the order they are drawn, so
           it is where they are reordered: one move writes `order`, and the
           figure, the colors and the brackets all follow it because they are
           all read off the one sequence. */
        const move = (step) => () => {
          const i = colourKeys.indexOf(key);
          const j = i + step;
          if (i < 0 || j < 0 || j >= colourKeys.length) return;
          const next = colourKeys.slice();
          next.splice(j, 0, next.splice(i, 1)[0]);
          colourKeys = next;
          state.order = next.slice();
          colourSig = null;            // the signature IS the key list: force the rebuild
          refresh();
        };
        const arrow = (text, step, dead) => el('button', {
          type: 'button', class: 'nudge', text, disabled: dead,
          title: dead ? '' : (step < 0 ? 'Draw ' + key + ' earlier'
                                       : 'Draw ' + key + ' later'),
          onclick: move(step),
        });
        swatches.appendChild(el('label', { class: 'swatch-pick' },
          [arrow('\u25c2', -1, at === 0), input, LK.hexBox(input),
           el('span', { text: key }),
           arrow('\u25b8', 1, at === colourKeys.length - 1)]));
      });
      orderReset.hidden = !(state.order && state.order.length);
      writeColours();
    }

    function readFields() {
      readSize();
      for (const key in fields) {
        const field = fields[key];
        if (field.type === 'checkbox') state[key] = field.checked;
        // A select with no matching option is showing nothing, not a choice:
        // a spec restored from a saved session can name a group to compare
        // against, or a field to color by, before the options for it exist.
        // Reading the widget back here would replace it with whatever happened
        // to be on screen -- silently, and the figure would then be drawn to a
        // setting nobody chose.
        else if (field.tagName === 'SELECT'
                 && (field.selectedIndex < 0
                     || (field.selectedOptions[0] || {}).dataset?.keep)) continue;
        else if (field.type === 'number') state[key] = parseFloat(field.value);
        else state[key] = field.value;
      }
      pBlock.classList.toggle('off', !state.show_p);
      // A paired test is the only one that needs a pairing column, so the box
      // only exists when one is chosen -- an empty field nobody has to wonder
      // about is worse than no field.
      pairOnRow.hidden = !(state.p_test === 'wilcoxon' || state.p_test === 'paired-t');
      buildSwatches();
      preview.textContent = describe(state);
    }

    orderReset.addEventListener('click', () => {
      state.order = [];
      colourKeys = arrivedKeys.slice();   // back to the order the data arrived in
      colourSig = null;
      refresh();
    });

    let notifying = false;
    function refresh() {
      readFields();
      if (!opts.onchange || notifying) return;
      notifying = true;
      try { opts.onchange(Object.assign({}, state)); } finally { notifying = false; }
    }

    /* Saved schemes live in labkit's own store, shared by every app, so the
       row only appears once this one answers for it -- a Save button that
       cannot save is worse than no button. */
    const schemesUrl = opts.schemes === false ? '' :
      (opts.schemes || '/api/colour_schemes');
    let saved = {};

    async function loadSchemes(pick) {
      if (!schemesUrl) return;
      try {
        saved = (await LK.api(schemesUrl)).schemes || {};
      } catch (e) { schemeRow.hidden = true; return; }
      schemeField.innerHTML = '';
      schemeField.appendChild(el('option', { value: '', text: 'From the palette' }));
      Object.keys(saved).forEach((name) =>
        schemeField.appendChild(el('option', { value: name, text: name })));
      schemeField.value = pick || '';
      schemeRow.hidden = false;
    }

    schemeField.addEventListener('change', (event) => {
      event.stopPropagation();     // this is not a spec field; it sets one
      // "From the palette" is the way back: no overrides at all, every series
      // takes its palette color again.
      state.colours = Object.assign({}, saved[schemeField.value] || {});
      colourSig = null;            // the pickers are showing the old scheme
      buildSwatches();
      writeColours();
      refresh();
    });

    saveScheme.addEventListener('click', async () => {
      const name = (window.prompt('Save these colors as',
        schemeField.value || 'scheme') || '').trim();
      if (!name) return;
      readFields();
      try {
        await LK.api(schemesUrl + '/' + encodeURIComponent(name),
          { method: 'PUT', body: { colours: state.colours } });
        await loadSchemes(name);
      } catch (e) { window.alert(e.message); }
    });

    dropScheme.addEventListener('click', async () => {
      const name = schemeField.value;
      if (!name || !window.confirm('Delete the saved scheme \u201c' + name + '\u201d?')) return;
      try {
        await LK.api(schemesUrl + '/' + encodeURIComponent(name), { method: 'DELETE' });
        await loadSchemes('');
      } catch (e) { window.alert(e.message); }
    });

    function describe(s) {
      if (!s.show_p) return 'No p-values drawn.';
      const test = (fields.p_test.selectedOptions[0] || {}).text || s.p_test;
      const correction = s.p_correction === 'none' ? 'uncorrected'
        : (fields.p_correction.selectedOptions[0] || {}).text.replace(/ \(.*\)/, '') + '-corrected';
      const shape = { auto: 'p = 0.031', exact: 'p = 0.031', stars: '✱✱', both: 'p = 0.031 ✱' }[s.p_style];
      return test + ', ' + correction + ' → ' + shape;
    }

    /* `change` alone for selects and checkboxes; `input` as well for the
       number boxes, where waiting for blur would feel broken.  Listening for
       both on everything meant two notifications per edit -- and two figures
       rendered -- for every dropdown. */
    root.addEventListener('change', refresh);
    root.addEventListener('input', (event) => {
      if (event.target.type === 'number' || event.target.type === 'text') refresh();
    });

    /* Putting a value on a widget. A select with no option for the value shows
       one that describes it -- "5 chosen pairs" -- rather than an unrelated
       option that would then be read back as the setting, or a blank box over
       a setting that is really there.

       A list is the case that needs this: "each construct against itself" is
       a set of pairs no dropdown can express. The describing option is made
       here, on demand, and relabelled every time, because it outlives the
       list that first needed it and "4 chosen pairs" over a five-pair list is
       a lie about the figure -- the one thing it exists to prevent. */
    function show(field, value) {
      if (field.type === 'checkbox') { field.checked = !!value; return; }
      field.value = value;
      if (field.tagName !== 'SELECT' || field.selectedIndex >= 0) return;
      let described = [...field.options].find((o) => o.dataset.keep);
      if (!described && Array.isArray(value) && value.length) {
        described = el('option', { value: '' });
        described.dataset.keep = 'chosen pairs';
        field.insertBefore(described, field.firstChild);
      }
      if (!described) { field.selectedIndex = -1; return; }
      if (Array.isArray(value)) described.textContent = value.length + ' ' + described.dataset.keep;
      field.selectedIndex = described.index;
    }

    // Push the initial state into the widgets, then read it straight back so
    // `spec()` is correct before anything has been touched.
    for (const key in fields) show(fields[key], state[key]);
    unitField.value = state.size_unit || 'mm';
    writeSize();
    readFields();

    if (host) host.appendChild(root);
    loadSchemes('');

    return {
      node: root,
      spec() { readFields(); return Object.assign({}, state); },
      /* Setting values programmatically does not notify: the caller already
         knows what it just changed, and firing onchange here is the other half
         of the loop above. */
      set(patch) {
        Object.assign(state, patch || {});
        for (const key in fields) {
          if (key in state) show(fields[key], state[key]);
        }
        unitField.value = state.size_unit || 'mm';
        writeSize();
        readFields();
        writeColours();
      },
      /** One color box per series, named, in the order they will be drawn.
          Call it once the labels are known: it is also the reorder control, so
          an app that never calls it gets palette swatches and no way to say
          what sequence the conditions go in. */
      setColourKeys(labels) {
        arrivedKeys = (labels || []).map(String);
        colourKeys = LK.orderSeries(arrivedKeys, state.order);
        buildSwatches();
      },
      /** Offer "everything against <control>" once the group labels are known. */
      setGroups(labels) {
        // What is *set*, not what the box happens to show: rebuilding the
        // options empties it, and reading the empty box back would be how a
        // choice gets lost.
        const current = state.p_pairs === undefined ? pairsField.value : state.p_pairs;
        pairsField.innerHTML = '';
        [['all', 'Every pair'], ['adjacent', 'Neighbors only']].forEach(([v, t]) =>
          pairsField.appendChild(el('option', { value: v, text: t })));
        (labels || []).forEach((label) => pairsField.appendChild(
          el('option', { value: label, text: 'All vs ' + label })));
        /* Rebuilding the options threw away the describing option along with
           the rest, so `show` puts it back. Falling straight to "Every pair"
           here would quietly restore the cross-group brackets a chosen list
           exists to prevent, and the figure would carry p-values nobody asked
           for. */
        show(pairsField, current);
        if (pairsField.selectedIndex < 0) pairsField.value = 'all';
        readFields();
      },
      /** Offer the grouping fields as trace colors, once the panels are known. */
      setColourBy(names) {
        const colourField = fields.family_colour;
        const current = state.family_colour || colourField.value;
        colourField.innerHTML = '';
        [['voltage', 'Step voltage'], ['black', 'Black only']].forEach(([v, t]) =>
          colourField.appendChild(el('option', { value: v, text: t })));
        (names || []).forEach((name) => colourField.appendChild(
          el('option', { value: name, text: 'By ' + String(name).replace(/_/g, ' ') })));
        colourField.value = current;
        if (colourField.selectedIndex < 0) colourField.value = 'voltage';
        readFields();
      },
    };
  };

  // ----------------------------------------------------------- plate map ---

  /* labkit.plate.FORMATS: rows and columns of each SBS plate, smallest first. */
  LK.PLATE_FORMATS = {
    '6-well': [2, 3], '12-well': [3, 4], '24-well': [4, 6],
    '48-well': [6, 8], '96-well': [8, 12], '384-well': [16, 24],
  };

  /* labkit.plate.canon_well for the spellings a layout file can hold: "b1",
     "B-01" -> "B01". The server reads the rest (a plate reader's r02c01). */
  function canonWell(text) {
    const m = /^([A-Pa-p])[\s_-]?0*(\d{1,2})$/.exec(String(text).trim());
    return m ? m[1].toUpperCase() + m[2].padStart(2, '0') : null;
  }

  /**
   * A plate map: which wells got what.
   *
   *   layout    { format, labels: { column: { B01: 'WT' } } } -- the shape
   *             labkit.plate reads and writes, edited in place
   *   wells     the ids that have data; every other well is drawn faint
   *   emptyNote what a well with no data is called on hover ('no images')
   *   columns   names to offer for a column of labels
   *   onchange  called after every edit
   *
   * Select wells (click, drag, or a row or column heading), type what is in
   * them, Enter.  One column of labels at a time: a two-factor plate is two
   * passes over the same grid, not one cell holding two things.
   */
  LK.plateEditor = function (host, options) {
    const opts = Object.assign({ layout: null, wells: [], onchange: null,
      columns: ['variant', 'treatment', 'condition', 'concentration', 'cell_line'],
      // What a well with no data is called on hover. Most callers here mean
      // images; an app planning a plate it has not run yet means something
      // else, and a hard-coded 'no images' was simply wrong on its screen.
      emptyNote: 'no images' },
    options || {});
    const layout = opts.layout || {};
    layout.labels = layout.labels || {};
    const present = new Set(opts.wells);
    const picked = new Set();
    let column = Object.keys(layout.labels)[0] || opts.columns[0];
    let adding = true;          // what the drag in progress does to a well
    let dragging = false;       // a press on a well, not yet let go

    // The smallest plate that holds every well in use. Nothing in a filename
    // says which of the six it was, but r08c02 rules out the four smaller ones.
    function fit() {
      const used = opts.wells.concat(...Object.values(layout.labels).map(Object.keys));
      const holds = (f) => used.every((id) => id.charCodeAt(0) - 65 < LK.PLATE_FORMATS[f][0]
        && Number(id.slice(1)) <= LK.PLATE_FORMATS[f][1]);
      if (!LK.PLATE_FORMATS[layout.format] || !holds(layout.format)) {
        layout.format = Object.keys(LK.PLATE_FORMATS).find(holds) || '384-well';
      }
    }
    const changed = () => opts.onchange && opts.onchange(layout);

    const formatSelect = el('select', {
      title: 'Plate format',
      onchange: (e) => { layout.format = e.target.value; picked.clear(); paint(); changed(); },
    });
    const columnSelect = el('select', {
      title: 'The column these labels make. One at a time: pick another to label '
        + 'the same wells by something else, such as a treatment.',
      onchange: (e) => {
        if (e.target.value === '') {
          column = (prompt('Name of the new column') || '').trim() || column;
        } else column = e.target.value;
        paint();
      },
    });
    const labelBox = el('input', {
      type: 'text', placeholder: 'what is in the selected wells',
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); assign(); } },
    });
    // Select, type, Enter: the label box takes the keyboard as soon as the
    // mouse lets go of the grid.
    const grid = el('div', { class: 'lk-plate',
      onmouseup: () => { dragging = false; labelBox.focus(); } });
    const legend = el('div', { class: 'lk-plate-bar' });
    const fileBox = el('input', {
      type: 'file', accept: '.json,application/json', style: { display: 'none' },
      onchange: async (e) => {
        const file = e.target.files[0];
        e.target.value = '';
        if (!file) return;
        let read;
        try { read = JSON.parse(await file.text()); } catch (err) {
          alert('That is not a layout file: ' + err.message); return;
        }
        // A layout saved here, or one saved by Scrambler (one `assignments` map).
        const labels = read.labels || (read.assignments ? { condition: read.assignments } : {});
        layout.format = read.format || read.plate_format;
        layout.labels = {};
        Object.keys(labels).forEach((name) => Object.keys(labels[name] || {}).forEach((well) => {
          const id = canonWell(well), value = String(labels[name][well] || '').trim();
          if (id && value) (layout.labels[name] = layout.labels[name] || {})[id] = value;
        }));
        column = Object.keys(layout.labels)[0] || column;
        picked.clear();
        paint();
        changed();
      },
    });

    function assign() {
      const value = labelBox.value.trim();
      const into = layout.labels[column] = layout.labels[column] || {};
      picked.forEach((id) => { if (value) into[id] = value; else delete into[id]; });
      if (!Object.keys(into).length) delete layout.labels[column];
      picked.clear();
      labelBox.value = '';
      paint();
      changed();
    }

    function paint() {
      fit();
      const rows = LK.PLATE_FORMATS[layout.format][0], cols = LK.PLATE_FORMATS[layout.format][1];
      const labels = layout.labels[column] || {};
      // In the order they were first given, not sorted: a label keeps its color
      // while the next ones are added.
      const values = Array.from(new Set(Object.values(labels)));
      const tint = (value) => LK.seriesColour(values.indexOf(value));

      formatSelect.innerHTML = '';
      Object.keys(LK.PLATE_FORMATS).forEach((f) => formatSelect.appendChild(
        el('option', { value: f, text: f, selected: f === layout.format })));
      columnSelect.innerHTML = '';
      Array.from(new Set(Object.keys(layout.labels).concat(column, opts.columns)))
        .forEach((name) => {
          const n = Object.keys(layout.labels[name] || {}).length;
          columnSelect.appendChild(el('option', {
            value: name, selected: name === column,
            text: name + (n ? ' · ' + n + (n === 1 ? ' well' : ' wells') : ''),
          }));
        });
      columnSelect.appendChild(el('option', { value: '', text: 'other…' }));

      const ids = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) ids.push(String.fromCharCode(65 + r) + String(c + 1).padStart(2, '0'));
      }
      // A heading selects its whole line, or clears it if it already is.
      const toggle = (line) => {
        const on = !line.every((id) => picked.has(id));
        line.forEach((id) => (on ? picked.add(id) : picked.delete(id)));
        paint();
      };
      const heading = (text, line, title) => el('div', {
        class: 'lk-plate-head', text, title, onclick: () => toggle(line) });

      grid.innerHTML = '';
      grid.style.gridTemplateColumns = '1.7em repeat(' + cols + ', minmax(0, 1fr))';
      grid.classList.toggle('dense', cols > 12);
      grid.appendChild(heading('', ids, 'Every well'));
      for (let c = 0; c < cols; c++) {
        grid.appendChild(heading(String(c + 1), ids.filter((id, i) => i % cols === c), 'Column ' + (c + 1)));
      }
      ids.forEach((id, i) => {
        if (i % cols === 0) grid.appendChild(heading(id[0], ids.slice(i, i + cols), 'Row ' + id[0]));
        const said = Object.keys(layout.labels).filter((name) => layout.labels[name][id])
          .map((name) => name + ': ' + layout.labels[name][id]);
        const mark = (node) => {
          if (adding) picked.add(id); else picked.delete(id);
          node.classList.toggle('sel', adding);
        };
        grid.appendChild(el('div', {
          class: 'lk-well' + (present.has(id) ? ' has' : '') + (picked.has(id) ? ' sel' : ''),
          text: labels[id] || '',
          title: [id].concat(said, present.has(id) ? [] : [opts.emptyNote]).join(' · '),
          style: labels[id] ? { background: tint(labels[id]) + '66' } : null,
          // preventDefault: a drag across the grid is a selection of wells,
          // not of the text in them.
          onmousedown: (e) => {
            e.preventDefault(); dragging = true; adding = !picked.has(id); mark(e.currentTarget);
          },
          // Only a drag that began on a well. A press on a well the pointer was
          // not yet known to be over (a redrawn grid, a tap) arrives as an enter
          // with the button already down, then the press: marking on that enter
          // made the press undo it, so every other selection came out empty.
          onmouseenter: (e) => {
            if (!(e.buttons & 1)) dragging = false; else if (dragging) mark(e.currentTarget);
          },
        }));
      });

      legend.innerHTML = '';
      values.forEach((value) => legend.appendChild(el('button', {
        class: 'ghost small', title: 'Use this label for the selected wells',
        onclick: () => { labelBox.value = value; labelBox.focus(); },
      }, [el('span', { class: 'lk-plate-swatch', style: { background: tint(value) } }),
          value + ' · ' + Object.values(labels).filter((v) => v === value).length])));
    }

    const node = el('div', { class: 'lk-plate-editor' }, [
      el('div', { class: 'lk-plate-bar' }, [
        formatSelect, columnSelect, labelBox,
        el('button', { class: 'small', text: 'set',
          title: 'Label the selected wells. An empty label clears them.', onclick: assign }),
        el('button', { class: 'ghost small', text: 'load…',
          title: 'Open a saved layout (plate_layout.json, or a layout saved by Scrambler).',
          onclick: () => fileBox.click() }),
        el('button', { class: 'ghost small', text: 'save…',
          title: 'Download this layout as plate_layout.json.',
          onclick: () => {
            const link = el('a', { download: 'plate_layout.json', href: URL.createObjectURL(
              new Blob([JSON.stringify(layout, null, 1)], { type: 'application/json' })) });
            link.click();
            URL.revokeObjectURL(link.href);
          } }),
        fileBox,
      ]),
      grid, legend,
    ]);
    paint();
    if (host) host.appendChild(node);
    return { node, layout };
  };

  // ------------------------------------------------------------ toolbars ---

  /** Drop a theme toggle at the end of the app's toolbar, wherever it is. */
  LK.mountThemeToggle = function (selector) {
    const bar = document.querySelector(selector || 'header, .topbar, #toolbar');
    if (!bar || bar.querySelector('.lk-theme-btn')) return;
    const button = LK.theme.button();
    button.classList.add('lk-theme-btn');
    if (!bar.querySelector('.spacer') && !bar.querySelector('.group.right')) {
      button.style.marginLeft = 'auto';
    }
    bar.appendChild(button);
  };

  global.LK = LK;
}(window));
