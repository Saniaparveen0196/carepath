/*! Patient Guidance chat widget. Usage:
    <script src="https://YOUR-SERVER/widget.js" data-workspace="demo"
            data-title="Ask us" data-color="#1f6feb"></script>
*/
(function () {
  "use strict";

  // ---------- safe mini-markdown (HTML is escaped first) ----------
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inline(s) {
    s = esc(s);
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[\s(])_(.+?)_(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[\s(])\*(?!\s)(.+?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    return s;
  }
  function cells(row) {
    return row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(function (c) { return c.trim(); });
  }
  function table(rows) {
    var head = cells(rows[0]);
    var body = rows.slice(1)
      .filter(function (r) { return !/^\s*\|?\s*:?-{2,}/.test(r); })
      .map(cells);
    return '<div class="tw"><table><thead><tr>' +
      head.map(function (h) { return "<th>" + inline(h) + "</th>"; }).join("") +
      "</tr></thead><tbody>" +
      body.map(function (r) {
        return "<tr>" + r.map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>";
      }).join("") + "</tbody></table></div>";
  }
  function md(text) {
    var lines = String(text).split("\n"), html = "", i = 0;
    while (i < lines.length) {
      var line = lines[i], items;
      if (/^\s*\|/.test(line)) {
        var rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i]); i++; }
        html += table(rows); continue;
      }
      if (/^\s*[-*\u2022]\s+/.test(line)) {
        items = [];
        while (i < lines.length && /^\s*[-*\u2022]\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^\s*[-*\u2022]\s+/, "")); i++;
        }
        html += "<ul>" + items.map(function (x) { return "<li>" + inline(x) + "</li>"; }).join("") + "</ul>"; continue;
      }
      if (/^\s*\d+\.\s+/.test(line)) {
        items = [];
        while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^\s*\d+\.\s+/, "")); i++;
        }
        html += "<ol>" + items.map(function (x) { return "<li>" + inline(x) + "</li>"; }).join("") + "</ol>"; continue;
      }
      if (!line.trim()) { i++; continue; }
      html += "<p>" + inline(line) + "</p>"; i++;
    }
    return html;
  }

  if (typeof module !== "undefined") { module.exports = { md: md }; }
  if (typeof document === "undefined") { return; }

  // ---------- widget ----------
  var script = document.currentScript;
  var ws = (script && script.dataset.workspace) || "demo";
  var color = (script && script.dataset.color) || "#1f6feb";
  var title = (script && script.dataset.title) || "Ask about treatments & hospitals";
  var api = (script && script.dataset.api) ||
    (script && script.src ? new URL(script.src).origin : "");

  var host = document.createElement("div");
  document.body.appendChild(host);
  var root = host.attachShadow({ mode: "open" });
  root.innerHTML =
    "<style>" +
    ":host{all:initial}" +
    "*{box-sizing:border-box;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}" +
    ".fab{position:fixed;right:20px;bottom:20px;width:56px;height:56px;border-radius:50%;border:0;" +
    "background:" + color + ";color:#fff;font-size:26px;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.25);z-index:2147483646}" +
    ".panel{position:fixed;right:20px;bottom:90px;width:min(400px,calc(100vw - 24px));height:min(600px,calc(100vh - 110px));" +
    "background:#fff;border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.28);display:none;flex-direction:column;overflow:hidden;z-index:2147483647}" +
    ".panel.open{display:flex}" +
    ".hd{background:" + color + ";color:#fff;padding:14px 16px;font-weight:600;font-size:15px;display:flex;justify-content:space-between;align-items:center}" +
    ".hd button{background:none;border:0;color:#fff;font-size:20px;cursor:pointer}" +
    ".msgs{flex:1;overflow-y:auto;padding:14px;background:#f5f7fa}" +
    ".m{max-width:92%;margin:0 0 10px;padding:10px 12px;border-radius:12px;font-size:14px;line-height:1.5;color:#1b2a41;word-wrap:break-word}" +
    ".m p{margin:0 0 8px}.m p:last-child{margin:0}.m ul,.m ol{margin:4px 0 8px;padding-left:20px}" +
    ".u{background:" + color + ";color:#fff;margin-left:auto;border-bottom-right-radius:4px}" +
    ".b{background:#fff;border:1px solid #e1e6ee;border-bottom-left-radius:4px}" +
    ".tw{overflow-x:auto;margin:6px 0}table{border-collapse:collapse;font-size:12px}" +
    "th,td{border:1px solid #d9e0ea;padding:4px 6px;text-align:left;vertical-align:top}th{background:#eef2f8}" +
    ".src{margin-top:8px;padding-top:6px;border-top:1px solid #e8edf4;font-size:12px}" +
    ".src a{color:" + color + ";display:block;margin:2px 0;text-decoration:none}.src a:hover{text-decoration:underline}" +
    ".err{color:#b42318}.typing{color:#6b7a90;font-style:italic}" +
    ".qr{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}" +
    ".qr button{border:1px solid " + color + ";background:#fff;color:" + color + ";border-radius:16px;padding:6px 12px;font-size:13px;cursor:pointer}" +
    ".qr button:hover{background:" + color + ";color:#fff}" +
    ".ft{display:flex;gap:8px;padding:10px;border-top:1px solid #e1e6ee;background:#fff}" +
    ".ft input{flex:1;border:1px solid #cdd6e3;border-radius:20px;padding:10px 14px;font-size:14px;outline:none}" +
    ".ft input:focus{border-color:" + color + "}" +
    ".ft button{border:0;background:" + color + ";color:#fff;border-radius:20px;padding:0 16px;font-size:14px;cursor:pointer}" +
    ".ft button:disabled{opacity:.5;cursor:default}" +
    ".note{font-size:11px;color:#6b7a90;text-align:center;padding:0 10px 8px;background:#fff}" +
    "</style>" +
    '<button class="fab" aria-label="Open chat">\uD83D\uDCAC</button>' +
    '<div class="panel" role="dialog" aria-label="Chat assistant">' +
    '<div class="hd"><span class="t"></span><button class="x" aria-label="Close">\u00D7</button></div>' +
    '<div class="msgs"></div>' +
    '<div class="ft"><input type="text" maxlength="1000" placeholder="Type your question..." aria-label="Message">' +
    '<button class="send">Send</button></div>' +
    '<div class="note">Information only, not medical advice. In an emergency, call your local emergency number.</div>' +
    "</div>";

  var $ = function (s) { return root.querySelector(s); };
  var panel = $(".panel"), msgs = $(".msgs"), input = $("input"), send = $(".send");
  $(".t").textContent = title;

  var sid = null;
  try { sid = sessionStorage.getItem("pg_sid_" + ws); } catch (e) {}

  function add(cls, html) {
    var d = document.createElement("div");
    d.className = "m " + cls;
    d.innerHTML = html;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  function addSources(el, sources) {
    if (!sources || !sources.length) return;
    var box = document.createElement("div");
    box.className = "src";
    box.appendChild(document.createTextNode("Sources"));
    sources.forEach(function (s) {
      var line;
      if (s.url && /^https?:\/\//.test(s.url)) {
        line = document.createElement("a");
        line.href = s.url; line.target = "_blank"; line.rel = "noopener noreferrer";
      } else {
        line = document.createElement("div");
      }
      line.textContent = "[" + s.n + "] " + s.title;
      box.appendChild(line);
    });
    el.appendChild(box);
  }

  function clearChips() {
    var old = msgs.querySelectorAll(".qr");
    for (var i = 0; i < old.length; i++) old[i].remove();
  }

  function addChips(list) {
    if (!list || !list.length) return;
    var box = document.createElement("div");
    box.className = "qr";
    list.forEach(function (label) {
      var b = document.createElement("button");
      b.textContent = label;
      b.addEventListener("click", function () { ask(label); });
      box.appendChild(b);
    });
    msgs.appendChild(box);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function ask(preset) {
    var q = (typeof preset === "string" ? preset : input.value).trim();
    if (!q || send.disabled) return;
    clearChips();
    input.value = "";
    add("u", esc(q));
    send.disabled = true;
    var wait = add("b typing", "Looking that up...");

    fetch(api + "/api/" + encodeURIComponent(ws) + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, session_id: sid })
    }).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, status: r.status, body: j }; });
    }).then(function (res) {
      wait.remove();
      if (!res.ok) {
        var msg = (res.body && res.body.detail) || "Something went wrong.";
        add("b err", esc(typeof msg === "string" ? msg : "Please check your message and try again."));
        return;
      }
      sid = res.body.session_id;
      try { sessionStorage.setItem("pg_sid_" + ws, sid); } catch (e) {}
      var el = add("b", md(res.body.answer));
      addSources(el, res.body.sources);
      addChips(res.body.quick_replies);
    }).catch(function () {
      wait.remove();
      add("b err", "Could not reach the assistant. Please check your connection and try again.");
    }).then(function () {
      send.disabled = false;
      input.focus();
    });
  }

  $(".fab").addEventListener("click", function () {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) input.focus();
  });
  $(".x").addEventListener("click", function () { panel.classList.remove("open"); });
  send.addEventListener("click", function () { ask(); });
  input.addEventListener("keydown", function (e) { if (e.key === "Enter") ask(); });

  add("b", "Hi! I can help you compare hospitals and understand treatments. " +
    "I can't give medical advice or diagnose. Try: <em>best hospitals for bypass surgery in Delhi</em>.");
})();