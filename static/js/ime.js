/* ═══════════════════════════════════════════════════════════════════════
   Built-in Japanese / Korean keyboard.

   Nobody editing a page on a normal keyboard can type 日本語 or 한국어, so
   when the OCR misses a bubble there was no way to key the original in by
   hand. This provides both routes without installing a system IME:

     - type romaji and watch it become kana live ("konnichiwa" -> こんにちは),
       the same way a real IME works, including small tsu (kk -> っ) and
       compound sounds (kya -> きゃ);
     - or click the characters straight off an on-screen pad (kana for
       Japanese, jamo for Korean — Hangul jamo are composed into proper
       syllables exactly as a Korean keyboard does: ㅎ + ㅏ + ㄴ -> 한).

   Pasting works too, for anyone copying text from elsewhere.
   window.KaisukiIME.open({...}) is the only entry point.
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  /* ── romaji → hiragana ─────────────────────────────────────────────── */
  const KANA = {
    a: "あ", i: "い", u: "う", e: "え", o: "お",
    ka: "か", ki: "き", ku: "く", ke: "け", ko: "こ",
    ga: "が", gi: "ぎ", gu: "ぐ", ge: "げ", go: "ご",
    sa: "さ", si: "し", shi: "し", su: "す", se: "せ", so: "そ",
    za: "ざ", zi: "じ", ji: "じ", zu: "ず", ze: "ぜ", zo: "ぞ",
    ta: "た", ti: "ち", chi: "ち", tu: "つ", tsu: "つ", te: "て", to: "と",
    da: "だ", di: "ぢ", du: "づ", de: "で", do: "ど",
    na: "な", ni: "に", nu: "ぬ", ne: "ね", no: "の",
    ha: "は", hi: "ひ", hu: "ふ", fu: "ふ", he: "へ", ho: "ほ",
    ba: "ば", bi: "び", bu: "ぶ", be: "べ", bo: "ぼ",
    pa: "ぱ", pi: "ぴ", pu: "ぷ", pe: "ぺ", po: "ぽ",
    ma: "ま", mi: "み", mu: "む", me: "め", mo: "も",
    ya: "や", yu: "ゆ", yo: "よ",
    ra: "ら", ri: "り", ru: "る", re: "れ", ro: "ろ",
    wa: "わ", wi: "ゐ", we: "ゑ", wo: "を",
    va: "ヴぁ", vi: "ヴぃ", vu: "ヴ", ve: "ヴぇ", vo: "ヴぉ",
    // compound (youon)
    kya: "きゃ", kyu: "きゅ", kyo: "きょ",
    gya: "ぎゃ", gyu: "ぎゅ", gyo: "ぎょ",
    sha: "しゃ", shu: "しゅ", sho: "しょ",
    sya: "しゃ", syu: "しゅ", syo: "しょ",
    ja: "じゃ", ju: "じゅ", jo: "じょ",
    jya: "じゃ", jyu: "じゅ", jyo: "じょ",
    zya: "じゃ", zyu: "じゅ", zyo: "じょ",
    cha: "ちゃ", chu: "ちゅ", cho: "ちょ",
    tya: "ちゃ", tyu: "ちゅ", tyo: "ちょ",
    nya: "にゃ", nyu: "にゅ", nyo: "にょ",
    hya: "ひゃ", hyu: "ひゅ", hyo: "ひょ",
    bya: "びゃ", byu: "びゅ", byo: "びょ",
    pya: "ぴゃ", pyu: "ぴゅ", pyo: "ぴょ",
    mya: "みゃ", myu: "みゅ", myo: "みょ",
    rya: "りゃ", ryu: "りゅ", ryo: "りょ",
    fa: "ふぁ", fi: "ふぃ", fe: "ふぇ", fo: "ふぉ",
    // explicit small kana
    xa: "ぁ", xi: "ぃ", xu: "ぅ", xe: "ぇ", xo: "ぉ",
    la: "ぁ", li: "ぃ", lu: "ぅ", le: "ぇ", lo: "ぉ",
    xtu: "っ", ltu: "っ", xya: "ゃ", xyu: "ゅ", xyo: "ょ",
    "-": "ー", ".": "。", ",": "、", "!": "！", "?": "？",
    "~": "〜", "[": "「", "]": "」",
  };
  const MAXLEN = 3;

  function romajiToKana(src, katakana) {
    let out = "", i = 0;
    const s = (src || "").toLowerCase();
    while (i < s.length) {
      // "kk" / "tt" … -> small tsu, but never "nn" (that's ん)
      const c = s[i];
      if (c !== "n" && /[a-z]/.test(c) && s[i + 1] === c) {
        out += "っ"; i++; continue;
      }
      let hit = null;
      for (let len = Math.min(MAXLEN, s.length - i); len > 0; len--) {
        const seg = s.slice(i, i + len);
        if (KANA[seg]) { hit = [seg, KANA[seg]]; break; }
      }
      if (hit) { out += hit[1]; i += hit[0].length; continue; }
      // Bare "n" is ん. The subtle part is "nn": in konnichiwa the second n
      // belongs to the NEXT syllable (ん + に), so only swallow both when no
      // vowel follows — which is how a real IME behaves.
      if (c === "n") {
        const nxt = s[i + 1];
        const after = s[i + 2];
        out += "ん";
        // NB: "".includes-style checks are a trap here — an empty string is a
        // substring of everything, so test for a real vowel character.
        i += (nxt === "n" && !(after && "aiueoy".includes(after))) ? 2 : 1;
        continue;
      }
      out += s[i]; i++;      // not convertible yet — leave it (still typing)
    }
    return katakana ? toKatakana(out) : out;
  }

  function toKatakana(str) {
    return str.replace(/[ぁ-ゖ]/g,
      ch => String.fromCharCode(ch.charCodeAt(0) + 0x60));
  }

  /* ── kana pads ─────────────────────────────────────────────────────── */
  const HIRA_ROWS = [
    "あいうえお", "かきくけこ", "さしすせそ", "たちつてと", "なにぬねの",
    "はひふへほ", "まみむめも", "やゆよ", "らりるれろ", "わをん",
    "がぎぐげご", "ざじずぜぞ", "だぢづでど", "ばびぶべぼ", "ぱぴぷぺぽ",
    "ゃゅょっぁぃぅぇぉ", "、。！？ー〜「」",
  ];
  const KATA_ROWS = HIRA_ROWS.map(r => toKatakana(r));

  /* ── Korean jamo, composed into syllables like a real Hangul keyboard ── */
  const CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
               "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"];
  const JUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
                "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"];
  const JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
                "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
                "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"];

  function composeHangul(jamos) {
    // Greedy left-to-right: initial + medial (+ final, released when the next
    // medial arrives, exactly as a Korean keyboard behaves).
    let out = "", ci = -1, ji = -1, ki = 0;
    const flush = () => {
      if (ci >= 0 && ji >= 0) {
        out += String.fromCharCode(0xAC00 + (ci * 21 + ji) * 28 + ki);
      } else if (ci >= 0) {
        out += CHO[ci];
      }
      ci = -1; ji = -1; ki = 0;
    };
    for (const j of jamos) {
      const c = CHO.indexOf(j), v = JUNG.indexOf(j), k = JONG.indexOf(j);
      if (v >= 0) {
        if (ci >= 0 && ji >= 0 && ki > 0) {
          // the trailing consonant actually starts the NEXT syllable
          const moved = JONG[ki];
          ki = 0; flush();
          ci = CHO.indexOf(moved); ji = v;
        } else if (ci >= 0 && ji < 0) {
          ji = v;
        } else {
          flush(); out += j;
        }
      } else if (c >= 0 || k > 0) {
        if (ci >= 0 && ji >= 0 && ki === 0 && k > 0) {
          ki = k;                       // becomes this syllable's final
        } else {
          flush(); ci = c >= 0 ? c : -1;
          if (ci < 0) out += j;
        }
      } else {
        flush(); out += j;              // space, punctuation, anything else
      }
    }
    flush();
    return out;
  }

  /* ── the dialog ────────────────────────────────────────────────────── */
  let el = null;

  function build() {
    if (el) return el;
    el = document.createElement("div");
    el.className = "ime-back";
    el.innerHTML = `
      <div class="ime-box" role="dialog" aria-label="Type Japanese or Korean">
        <div class="ime-head">
          <strong class="ime-title">Type the original text</strong>
          <button class="ime-x" title="Close">✕</button>
        </div>
        <div class="ime-langs">
          <button class="ime-lang on" data-l="hira">Hiragana</button>
          <button class="ime-lang" data-l="kata">Katakana</button>
          <button class="ime-lang" data-l="ko">Korean</button>
        </div>
        <label class="ime-lbl ime-romaji-lbl">
          Type it how it sounds — it turns into characters as you go
        </label>
        <input class="ime-romaji" placeholder="konnichiwa  →  こんにちは" autocomplete="off">
        <div class="ime-pad"></div>
        <label class="ime-lbl">Original text (edit or paste freely)</label>
        <textarea class="ime-src" rows="2" placeholder="…"></textarea>
        <div class="ime-actions">
          <button class="btn btn-ghost btn-sm ime-clear">Clear</button>
          <button class="btn btn-primary btn-sm ime-go">Translate this</button>
        </div>
        <label class="ime-lbl">Translation</label>
        <textarea class="ime-out" rows="2" placeholder="the English goes here — you can edit it"></textarea>
        <div class="ime-foot">
          <span class="ime-msg"></span>
          <span style="flex:1"></span>
          <button class="btn btn-ghost btn-sm ime-cancel">Cancel</button>
          <button class="btn btn-primary btn-sm ime-use">Use this</button>
        </div>
      </div>`;
    document.body.appendChild(el);
    return el;
  }

  function open(opts) {
    const o = opts || {};
    const box = build();
    box.style.display = "flex";
    const q = s => box.querySelector(s);
    const romaji = q(".ime-romaji");
    const src = q(".ime-src");
    const out = q(".ime-out");
    const pad = q(".ime-pad");
    const msg = q(".ime-msg");

    q(".ime-title").textContent = o.title || "Type the original text";
    src.value = o.text || "";
    out.value = o.translation || "";
    romaji.value = "";
    msg.textContent = "";

    // Korean pages open on the Korean pad — that's what they'll need.
    let mode = (o.sourceLang || "").toLowerCase().startsWith("korean")
      ? "ko" : "hira";
    let jamos = [];

    function setMode(m) {
      mode = m;
      jamos = [];
      box.querySelectorAll(".ime-lang").forEach(
        b => b.classList.toggle("on", b.dataset.l === m));
      const jp = m !== "ko";
      q(".ime-romaji-lbl").style.display = jp ? "" : "none";
      romaji.style.display = jp ? "" : "none";
      drawPad();
    }

    function drawPad() {
      pad.innerHTML = "";
      const rows = mode === "hira" ? HIRA_ROWS
        : mode === "kata" ? KATA_ROWS
        : [CHO.join(""), JUNG.slice(0, 11).join(""),
           JUNG.slice(11).join(""), "、。！？…"];
      rows.forEach(r => {
        const line = document.createElement("div");
        line.className = "ime-row";
        for (const ch of r) {
          const b = document.createElement("button");
          b.className = "ime-key";
          b.textContent = ch;
          b.addEventListener("click", () => tap(ch));
          line.appendChild(b);
        }
        pad.appendChild(line);
      });
      const line = document.createElement("div");
      line.className = "ime-row";
      [["space", " "], ["⌫", null]].forEach(([lbl, ch]) => {
        const b = document.createElement("button");
        b.className = "ime-key ime-key-wide";
        b.textContent = lbl;
        b.addEventListener("click", () => ch === null ? backspace() : tap(ch));
        line.appendChild(b);
      });
      pad.appendChild(line);
    }

    function tap(ch) {
      if (mode === "ko") {
        jamos.push(ch);
        src.value = composeHangul(jamos);
      } else {
        src.value += ch;
      }
      src.focus();
    }

    function backspace() {
      if (mode === "ko" && jamos.length) {
        jamos.pop();
        src.value = composeHangul(jamos);
      } else {
        src.value = src.value.slice(0, -1);
      }
    }

    // Live romaji conversion: the converted kana are appended to the source
    // box as soon as a full syllable is formed, so it reads like an IME.
    romaji.oninput = () => {
      const conv = romajiToKana(romaji.value, mode === "kata");
      // keep any trailing un-converted latin in the romaji box
      const m = conv.match(/[a-z]+$/);
      const tail = m ? m[0] : "";
      const done = tail ? conv.slice(0, -tail.length) : conv;
      if (done) {
        src.value += done;
        romaji.value = tail;
      }
    };
    // Typing straight into the source box invalidates the jamo buffer.
    src.oninput = () => { if (mode === "ko") jamos = []; };

    box.querySelectorAll(".ime-lang").forEach(b => {
      b.onclick = () => setMode(b.dataset.l);
    });
    q(".ime-clear").onclick = () => {
      src.value = ""; out.value = ""; romaji.value = ""; jamos = []; msg.textContent = "";
    };

    q(".ime-go").onclick = async () => {
      const text = src.value.trim();
      if (!text) { msg.textContent = "Type or paste some text first."; return; }
      if (!o.translate) { msg.textContent = "Translation isn't available here."; return; }
      const btn = q(".ime-go");
      btn.disabled = true; btn.textContent = "Translating…";
      msg.textContent = "";
      try {
        const t = await o.translate(text);
        out.value = t || "";
        if (!t) msg.textContent = "The model returned nothing — try rewording.";
      } catch (e) {
        msg.textContent = e.message || "Translation failed";
      } finally {
        btn.disabled = false; btn.textContent = "Translate this";
      }
    };

    const close = () => { box.style.display = "none"; document.removeEventListener("keydown", esc); };
    const esc = e => { if (e.key === "Escape") { close(); if (o.onCancel) o.onCancel(); } };
    document.addEventListener("keydown", esc);
    q(".ime-x").onclick = q(".ime-cancel").onclick = () => { close(); if (o.onCancel) o.onCancel(); };
    q(".ime-use").onclick = () => {
      const payload = { original: src.value.trim(), translation: out.value.trim() };
      if (!payload.translation) { msg.textContent = "Nothing to place — translate it first, or type the English yourself."; return; }
      close();
      if (o.onUse) o.onUse(payload);
    };

    setMode(mode);
    setTimeout(() => (mode === "ko" ? src : romaji).focus(), 30);
  }

  window.KaisukiIME = { open, romajiToKana, composeHangul, toKatakana };
})();
