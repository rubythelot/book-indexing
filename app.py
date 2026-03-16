from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import csv
import json
import os
import sqlite3
import urllib.error
import urllib.request

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "reading_list.db"
HOST = "0.0.0.0"
PORT = 5000


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                status TEXT NOT NULL,
                rating INTEGER,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def fetch_books() -> list[sqlite3.Row]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM books ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()


def add_book(title: str, author: str, status: str, rating: str, notes: str) -> None:
    if status not in {"to-read", "reading", "finished"}:
        status = "to-read"

    normalized_rating = None
    try:
        if rating:
            parsed = int(rating)
            if 1 <= parsed <= 5:
                normalized_rating = parsed
    except ValueError:
        normalized_rating = None

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO books (title, author, status, rating, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                title.strip(),
                author.strip(),
                status,
                normalized_rating,
                notes.strip(),
                datetime.utcnow().isoformat(),
            ),
        )


def remove_book(book_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))


def books_to_csv() -> str:
    books = fetch_books()
    output_rows = [["id", "title", "author", "status", "rating", "notes", "created_at"]]
    for book in books:
        output_rows.append(
            [
                str(book["id"]),
                book["title"],
                book["author"],
                book["status"],
                "" if book["rating"] is None else str(book["rating"]),
                book["notes"] or "",
                book["created_at"],
            ]
        )

    from io import StringIO

    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerows(output_rows)
    return stream.getvalue()


def extract_book_from_image(image_data_url: str) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"title": "", "author": "", "note": "OPENAI_API_KEY missing; fill details manually."}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Identify this book cover. Return strict JSON with keys: "
                            "title and author. If uncertain, use best guess or empty string."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 120,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return {
                "title": str(parsed.get("title", "")).strip(),
                "author": str(parsed.get("author", "")).strip(),
                "note": "",
            }
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return {"title": "", "author": "", "note": "Could not auto-detect. Fill manually and save."}


def render_index() -> str:
    books = fetch_books()
    summary = {
        "all": len(books),
        "to_read": sum(1 for b in books if b["status"] == "to-read"),
        "reading": sum(1 for b in books if b["status"] == "reading"),
        "finished": sum(1 for b in books if b["status"] == "finished"),
    }

    rows = []
    for book in books:
        rating_text = f" · ★ {book['rating']}/5" if book["rating"] else ""
        notes_html = (
            f"<p class='notes'>{escape(book['notes'])}</p>" if book["notes"] else ""
        )
        rows.append(
            f"""
            <li class='book-item'>
              <div>
                <h3>{escape(book['title'])}</h3>
                <p class='muted'>By {escape(book['author'])}</p>
                <p><span class='status status-{book['status']}'>{book['status'].replace('-', ' ')}</span>{rating_text}</p>
                {notes_html}
              </div>
              <form method='post' action='/books/{book['id']}/delete'>
                <button type='submit' class='delete'>Delete</button>
              </form>
            </li>
            """
        )

    book_list = "\n".join(rows) if rows else "<p class='muted'>No books added yet. Capture one from camera or add manually.</p>"

    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>Reading List Catalog</title>
  <style>
    :root {{
      --bg:#090c1a;
      --bg2:#0f1630;
      --panel:rgba(255,255,255,0.06);
      --stroke:rgba(255,255,255,0.15);
      --text:#f6f7ff;
      --muted:#b8bdd6;
      --accent:#7c8cff;
      --accent2:#39d0ff;
      --danger:#ff5f77;
      font-family: Inter, 'Segoe UI', sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; color:var(--text);
      background: radial-gradient(circle at 20% 0%, #1f2d66 0%, var(--bg) 45%),
                  radial-gradient(circle at 100% 100%, #20356d 0%, var(--bg) 40%);
      min-height:100vh;
    }}
    .container {{ max-width:1120px; margin:0 auto; padding:2rem 1rem 3rem; }}
    .hero {{ display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:flex-end; margin-bottom:1rem; }}
    .hero h1 {{ margin:.1rem 0; font-size:clamp(1.8rem,5vw,3.2rem); letter-spacing:.02em; }}
    .muted {{ color:var(--muted); }}
    .action-links a {{ color:var(--text); text-decoration:none; border:1px solid var(--stroke); background:var(--panel); padding:.6rem .9rem; border-radius:999px; margin-left:.5rem; }}
    .summary-grid {{ display:grid; gap:.8rem; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); margin-bottom:1rem; }}
    .summary-grid article,.card {{ background:var(--panel); border:1px solid var(--stroke); border-radius:18px; padding:1rem; backdrop-filter: blur(8px); box-shadow:0 10px 40px rgba(0,0,0,.25); }}
    .summary-grid h2 {{ margin:0; font-size:2rem; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
    @media (max-width: 920px) {{ .grid {{ grid-template-columns:1fr; }} }}
    .book-form {{ display:grid; gap:.75rem; grid-template-columns:repeat(2,minmax(140px,1fr)); }}
    .book-form .full {{ grid-column:1/-1; }}
    label {{ display:flex; flex-direction:column; gap:.35rem; font-size:.94rem; color:var(--muted); }}
    input,select,textarea,button {{ font:inherit; border-radius:12px; border:1px solid var(--stroke); padding:.62rem .72rem; background:rgba(255,255,255,0.06); color:var(--text); }}
    textarea {{ resize:vertical; }}
    button {{ background:linear-gradient(120deg, var(--accent), var(--accent2)); color:white; border:none; cursor:pointer; font-weight:600; }}
    .book-list {{ list-style:none; margin:0; padding:0; display:grid; gap:.7rem; }}
    .book-item {{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; border:1px solid var(--stroke); border-radius:14px; padding:.75rem; background:rgba(255,255,255,0.03); }}
    .book-item h3 {{ margin:0; }}
    .book-item p {{ margin:.25rem 0; }}
    .status {{ text-transform:capitalize; font-weight:700; letter-spacing:.02em; }}
    .status-to-read {{ color:#adb5ff; }} .status-reading {{ color:#70ffd2; }} .status-finished {{ color:#7ecbff; }}
    .notes {{ color:#d2d7ee; }}
    .delete {{ background:var(--danger); }}
    .camera-grid {{ display:grid; gap:.7rem; }}
    video {{ width:100%; border-radius:14px; border:1px solid var(--stroke); background:#02030a; aspect-ratio:16/9; object-fit:cover; }}
    .camera-row {{ display:flex; gap:.6rem; flex-wrap:wrap; }}
    .camera-row button.secondary {{ background:rgba(255,255,255,0.1); border:1px solid var(--stroke); }}
    .hint {{ font-size:.9rem; color:var(--muted); min-height:1.2rem; }}
  </style>
</head>
<body>
  <main class='container'>
    <header class='hero'>
      <div>
        <p class='muted'>Reading tracker</p>
        <h1>Catalog your reading list.</h1>
        <p class='muted'>Camera-assisted indexing, manual edits, and instant CSV export.</p>
      </div>
      <div class='action-links'>
        <a href='/books.csv'>Export CSV</a>
      </div>
    </header>

    <section class='summary-grid'>
      <article><h2>{summary['all']}</h2><p class='muted'>Total books</p></article>
      <article><h2>{summary['to_read']}</h2><p class='muted'>To read</p></article>
      <article><h2>{summary['reading']}</h2><p class='muted'>Reading</p></article>
      <article><h2>{summary['finished']}</h2><p class='muted'>Finished</p></article>
    </section>

    <section class='grid'>
      <article class='card'>
        <h2>Camera indexer</h2>
        <p class='muted'>Use your webcam to capture a cover and auto-fill title/author.</p>
        <div class='camera-grid'>
          <video id='preview' autoplay playsinline muted></video>
          <div class='camera-row'>
            <button id='start-camera' type='button'>Start camera</button>
            <button id='scan-cover' type='button'>Scan cover</button>
          </div>
          <p id='camera-status' class='hint'></p>
        </div>
      </article>

      <article class='card'>
        <h2>Add or edit book</h2>
        <form method='post' action='/books' class='book-form' id='book-form'>
          <label>Title <input id='title' type='text' name='title' required /></label>
          <label>Author <input id='author' type='text' name='author' required /></label>
          <label>Status
            <select name='status'>
              <option value='to-read'>To read</option>
              <option value='reading'>Reading</option>
              <option value='finished'>Finished</option>
            </select>
          </label>
          <label>Rating (1-5) <input type='number' name='rating' min='1' max='5' /></label>
          <label class='full'>Notes <textarea name='notes' rows='3' placeholder='Key thought, quote, or reason to read'></textarea></label>
          <button type='submit'>Save book</button>
        </form>
      </article>
    </section>

    <section class='card' style='margin-top:1rem'>
      <h2>Your books</h2>
      <ul class='book-list'>{book_list}</ul>
    </section>
  </main>

  <script>
    const video = document.getElementById('preview');
    const startBtn = document.getElementById('start-camera');
    const scanBtn = document.getElementById('scan-cover');
    const statusEl = document.getElementById('camera-status');
    const titleInput = document.getElementById('title');
    const authorInput = document.getElementById('author');
    let stream = null;

    startBtn.addEventListener('click', async () => {{
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'environment' }}, audio: false }});
        video.srcObject = stream;
        statusEl.textContent = 'Camera started. Position the book cover and click “Scan cover”.';
      }} catch (err) {{
        statusEl.textContent = 'Could not access camera. Check browser permission.';
      }}
    }});

    scanBtn.addEventListener('click', async () => {{
      if (!stream) {{
        statusEl.textContent = 'Start the camera first.';
        return;
      }}
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL('image/jpeg', 0.85);

      statusEl.textContent = 'Scanning cover...';
      try {{
        const resp = await fetch('/books/from-camera', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ image }})
        }});
        const data = await resp.json();
        if (data.title) titleInput.value = data.title;
        if (data.author) authorInput.value = data.author;
        statusEl.textContent = data.note || 'Scan complete. Confirm and save.';
      }} catch (err) {{
        statusEl.textContent = 'Scan failed. Fill fields manually.';
      }}
    }});
  </script>
</body>
</html>
"""


class ReadingListHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = render_index().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/books.csv":
            content = books_to_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=reading_list.csv")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/books/from-camera":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw)
                image = str(payload.get("image", ""))
                result = extract_book_from_image(image)
                self.json_response(200, result)
            except json.JSONDecodeError:
                self.json_response(400, {"title": "", "author": "", "note": "Invalid JSON body."})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        fields = {k: v[0] for k, v in parse_qs(raw).items()}

        if parsed.path == "/books":
            title = fields.get("title", "").strip()
            author = fields.get("author", "").strip()
            if title and author:
                add_book(
                    title=title,
                    author=author,
                    status=fields.get("status", "to-read").strip(),
                    rating=fields.get("rating", "").strip(),
                    notes=fields.get("notes", "").strip(),
                )
            self.redirect_home()
            return

        if parsed.path.startswith("/books/") and parsed.path.endswith("/delete"):
            try:
                book_id = int(parsed.path.split("/")[2])
                remove_book(book_id)
                self.redirect_home()
            except (IndexError, ValueError):
                self.send_error(400)
            return

        self.send_error(404)

    def json_response(self, status_code: int, payload: dict[str, str]):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def redirect_home(self):
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format, *args):
        return


def run_server() -> None:
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), ReadingListHandler)
    print(f"Serving Reading List Catalog at http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
