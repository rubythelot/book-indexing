# Reading List Catalog

A camera-assisted local site for cataloging your reading list.

## Features

- Add books with title, author, reading status, optional rating, and notes.
- Use your webcam in the browser to capture a cover and auto-fill title/author.
- Export all indexed books as CSV (`/books.csv`).
- Track summary counts for books to read, currently reading, and finished.
- Remove books you no longer want in your list.
- Data is stored locally in `reading_list.db` (SQLite).

## Run locally

1. (Optional) Set your OpenAI key to enable cover auto-detection from camera captures:

```bash
export OPENAI_API_KEY="your_key_here"
```

2. Start the app:

```bash
python app.py
```

3. Open in your browser:

```text
http://127.0.0.1:5000
```

## Usage

- Click **Start camera**, position a cover, and click **Scan cover**.
- Confirm/edit title and author, then click **Save book**.
- Click **Export CSV** to download your indexed books.

No external Python dependencies are required (uses only Python standard library).

## License

This project is licensed under the [MIT license](LICENSE).
