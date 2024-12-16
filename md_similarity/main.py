# %%

from functools import lru_cache
import re
import sqlite3
from pathlib import Path
from typing import cast

import frontmatter
import openai
import sqlite_vec
import typer
# from tokenizers import Tokenizer

# ollama model
MODEL = "nomic-embed-text"
# huggingface version
TOKENIZER_MODEL = "nomic-ai/nomic-embed-text-v1.5"
max_tokens = 8192
BATCH_SIZE = 64

app = typer.Typer()


def _process_markdown(markdown_contents) -> tuple[frontmatter.Post, list[str]]:
    # UUUUHHHHH ensure that toml is installed, else this will silently not parse +++ headers
    post = frontmatter.loads(markdown_contents)

    # Split at headers (up to level N)
    # note that we have the heading in a capturing group, so it will be added as a section
    sections = re.split(r"^(#{1,3} .*$)", post.content, flags=re.MULTILINE)

    # Combine headers with their content
    result = []
    i = 0
    while i < len(sections):
        if sections[i].startswith("#") and i < len(sections) - 1:
            # the extra i < check is that there is actually a i+1 after us
            result.append(sections[i] + "\n" + sections[i + 1])
            i += 2
        else:
            result.append(sections[i])
            i += 1

    return post, cast(list[str], result)


# creating this tokenizer is taking 0.5s
# tokenizer = Tokenizer.from_pretrained(TOKENIZER_MODEL)


@lru_cache
def _get_db(filename):
    db = sqlite3.connect(filename)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # base_dir
    # rel_name, post-title, section first-line, embedding
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(embedding float[768]);")
    # TODO: create separate posts table but I am lazy
    db.execute(
        "CREATE TABLE IF NOT EXISTS chunks (rel_name TEXT, post_title TEXT, section_line0 TEXT, embedding_id INTEGER);"
    )

    return db


@lru_cache
def _get_oai():
    # connect to localhost:11434/v1/embeddings using openai API
    oai = openai.OpenAI(base_url="http://localhost:11434/v1")
    return oai


def _handle_batch(batch, oai, db):
    # create a batch of sections
    sections = [b[2] for b in batch]
    res: openai.types.CreateEmbeddingResponse = oai.embeddings.create(input=sections, model=MODEL)
    # res.data is a list of Embedding objects, each of which has an array named "embedding"
    embeddings_list = res.data

    for chunk, e in zip(batch, embeddings_list):
        db.execute(
            "INSERT INTO chunk_embeddings (embedding) VALUES (vec_normalize(?))",
            (sqlite_vec.serialize_float32(e.embedding),),
        )
        db.execute(
            "INSERT INTO chunks (rel_name, post_title, section_line0, embedding_id) VALUES (?, ?, ?, (SELECT last_insert_rowid()))",
            (str(chunk[0]), chunk[1], chunk[2].split("\n")[0]),
        )


@app.command()
def embed_markdown_posts(base_dir: Path):
    # base_dir = Path("/Users/charlbotha/OneDrive/web/cpbotha.net/content/posts/")
    db = _get_db("chunks.db")
    oai = _get_oai()
    cur_batch = []
    num_chunks = 0
    for md_filename in Path(base_dir).glob("**/*.md"):
        post, sections = _process_markdown(md_filename.read_text())
        title = post["title"] if "title" in post else md_filename.stem

        for section in sections:
            cur_batch.append((md_filename.relative_to(base_dir), title, section))

        if len(cur_batch) >= BATCH_SIZE:
            _handle_batch(cur_batch, oai, db)
            num_chunks += len(cur_batch)
            print(f"{num_chunks} chunks processed")
            cur_batch = []

    if len(cur_batch) > 0:
        _handle_batch(cur_batch, oai, db)
        num_chunks += len(cur_batch)
        print(f"{num_chunks} chunks processed after end")

    # commit the transaction
    db.execute("commit")
    # close the connection, ensure everything is written!
    db.close()


@app.command()
def list_similar(q: str):
    db = _get_db("chunks.db")
    oai = _get_oai()
    res: openai.types.CreateEmbeddingResponse = oai.embeddings.create(input=q, model=MODEL)
    # res.data is a list of Embedding objects, each of which has an array named "embedding"
    embeddings_list = res.data

    rows = db.execute("select count(*) from chunk_embeddings").fetchone()
    print(rows)
    rows = db.execute(
        "select distance, chunks.rel_name, chunks.post_title, chunks.section_line0 from chunk_embeddings LEFT JOIN chunks ON chunks.embedding_id = chunk_embeddings.rowid where embedding match vec_normalize(?) AND chunk_embeddings.k = 5 order by distance",
        (sqlite_vec.serialize_float32(embeddings_list[0].embedding),),
    ).fetchall()

    print(rows)
    return rows


if __name__ == "__main__":
    app()
