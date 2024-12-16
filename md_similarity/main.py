# %%

import hashlib
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, cast

import frontmatter
import openai
import sqlite_vec
import tqdm
import typer

# from tokenizers import Tokenizer

DEFAULT_DB_NAME = "mdsim.db"
# ollama model
MODEL = "nomic-embed-text"
# huggingface version
TOKENIZER_MODEL = "nomic-ai/nomic-embed-text-v1.5"
max_tokens = 8192
BATCH_SIZE = 128

app = typer.Typer()


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


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


class InputChunk(NamedTuple):
    """
    Attributes
    ----------
    cached_emb
        A cached embedding that was retrieved from the previous DB by hash, so it's already serialized
    """

    rel_name: str
    post_title: str
    section: str
    cached_emb: str | None = None


def _get_db(filename=DEFAULT_DB_NAME):
    db = sqlite3.connect(filename)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("PRAGMA auto_vacuum = FULL;")

    # base_dir
    # rel_name, post-title, section first-line, embedding
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(embedding float[768]);")
    # TODO: create separate posts table but I am lazy
    db.execute(
        "CREATE TABLE IF NOT EXISTS chunks (rel_name TEXT, post_title TEXT, section_line0 TEXT, hash TEXT, embedding_id INTEGER);"
    )

    return db


@lru_cache
def _get_oai():
    # connect to localhost:11434/v1/embeddings using openai API
    oai = openai.OpenAI(base_url="http://localhost:11434/v1")
    return oai


def _handle_batch(batch: list[InputChunk], oai, db, cache):
    cached_batch: list[InputChunk] = []
    new_batch: list[InputChunk] = []
    for c in batch:
        cached_emb = cache.get(_hash_str(c.section))
        if cached_emb is None:
            new_batch.append(c)
        else:
            cached_batch.append(InputChunk(c.rel_name, c.post_title, c.section, cached_emb))

    # print(f"{len(new_batch)} new chunks, {len(cached_batch)} cached chunks")

    def _add_to_db(chunk: InputChunk, serialized_emb):
        db.execute("INSERT INTO chunk_embeddings (embedding) VALUES (vec_normalize(?))", (serialized_emb,))
        db.execute(
            "INSERT INTO chunks (rel_name, post_title, section_line0, hash, embedding_id) VALUES (?, ?, ?, ?, (SELECT last_insert_rowid()))",
            (chunk.rel_name, chunk.post_title, chunk.section.split("\n")[0], _hash_str(chunk.section)),
        )

    if len(new_batch) > 0:
        # create a batch of sections for the embedder
        sections = [b[2] for b in new_batch]
        res: openai.types.CreateEmbeddingResponse = oai.embeddings.create(input=sections, model=MODEL)
        # res.data is a list of Embedding objects, each of which has an array named "embedding"
        embeddings_list = res.data

        for chunk, e in zip(new_batch, embeddings_list):
            _add_to_db(chunk, sqlite_vec.serialize_float32(e.embedding))

    for chunk in cached_batch:
        _add_to_db(chunk, chunk.cached_emb)

    return len(new_batch), len(cached_batch)


@app.command()
def embed_posts(base_dir: str, db_filename: str = DEFAULT_DB_NAME):
    db = _get_db(db_filename)

    cache = {}
    for row in db.execute(
        "SELECT hash, embedding FROM chunk_embeddings LEFT JOIN chunks ON chunks.embedding_id = chunk_embeddings.rowid"
    ):
        cache[row[0]] = row[1]

    if len(cache) > 0:
        print(f"Found {len(cache)} existing embeddings in the database, caching for reuse...")
        # doing "DELETE FROM the_table ..." and then "VACUUM" grew the database by 30% in size
        # so now we're just creating from scratch instead
        db.close()
        Path(db_filename).unlink()
        db = _get_db(db_filename)

    oai = _get_oai()
    cur_batch = []
    num_new_chunks = 0
    num_cached_chunks = 0
    # this is generally not more than hundreds of files, so I listify the generator to get better tqdm progress display
    for md_filename in tqdm.tqdm(list(Path(base_dir).glob("**/*.md"))):
        post, sections = _process_markdown(md_filename.read_text())
        title = cast(str, post["title"]) if "title" in post else md_filename.stem

        for section in sections:
            cur_batch.append(InputChunk(str(md_filename.relative_to(base_dir)), title, section))

        if len(cur_batch) >= BATCH_SIZE:
            new_chunks, cached_chunks = _handle_batch(cur_batch, oai, db, cache)
            num_new_chunks += new_chunks
            num_cached_chunks += cached_chunks
            # print(f"{num_new_chunks + num_cached_chunks} chunks processed")
            cur_batch = []

    if len(cur_batch) > 0:
        new_chunks, cached_chunks = _handle_batch(cur_batch, oai, db, cache)
        num_new_chunks += new_chunks
        num_cached_chunks += cached_chunks
        # print(f"{num_new_chunks + num_cached_chunks} chunks processed after loop end")

    # commit the transaction
    db.execute("commit")
    # close the connection, ensure everything is written!
    db.close()

    print(f"Processed {num_new_chunks} new chunks and reused {num_cached_chunks} cached chunks")


# I was not able to derive a NamedTuple grandchild with success
class SimilarChunk(NamedTuple):
    rel_name: str
    post_title: str
    section_line0: str
    distance: float


def _find_similar(q: str, db_filename):
    db = _get_db(db_filename)
    oai = _get_oai()
    res: openai.types.CreateEmbeddingResponse = oai.embeddings.create(input=q, model=MODEL)
    # res.data is a list of Embedding objects, each of which has an array named "embedding"
    embeddings_list = res.data

    # rows = db.execute("select count(*) from chunk_embeddings").fetchone()
    rows = db.execute(
        "select distance, chunks.rel_name, chunks.post_title, chunks.section_line0 from chunk_embeddings LEFT JOIN chunks ON chunks.embedding_id = chunk_embeddings.rowid where embedding match vec_normalize(?) AND chunk_embeddings.k = 5 order by distance",
        (sqlite_vec.serialize_float32(embeddings_list[0].embedding),),
    ).fetchall()

    return [SimilarChunk(distance=r[0], rel_name=r[1], post_title=r[2], section_line0=r[3]) for r in rows]


@app.command()
def search(q: str, db_filename: str = DEFAULT_DB_NAME):
    similars = _find_similar(q, db_filename)
    for sc in similars:
        print(f"{sc.distance:.3f} - {sc.rel_name} - {sc.post_title} - {sc.section_line0}")


@app.command()
def list_similar(post_filename: str, db_filename: str = DEFAULT_DB_NAME):
    post_path = Path(post_filename)
    post, sections = _process_markdown(post_path.read_text())
    title = cast(str, post["title"]) if "title" in post else post_path.stem

    # here we use title + section_line0 as the key

    print(f"Chunks similar to {title}:")
    similars = {}
    for section in sections:
        my_line0 = section.split("\n")[0]
        my_key = f"{title} - {my_line0}"
        similar_chunks = _find_similar(section, db_filename)
        for sc in similar_chunks:
            key = f"{sc.post_title} - {sc.section_line0}"
            # we ignore that we are similar to ourselves
            if key != my_key:
                if key not in similars or sc.distance < similars[key].distance:
                    similars[key] = (sc, my_line0)

    sorted_similars = sorted(similars.values(), key=lambda x: x[0].distance)
    for sc, my_line0 in sorted_similars[:5]:
        print(f"{sc.distance:.3f} - {sc.rel_name} - {sc.post_title} - {sc.section_line0} 👉 {my_line0}")


if __name__ == "__main__":
    app()
