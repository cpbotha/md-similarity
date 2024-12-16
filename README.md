# markdown-similarity

Chunk Hugo markdown files by headings, embed all chunks using ollama nomic-embed-text by default, then offer chunk-based retrieval of similar chunks.

My use case is to surface older blog posts that are similar to something I'm currently working on.

## Quickstart

### Install everything

- [install ollama](https://ollama.com/download)
- In a terminal: `ollama start`
- In another terminal: `ollama pull nomic-embed-text`
- Install this tool: `uv tool install git+https://github.com/cpbotha/md-similarity.git` -- if you follow instructions, you should now have `mdsim` in your path.

### Pre-process existing posts

```shell
cd ~/your/hugo-site/content/posts && mdsim embed-posts .
```

This will create `mdsim.db` with all embeddings in `content/posts`.

Any future invocations will reuse as many of these embeddings as possible, based on sha256 hashes of document chunks.

### Find similar posts

As you're working on your new blog post, or just with any other post:

```shell
cd ~/your/hugo/site/content/posts && mdsim list-similar subdir/my_new_post.md
```

You can also search for any query string:

```shell
mdsim search "any amount of text here"
```
