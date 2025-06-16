# markdown-similarity

Chunk Hugo markdown files by headings, embed all chunks using ollama nomic-embed-text by default, then offer chunk-based retrieval of similar chunks.

My use case is to surface older blog posts that are similar to something I'm currently working on.

## Quickstart

### Install everything

- Use ollama
  - [install ollama](https://ollama.com/download)
  - In a terminal: `ollama start`
  - In another terminal: `ollama pull nomic-embed-text`
- OR use lmstudio
  - [install lmstudio](https://lmstudio.ai/)
  - cog / model search / nomic -> download an embedding model e.g. `text-embedding-nomic-embed-text-v1.5@q8_0`
  - developer / settings: set port to 11434 (same as ollama) / set status to running
- Install this tool: `uv tool install git+https://github.com/cpbotha/md-similarity.git` -- if you follow instructions, you should now have `mdsim` in your path.

### Pre-process existing posts

```shell
cd ~/your/hugo-site/content/posts && mdsim embed-posts .
```

This will create `mdsim.db` with all embeddings in `content/posts`.

Any future invocations will reuse / cache as many of these embeddings as possible, based on sha256 hashes of document chunks.

### Find similar posts

As you're working on your new blog post, or just with any other post:

```shell
cd ~/your/hugo/site/content/posts && mdsim list-similar subdir/my_new_post.md
```

You can also search for any query string:

```shell
mdsim search "any amount of text here"
```

## Examples

For each of the chunks in the input post, `mdsim list-similar` will find the 5 most similar chunks in your whole archive. It will then sort all of these combined chunks by distance, and then list the top 5 along with the input post chunk they are most similar to.

```shell-session
$ mdsim list-similar 2024/whv-256-word/index.md          
Chunks similar to Weekly Head Voices #256: Word:
0.557 - 2024/whv-255-you-lift-me-up/index.md - Weekly Head Voices #255: You lift me up - ### There is a TiL section on my other website {#there-is-a-til-section-on-my-other-website} 👉 ## Making TiL go fast
0.707 - 2019/whv-168-postcards-from-the-edge/index.md - Weekly Head Voices #168: Postcards from the edge. - ## Working late like it's 1998. 👉 ## Making TiL go fast
0.731 - 2019/whv-171-icemirb/index.md - Weekly Head Voices #171: ICEMiRB. - # TIL: Things I learned. 👉 ## Making TiL go fast
0.731 - 2022/whv-250-durable-blissful-contentment/index.md - Weekly Head Voices #250: Durable, blissful contentment - This, the quarter-thousandth edition of the Weekly Head Voices, covers two 👉 The title of this post is another nerd-dad joke I probably should not have made, but which I simply could not resist.
0.735 - 2020/whv-207-kogelberg/index.md - Weekly Head Voices #207: Kogelberg - The 207th edition of the Weekly Head Voices, which in theory should cover the 👉 The title of this post is another nerd-dad joke I probably should not have made, but which I simply could not resist.
```
