# AI Photo Search

Search a folder of photos using a text description or a reference image. Built with Python and CLIP ViT-B/32, with cosine similarity ranking and a local results gallery.

## Features

- Search photos using natural-language descriptions.
- Find visually similar photos using a reference image.
- Cache image embeddings to avoid processing the collection on every search.
- View the top three matches in a Matplotlib gallery.

## Setup

Requires Python 3.10 or newer.

```powershell
git clone https://github.com/Techwithpraisejames/ai-photo-search-.git
cd ai-photo-search-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Copy your own JPG, JPEG, or PNG images into the `photos/` folder and run:

```powershell
python search.py
```

Choose **1** to search with text, **2** to search using a reference image path, or **3** to exit. The results appear in a separate window. Close the window to return to the menu.

For image search, enter a path inside `photos/` or an absolute path to an image elsewhere on your computer. If the reference is already in the collection, that file is excluded from the results.

## How it works

CLIP encodes each photo into a vector. The script normalizes the vectors and saves them to `photo_embeddings.pt`. It reuses the cache until the photo collection changes. A search query is encoded with CLIP's text or image encoder, then compared with the stored vectors using cosine similarity.

Similarity scores represent relative matches, not confidence percentages. Search quality depends on the images and descriptions.

## Privacy and limitations

The application runs locally after downloading model weights from Hugging Face. Personal photos, cached embeddings, and virtual environments are excluded from Git. The script supports JPG, JPEG, and PNG. Cache invalidation uses filenames, modification times, and file sizes rather than file hashes.

## Built with

Python, PyTorch, Hugging Face Transformers, Pillow, and Matplotlib.
