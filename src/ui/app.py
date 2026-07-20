from __future__ import annotations

import os
import requests
# pyrefly: ignore [missing-import]
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8008").rstrip("/")

st.set_page_config(page_title="Lumos", layout="wide")

st.title("Lumos")
st.caption("Multimodal Retrieval • OpenCLIP + Qdrant + FastAPI + Streamlit")


@st.cache_data(show_spinner=False)
def api_health() -> dict:
    r = requests.get(f"{API_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def fetch_image_bytes(image_url: str) -> bytes:
    """
    image_url: API-relative path like '/image/xxx.jpg' OR absolute 'http://.../image/xxx.jpg'
    """
    if image_url.startswith("/"):
        url = f"{API_URL}{image_url}"
    else:
        url = image_url

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def render_results(results, max_cols: int = 5):
    if not results:
        st.info("No results")
        return

    cols = st.columns(min(max_cols, len(results)))
    for i, item in enumerate(results):
        col = cols[i % len(cols)]
        with col:
            try:
                img_bytes = fetch_image_bytes(item["image_url"])
                st.image(
                    img_bytes,
                    caption=f"#{i+1} score={item['score']:.3f}\n{item['filename']}",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Failed to load image: {item.get('filename','?')} ({e})")

            cap = item.get("caption0")
            if cap:
                st.caption(cap)


with st.expander("System status", expanded=False):
    try:
        st.json(api_health())
    except Exception as e:
        st.error(f"API not reachable: {e}")


tab1, tab2, tab3 = st.tabs(["Text → Image", "Image → Image (upload)", "Similar by image_id"])

# ---------------------------
# Text -> Image
# ---------------------------
with tab1:
    colA, colB = st.columns([2, 1])
    with colA:
        query = st.text_input("Text query", value="a dog running on the grass")
    with colB:
        top_k = st.number_input("top_k", min_value=1, max_value=50, value=5, step=1)

    if st.button("Search (text)", type="primary"):
        try:
            payload = {"query": query, "top_k": int(top_k)}
            r = requests.post(f"{API_URL}/search_text", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            results = data["results"]

            st.write(f"Results: {len(results)}")
            render_results(results, max_cols=5)

        except Exception as e:
            st.error(str(e))

# ---------------------------
# Image -> Image (upload)
# ---------------------------
with tab2:
    colA, colB = st.columns([2, 1])
    with colA:
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    with colB:
        top_k2 = st.number_input("top_k ", min_value=1, max_value=50, value=6, step=1)

    if st.button("Search (image upload)", type="primary", disabled=(up is None)):
        try:
            files = {"file": (up.name, up.getvalue(), up.type)}
            r = requests.post(f"{API_URL}/search_image?top_k={int(top_k2)}", files=files, timeout=120)
            r.raise_for_status()
            data = r.json()
            results = data["results"]

            st.write(f"Results: {len(results)}")
            render_results(results, max_cols=5)

        except Exception as e:
            st.error(str(e))

# ---------------------------
# Similar by image_id
# ---------------------------
with tab3:
    colA, colB = st.columns([2, 1])
    with colA:
        image_id = st.number_input("image_id", min_value=0, max_value=1000000, value=0, step=1)
    with colB:
        top_k3 = st.number_input("top_k  ", min_value=1, max_value=50, value=6, step=1)

    if st.button("Search similar", type="primary"):
        try:
            r = requests.get(f"{API_URL}/similar_image/{int(image_id)}?top_k={int(top_k3)}", timeout=60)
            r.raise_for_status()
            data = r.json()
            results = data["results"]

            st.write(f"Results: {len(results)}")
            render_results(results, max_cols=5)

        except Exception as e:
            st.error(str(e))