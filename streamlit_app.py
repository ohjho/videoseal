import os, sys, json
import pandas as pd
import streamlit as st
from inference_streaming_helpers import get_signature


@st.cache_data
def get_watermark(vid_url, max_frames: int = None):
    wm = get_signature(vid_url, max_frames=max_frames)
    return wm


def show_original(
    df, signature, st_asset, id_col="media_id", source_col="hq_download_link"
):
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col, drop=True).copy()
    with st_asset:
        if signature.startswith("gf:"):
            gf_id = signature.replace("gf:", "")
            if gf_id in df.index.tolist():
                org_vid_url = df.at[gf_id, source_col]
                st.caption(f"matched {signature} to {source_col}: {org_vid_url}")
                st.video(org_vid_url, autoplay=True, muted=True)
            else:
                st.warning(f"cannot find {gf_id} in {id_col}")
        else:
            st.warning(f"signature {signature} is not recognized.")


def Main():
    # App title
    st.set_page_config(
        page_title="VideoSeal",
        page_icon="http://www.google.com/s2/favicons?domain=greenfly.com",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.logo(
        "https://www.greenfly.com/wp-content/themes/greenfly/assets/images/greenfly-logo-black-R.svg",
        icon_image="http://www.google.com/s2/favicons?domain=greenfly.com&sz=128",
        size="large",
    )
    st.title("Video Seal Demo")
    # show_readme(st.sidebar)
    video_url = st.text_input(
        "Video URL",
        # value="https://miro-ps-bucket-copy.s3.us-west-2.amazonaws.com/storage/jho/test_videos/finishline_videos/hc-charlotte-2025/rt103124_tracks-dx39-faceanchors-v3-matched-cossim63.csv",
        # help="dataset should be created following [these instructions](https://improved-eureka-7ke59kk.pages.github.io/notebooks/vinyasa-face-anchor-matching.html#tracks-face-anchors)",
    )
    max_frames = (
        None
        if st.sidebar.toggle("Analyze Full Videos")
        else st.sidebar.slider("Max Frames", min_value=16, max_value=160, step=16)
    )
    csv_url = st.sidebar.text_input(
        "Reference Video Dataset CSV URL",
        value="https://miro-ps-bucket-copy.s3.us-west-2.amazonaws.com/storage/jho/test_videos/videoseal/nfl_video_test_apr2025.csv",
        help="reference video dataset",
    )
    df = pd.read_csv(csv_url) if csv_url else None

    if video_url:
        cols = st.columns(3)
        cols[0].video(video_url, muted=True)
        if cols[1].button("Get Watermark"):
            wm = get_watermark(video_url, max_frames=max_frames)
            with cols[1].expander("signature", expanded=True):
                st.write(wm["signature"])
            with cols[1].expander("soft message", expanded=False):
                st.write(wm["soft_message"])

            if isinstance(df, pd.DataFrame):
                show_original(df, signature=wm["signature"], st_asset=cols[2])
    else:
        st.warning(f":point_left: please provide your data")


if __name__ == "__main__":
    Main()
