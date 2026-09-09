import gzip
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

UCC_BASE = "https://ucc.ar"


@st.cache_data(show_spinner=False)
def fetch_ucc_members(fname):
    fname = re.sub(r"\s+", "", fname).lower()
    page = requests.get(f"{UCC_BASE}/_clusters/{fname}/", timeout=15)
    page.raise_for_status()
    m = re.search(r'window\.members_file\s*=\s*"([^"]+)"', page.text)
    if not m:
        raise ValueError(f"Cluster '{fname}' not found in the UCC.")
    csv_url = f"{UCC_BASE}/assets/members/membs_{m.group(1)}.csv.gz"
    resp = requests.get(csv_url, timeout=30)
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode()
    data = pd.read_csv(io.StringIO(text))
    data = data[data["name"] == fname].drop(columns="name").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No members found for '{fname}' in {csv_url}.")
    return data


st.set_page_config(page_title="Parquet 2D Plotter", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 90vw;
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .stFileUploader {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.title("Parquet 2D Plotter")

if "hide_panel" not in st.session_state:
    st.session_state.hide_panel = False

toggle_col, _ = st.columns([1, 9])
if toggle_col.button("Show panel" if st.session_state.hide_panel else "Hide panel"):
    st.session_state.hide_panel = not st.session_state.hide_panel
    st.rerun()

if not st.session_state.hide_panel:
    c_src, c_input = st.columns([1, 4])
    src_opts = ["UCC catalogue", "Upload file", "Load frame"]
    default_src = st.session_state.get("data_source", src_opts[0])
    source = c_src.radio(
        "Data source",
        src_opts,
        horizontal=True,
        index=src_opts.index(default_src),
        key="source_radio",
    )
    st.session_state.data_source = source

    n_max = c_src.number_input(
        "N_max", min_value=1, value=st.session_state.get("n_max", 50000), step=1000
    )
    st.session_state.n_max = n_max

    if source == "Upload file":
        uploaded = c_input.file_uploader(
            "Load a CSV or Parquet file", type=["csv", "parquet"]
        )
        if uploaded is not None and uploaded.file_id != st.session_state.get(
            "uploaded_file_id"
        ):
            st.cache_data.clear()
            st.session_state.uploaded_file_id = uploaded.file_id

            if uploaded.name.lower().endswith(".csv"):
                st.session_state.uploaded_df = pd.read_csv(uploaded)
            else:
                st.session_state.uploaded_df = pd.read_parquet(uploaded)
    elif source == "UCC catalogue":
        with c_input, st.form("ucc_load_form"):
            c1, c2 = st.columns([4, 1])
            fname = c1.text_input("Cluster fname (e.g. collinder107)")
            submitted = c2.form_submit_button("Load", use_container_width=True)
        if submitted and fname:
            st.cache_data.clear()
            st.session_state.sel_idx = []
            st.session_state.reset_ctr = st.session_state.get("reset_ctr", 0) + 1
            try:
                st.session_state.ucc_df = fetch_ucc_members(fname)
            except Exception as e:
                st.error(str(e))
    else:  # Load frame
        from query_gaia import query_run

        with c_input:
            with st.form("fetch_coords_form"):
                cc1, cc2 = st.columns([4, 1])
                ucc_fname = cc1.text_input(
                    "Cluster fname (autofills c_ra/c_dec from UCC)"
                )
                fetch_submitted = cc2.form_submit_button(
                    "Fetch coords", use_container_width=True
                )
            if fetch_submitted and ucc_fname:
                st.cache_data.clear()
                try:
                    members = fetch_ucc_members(ucc_fname)
                    w = members["probs"] if "probs" in members.columns else None
                    for ra_c, de_c in (
                        ("RA_ICRS", "DE_ICRS"),
                        ("RA", "DEC"),
                        ("_RAJ2000", "_DEJ2000"),
                    ):
                        if ra_c in members.columns and de_c in members.columns:
                            st.session_state.gaia_c_ra = float(
                                np.average(members[ra_c], weights=w)
                            )
                            st.session_state.gaia_c_dec = float(
                                np.average(members[de_c], weights=w)
                            )
                            break
                    else:
                        st.error("No RA/DEC columns found in member data.")
                except Exception as e:
                    st.error(str(e))
            with st.form("gaia_load_form"):
                c1, c2, c3 = st.columns(3)
                c_ra = c1.number_input(
                    "c_ra",
                    value=st.session_state.get("gaia_c_ra", 0.0),
                    format="%.6f",
                )
                c_dec = c2.number_input(
                    "c_dec",
                    value=st.session_state.get("gaia_c_dec", 0.0),
                    format="%.6f",
                )
                box_s = c3.number_input("box_s", value=0.5, format="%.4f")
                c4, c5, c6 = st.columns(3)
                gaia_data_path = c4.text_input(
                    "gaia_data_path", value="/media/gabriel/backup/gabriel/GaiaDR3"
                )
                max_mag = c5.number_input("max_mag", value=19.0, format="%.2f")
                plx_min = c6.number_input("plx_min", value=0.0, format="%.4f")
                submitted_gaia = st.form_submit_button("Load", use_container_width=True)
        if submitted_gaia:
            st.cache_data.clear()
            st.session_state.sel_idx = []
            st.session_state.reset_ctr = st.session_state.get("reset_ctr", 0) + 1
            try:
                st.session_state.gaia_df = query_run(
                    c_ra=c_ra,
                    c_dec=c_dec,
                    box_s=box_s,
                    gaia_data_path=gaia_data_path,
                    max_mag=max_mag,
                    plx_min=plx_min,
                )
            except Exception as e:
                st.error(str(e))

source = st.session_state.get("data_source", "UCC catalogue")
if source == "Upload file":
    df = st.session_state.get("uploaded_df")
elif source == "UCC catalogue":
    df = st.session_state.get("ucc_df")
else:
    df = st.session_state.get("gaia_df")

n_max = st.session_state.get("n_max", 50000)
if df is not None and len(df) > n_max:
    if "Gmag" in df.columns:
        df = df.sort_values("Gmag").iloc[:n_max].reset_index(drop=True)
    else:
        df = df.iloc[:n_max]

if df is not None:
    n_sel = len(st.session_state.get("sel_idx", []))
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns, {n_sel} selected")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns.")
    else:
        if "reset_ctr" not in st.session_state:
            st.session_state.reset_ctr = 0
        if "sel_idx" not in st.session_state:
            st.session_state.sel_idx = []
        if "sel_history" not in st.session_state:
            st.session_state.sel_history = []
        if "view_reset_ctr" not in st.session_state:
            st.session_state.view_reset_ctr = 0

        c_show, c_prob, c_clear, c_revert, c_download = st.columns([2, 1.5, 1, 1, 1.5])
        show_only_sel = c_show.checkbox(
            "Show only selected points", value=True, key="show_only_sel"
        )

        def _on_prob_change():
            st.session_state.sel_history.append(st.session_state.sel_idx)
            st.session_state.sel_idx = []
            st.session_state.reset_ctr += 1
            st.session_state.view_reset_ctr += 1

        if "probs" in df.columns:
            min_prob = c_prob.number_input(
                "Min probability",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="min_prob",
                on_change=_on_prob_change,
            )
            if min_prob > 0:
                df = df[df["probs"] >= min_prob].reset_index(drop=True)

        if c_clear.button("Clear selection", use_container_width=True):
            st.session_state.sel_history.append(st.session_state.sel_idx)
            st.session_state.sel_idx = []
            st.session_state.reset_ctr += 1
            st.session_state.view_reset_ctr += 1
            st.rerun()
        if c_revert.button(
            "Revert last selection",
            use_container_width=True,
            disabled=not st.session_state.sel_history,
        ):
            st.session_state.sel_idx = st.session_state.sel_history.pop()
            st.session_state.reset_ctr += 1
            st.rerun()

        _use_sel_dl = show_only_sel and st.session_state.sel_idx
        download_df = df.iloc[st.session_state.sel_idx] if _use_sel_dl else df
        c_download.download_button(
            "Download CSV",
            data=download_df.to_csv(index=False).encode(),
            file_name="oc_viewer_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

        def make_plot(
            key, default_x, default_y, default_color=None, default_invert=False
        ):
            c1, c2, c3, c4, c5 = st.columns(5)
            x_idx = numeric_cols.index(default_x) if default_x in numeric_cols else 0
            y_idx = (
                numeric_cols.index(default_y)
                if default_y in numeric_cols
                else min(1, len(numeric_cols) - 1)
            )
            color_opts = ["None"] + df.columns.tolist()
            color_idx = (
                color_opts.index(default_color) if default_color in color_opts else 0
            )

            vctr = st.session_state.view_reset_ctr
            x_col = c1.selectbox(
                "X axis", numeric_cols, index=x_idx, key=f"x_{key}_{vctr}"
            )
            y_col = c2.selectbox(
                "Y axis", numeric_cols, index=y_idx, key=f"y_{key}_{vctr}"
            )
            color_col = c3.selectbox(
                "Color by", color_opts, index=color_idx, key=f"c_{key}_{vctr}"
            )
            cmap = c4.selectbox(
                "Colormap",
                [
                    "Viridis",
                    "Plasma",
                    "Inferno",
                    "Magma",
                    "Cividis",
                    "Turbo",
                    "Bluered",
                    "RdBu",
                    "Jet",
                    "Rainbow",
                ],
                key=f"cmap_{key}_{vctr}",
            )
            invert_y = c5.checkbox(
                "Invert Y", value=default_invert, key=f"inv_{key}_{vctr}"
            )

            use_selection = show_only_sel and st.session_state.sel_idx
            plot_df = df.iloc[st.session_state.sel_idx] if use_selection else df

            is_numeric_color = color_col != "None" and pd.api.types.is_numeric_dtype(
                plot_df[color_col]
            )

            n_rows = len(plot_df)
            marker_opacity = float(np.clip(25 / np.sqrt(n_rows), 0.05, 0.75))
            marker_size = float(np.clip(150 / np.sqrt(n_rows), 5, 15))

            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                opacity=marker_opacity,
                color=None if color_col == "None" else color_col,
                color_continuous_scale=cmap if is_numeric_color else None,
                color_discrete_sequence=getattr(px.colors.sequential, cmap, None)
                if not is_numeric_color and color_col != "None"
                else None,
                title=f"{y_col} vs {x_col}",
            )
            fig.update_traces(
                marker={"size": marker_size, "line": {"width": 0.5, "color": "grey"}},
                selector={"mode": "markers"},
            )
            fig.update_layout(width=600, height=600)
            if invert_y:
                fig.update_yaxes(autorange="reversed")

            if st.session_state.sel_idx and not use_selection:
                sel = df.iloc[st.session_state.sel_idx]
                fig.add_scatter(
                    x=sel[x_col],
                    y=sel[y_col],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 2, "color": "black"},
                    },
                    showlegend=False,
                    hoverinfo="skip",
                )

            event = st.plotly_chart(
                fig,
                width="content",
                key=f"plot_{key}_{st.session_state.reset_ctr}",
                on_select="rerun",
                selection_mode=("points", "box", "lasso"),
            )
            index_map = st.session_state.sel_idx if use_selection else None
            return event, index_map

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            ev_a, map_a = make_plot("A", "GLON", "GLAT", "probs")
        with r1c2:
            ev_b, map_b = make_plot("B", "pmRA", "pmDE", "probs")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            ev_c, map_c = make_plot("C", "Plx", "Gmag", "probs", default_invert=True)
        with r2c2:
            ev_d, map_d = make_plot("D", "BP-RP", "Gmag", "probs", default_invert=True)

        for event, index_map in (
            (ev_a, map_a),
            (ev_b, map_b),
            (ev_c, map_c),
            (ev_d, map_d),
        ):
            if event and event.selection and event.selection.point_indices:
                new_sel = event.selection.point_indices
                if index_map is not None:
                    new_sel = [index_map[i] for i in new_sel]
                if new_sel != st.session_state.sel_idx:
                    st.session_state.sel_history.append(st.session_state.sel_idx)
                    st.session_state.sel_idx = new_sel
                    st.rerun()
                break
else:
    st.info(
        "Upload a CSV/Parquet file or load a cluster from the UCC catalogue to begin."
    )
