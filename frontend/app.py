import io
import time
from typing import Optional, Tuple

import requests
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# For real-time mode (serial streaming)
try:
    import serial  # pyserial
except ImportError:
    serial = None

# ======================================
#        GLOBAL CONFIG (SIGNAL)
# ======================================

SAMPLE_RATE = 50_000.0  # Hz (fixed in data_collection.py)
N_SAMPLES = 1024  # samples per axis (fixed in data_collection.py)
N_FFT_BINS = N_SAMPLES // 2 + 1  # rfft -> 513

# Serial defaults for real-time mode (same as config.py)
DEFAULT_SERIAL_PORT = "COM6"
DEFAULT_SERIAL_BAUDRATE = 921600
DEFAULT_SERIAL_TIMEOUT = 0.1  # seconds

# ======================================
#        STREAMLIT PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Rotating Machine Fault Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =======================
#       GLOBAL STYLES
# =======================

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #0066cc;
        color: white;
        padding: 14px 40px;
        border: none;
        border-radius: 8px;
        font-size: 20px;
        font-weight: 600;
        cursor: pointer;
        margin: auto;
        display: block;
    }
    div.stButton > button:first-child:hover {
        background-color: #0052a3;
    }
    .prediction-box {
        padding: 18px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }
    .normal-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fault-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================
#        CONFIG - API ENDPOINT
# ======================================

BACKEND_BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = "/predict"  # single endpoint


# ======================================
#        Helper Functions (shared)
# ======================================


def extract_waveform_and_fft_from_df(df: pd.DataFrame):
    """
    From a dataframe like:

        timestamp,axis,1,2,...,1024
        2025-11-14 16:34:39,X,...
        2025-11-14 16:34:39,Y,...
        2025-11-14 16:34:39,Z,...

    Returns:
        time_waveform: np.ndarray of shape (1024, 3)
        fft_magnitude: np.ndarray of shape (513, 3)
    """
    required_axes = {"X", "Y", "Z"}

    if "axis" not in df.columns:
        raise ValueError("CSV must contain an 'axis' column.")

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")

    axes_present = set(df["axis"].unique())
    if not required_axes.issubset(axes_present):
        raise ValueError(
            f"CSV must contain at least one row for each of X, Y, Z. "
            f"Found axes: {axes_present}"
        )

    # Feature columns: everything except timestamp and axis
    feature_cols = [c for c in df.columns if c not in ("timestamp", "axis")]

    # Expect N_SAMPLES samples per axis
    if len(feature_cols) != N_SAMPLES:
        raise ValueError(
            f"Expected {N_SAMPLES} sample columns, found {len(feature_cols)}. "
            f"Header should be: 'timestamp,axis,1,2,...,{N_SAMPLES}'."
        )

    # Order rows by axis X, Y, Z
    rows = {}
    for ax in ["X", "Y", "Z"]:
        row_df = df[df["axis"] == ax]
        if row_df.empty:
            raise ValueError(f"No row found for axis {ax}.")
        # Take the first row for this axis
        rows[ax] = row_df.iloc[0][feature_cols].to_numpy(dtype=np.float32)

    # Stack into channels_first: (3, 1024)
    stacked_ch_first = np.vstack([rows["X"], rows["Y"], rows["Z"]])  # (3, N_SAMPLES)

    # Convert to channels_last for the model: (timesteps, channels) = (1024, 3)
    time_waveform = stacked_ch_first.T  # (N_SAMPLES, 3)

    # FFT magnitude per channel: (3, 1024) -> (3, 513) -> (513, 3)
    fft_ch_first = np.abs(
        np.fft.rfft(stacked_ch_first, n=N_SAMPLES, axis=1)
    )  # (3, N_FFT_BINS)
    fft_magnitude = fft_ch_first.T  # (N_FFT_BINS, 3)

    return time_waveform, fft_magnitude


def compute_signal_metrics(
    time_waveform: np.ndarray,
    fft_magnitude: np.ndarray,
):
    """
    Compute basic vibration metrics per axis.

    Note: time_waveform is already DC-removed at collection time
    (or by parse_line in the real-time case).
    """
    axis_labels = ["X", "Y", "Z"]
    n_samples = time_waveform.shape[0]

    if n_samples != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples, got {n_samples}")

    if fft_magnitude.shape[0] != N_FFT_BINS:
        raise ValueError(
            f"Expected {N_FFT_BINS} FFT bins, got {fft_magnitude.shape[0]}"
        )

    # Frequencies corresponding to rfft bins (0 .. Nyquist)
    freqs_hz = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)

    rows = []
    for i, axis_name in enumerate(axis_labels):
        sig = time_waveform[:, i]

        # Classic vibration metrics
        rms = float(np.sqrt(np.mean(sig**2)))
        mean_val = float(np.mean(sig))  # should be ~0.0
        std_val = float(np.std(sig))
        peak = float(np.max(np.abs(sig)))
        peak_to_peak = float(np.max(sig) - np.min(sig))
        crest_factor = float(peak / rms) if rms > 0 else np.nan

        # Dominant frequency: ignore DC (bin 0)
        mag = fft_magnitude[:, i]
        dom_idx = int(np.argmax(mag[1:]) + 1)
        dom_freq_hz = float(freqs_hz[dom_idx])

        rows.append(
            {
                "Axis": axis_name,
                "RMS": rms,
                "Mean (should ≈ 0)": mean_val,
                "Std Dev": std_val,
                "Peak": peak,
                "Peak-to-Peak": peak_to_peak,
                "Crest Factor": crest_factor,
                "Dominant Freq (Hz)": dom_freq_hz,
            }
        )

    metrics_df = pd.DataFrame(rows)

    # Overall vibration level (vector RMS over axes)
    rms_x = metrics_df.loc[metrics_df["Axis"] == "X", "RMS"].item()
    rms_y = metrics_df.loc[metrics_df["Axis"] == "Y", "RMS"].item()
    rms_z = metrics_df.loc[metrics_df["Axis"] == "Z", "RMS"].item()
    overall_rms_vec = float(np.sqrt(rms_x**2 + rms_y**2 + rms_z**2))

    overall = {
        "Overall RMS (vector)": overall_rms_vec,
    }

    return metrics_df, overall


def call_fastapi_predict(
    time_waveform: np.ndarray,
    fft_magnitude: np.ndarray,
):
    """
    Call the FastAPI backend with the time_waveform and fft_magnitude.

    - time_waveform: (1024, 3)
    - fft_magnitude: (513, 3)
    """
    url = BACKEND_BASE_URL + PREDICT_ENDPOINT

    payload = {
        "time_waveform": time_waveform.tolist(),
        "fft_magnitude": fft_magnitude.tolist(),
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"API call failed with status {response.status_code}: {response.text}"
        )

    return response.json()


def render_metrics_and_plots(time_waveform: np.ndarray, fft_magnitude: np.ndarray):
    """Shared rendering of metrics + time/FFT plots (for individual mode)."""
    st.subheader("Signal metrics (per axis)")
    metrics_df, overall_metrics = compute_signal_metrics(
        time_waveform,
        fft_magnitude,
    )

    st.dataframe(
        metrics_df.set_index("Axis"),
        use_container_width=True,
    )

    st.markdown("**Overall vibration metrics**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Overall RMS (vector)",
        f"{overall_metrics['Overall RMS (vector)']:.4g}",
    )
    col2.metric(
        "RMS X",
        f"{metrics_df.loc[metrics_df['Axis']=='X','RMS'].item():.4g}",
    )
    col3.metric(
        "RMS Y",
        f"{metrics_df.loc[metrics_df['Axis']=='Y','RMS'].item():.4g}",
    )
    col4.metric(
        "RMS Z",
        f"{metrics_df.loc[metrics_df['Axis']=='Z','RMS'].item():.4g}",
    )

    st.markdown("---")
    st.subheader("Time Waveform and FFT per Axis")

    axis_labels = ["X", "Y", "Z"]
    freqs_hz = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)

    for i, axis_name in enumerate(axis_labels):
        st.markdown(f"### {axis_name}-Axis")

        col_time, col_fft = st.columns(2)

        # -------- TIME WAVEFORM (left) --------
        with col_time:
            wf_df = pd.DataFrame(
                {
                    "Sample": np.arange(time_waveform.shape[0]),
                    "Amplitude": time_waveform[:, i],
                }
            )

            time_chart = (
                alt.Chart(wf_df)
                .mark_line()
                .encode(
                    x=alt.X("Sample:Q", title="Sample index"),
                    y=alt.Y("Amplitude:Q", title="Amplitude (DC-removed)"),
                )
                .properties(
                    width="container",
                    height=400,
                    title=f"Time Waveform ({axis_name}-axis)",
                )
                .interactive()
            )

            st.altair_chart(time_chart, use_container_width=True)

        # -------- FFT MAGNITUDE (right) --------
        with col_fft:
            fft_df = pd.DataFrame(
                {
                    "Frequency (Hz)": freqs_hz,
                    "Magnitude": fft_magnitude[:, i],
                }
            )

            fft_chart = (
                alt.Chart(fft_df)
                .mark_line()
                .encode(
                    x=alt.X(
                        "Frequency (Hz):Q",
                        title="Frequency (Hz)",
                    ),
                    y=alt.Y("Magnitude:Q", title="Magnitude"),
                )
                .properties(
                    width="container",
                    height=400,
                    title=f"FFT Magnitude ({axis_name}-axis)",
                )
                .interactive()
            )

            st.altair_chart(fft_chart, use_container_width=True)

        st.markdown("---")


def render_prediction_result(result: dict):
    """Shared rendering of prediction result (summary box + probabilities)."""
    raw_label = result["predicted_class"]

    PRETTY_LABELS = {
        "normal": "normal",
        "bearing_fault": "BEARING FAULT",
        "unbalance_fault": "UNBALANCE",
        "misalignment_fault": "MISALIGNMENT",
        "mechanical_looseness_fault": "MECHANICAL LOOSENESS",
    }

    pred_label = PRETTY_LABELS.get(raw_label, raw_label)

    # ==============================
    #   Summary box
    # ==============================
    if pred_label.lower() == "normal":
        st.markdown(
            '<div class="prediction-box normal-box">'
            "✅ Normal condition detected."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="prediction-box fault-box">'
            f"⚠️ Fault detected: <b>{pred_label}!</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ==============================
    #   Probabilities bar chart
    # ==============================
    probs_dict = result.get("probabilities", {})

    if not probs_dict:
        st.info("No probability information returned by the API.")
        return

    prob_df = (
        pd.Series(probs_dict, name="Probability").rename_axis("Class").reset_index()
    )

    prob_df["Pretty Class"] = (
        prob_df["Class"]
        .map(
            {
                "normal": "Normal",
                "bearing_fault": "Bearing Fault",
                "unbalance_fault": "Unbalance",
                "misalignment_fault": "Misalignment",
                "mechanical_looseness_fault": "Looseness",
            }
        )
        .fillna(prob_df["Class"])
    )

    prob_df = prob_df.sort_values("Probability", ascending=False)

    st.subheader("Prediction probabilities")

    df_display = (
        prob_df[["Pretty Class", "Probability"]]
        .rename(columns={"Pretty Class": "Class"})
        .copy()
    )

    df_display["Probability"] = df_display["Probability"].apply(lambda x: f"{x:.3e}")

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

    CLASS_ORDER = [
        "Normal",
        "Bearing Fault",
        "Unbalance",
        "Misalignment",
        "Looseness",
    ]

    chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Pretty Class:N",
                sort=CLASS_ORDER,
                title="Class",
            ),
            y=alt.Y(
                "Probability:Q",
                title="Probability",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("Pretty Class:N", title="Class"),
                alt.Tooltip("Probability:Q", format=".3f"),
            ],
        )
        .properties(
            width="container",
            height=400,
            title="Class probabilities",
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# ======================================
#   Helper functions for REAL-TIME mode
# ======================================


def parse_serial_line(line_bytes: bytes) -> Optional[Tuple[str, np.ndarray]]:
    """
    Parse a single line of bytes from the serial port.

    Expected format (same as data_collection.py):

        b"<device_id> <axis> <hex1> <hex2> ... <hex1024>\\r\\n"

    Returns:
        (axis, wave_no_dc) on success
        None on failure
    """
    try:
        line = line_bytes.decode(errors="ignore").strip()
        if not line:
            return None

        parts = line.split()
        if len(parts) < 2 + N_SAMPLES:
            # Too short
            return None

        axis = parts[1]
        if axis not in ("X", "Y", "Z"):
            return None

        hex_samples = parts[2 : 2 + N_SAMPLES]

        # Parse hex to integers
        wave_array = np.array(
            [int(s, 16) for s in hex_samples],
            dtype=np.uint32,
        )

        # DC removal: convert to float and subtract mean
        wave_float = wave_array.astype(np.float64)
        wave_no_dc = wave_float - np.mean(wave_float)

        return axis, wave_no_dc.astype(np.float32)

    except Exception:
        # Ignore malformed lines
        return None


def read_xyz_triplet_from_serial(ser: "serial.Serial") -> Tuple[np.ndarray, np.ndarray]:
    """
    Block until a complete X-Y-Z triplet is read from the serial stream.

    Returns:
        time_waveform: (1024, 3)
        fft_magnitude: (513, 3)
    """
    axis_order = ["X", "Y", "Z"]
    axis_index = 0  # expecting X
    wave_buffer = {}

    while True:
        line_bytes = ser.readline()
        if not line_bytes:
            continue  # timeout, keep reading

        parsed = parse_serial_line(line_bytes)
        if parsed is None:
            continue

        axis, wave_no_dc = parsed
        expected_axis = axis_order[axis_index]

        if axis == "X" and axis_index == 0:
            # Start a new group
            wave_buffer = {"X": wave_no_dc}
            axis_index = 1  # next expect Y

        elif axis == "Y" and axis_index == 1:
            wave_buffer["Y"] = wave_no_dc
            axis_index = 2  # next expect Z

        elif axis == "Z" and axis_index == 2:
            wave_buffer["Z"] = wave_no_dc

            # We have X, Y, Z: assemble
            stacked_ch_first = np.vstack(
                [wave_buffer["X"], wave_buffer["Y"], wave_buffer["Z"]]
            )  # (3, 1024)

            time_waveform = stacked_ch_first.T.astype(np.float32)  # (1024, 3)

            fft_ch_first = np.abs(
                np.fft.rfft(stacked_ch_first, n=N_SAMPLES, axis=1)
            )  # (3, 513)
            fft_magnitude = fft_ch_first.T.astype(np.float32)  # (513, 3)

            return time_waveform, fft_magnitude

        else:
            # Out-of-order axis
            if axis == "X":
                # Start a new group from this X
                wave_buffer = {"X": wave_no_dc}
                axis_index = 1
            else:
                # Reset and wait for the next X
                wave_buffer = {}
                axis_index = 0


def realtime_prediction_loop(
    port: str,
    baudrate: int,
    timeout: float,
    update_interval: float,
):
    """
    Main loop for continuous real-time prediction.

    - Opens the serial port.
    - Repeatedly reads X-Y-Z triplets.
    - For each triplet:
        - Computes waveform + FFT.
        - Calls FastAPI /predict.
        - Updates the UI with metrics, plots, and prediction.

    This function returns when st.session_state["streaming"] is set to False.
    """
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    charts_placeholder = st.empty()
    prediction_placeholder = st.empty()

    if serial is None:
        status_placeholder.error(
            "pyserial is not installed. Please install it with `pip install pyserial`."
        )
        return

    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
    except Exception as e:
        status_placeholder.error(f"Failed to open serial port {port}: {e}")
        return

    status_placeholder.info(f"Streaming from {port} at {baudrate} baud...")

    axis_labels = ["X", "Y", "Z"]
    freqs_hz = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)

    try:
        while st.session_state.get("streaming", False):
            # 1. Get next X-Y-Z triplet from serial
            time_waveform, fft_magnitude = read_xyz_triplet_from_serial(ser)

            # 2. Compute metrics
            metrics_df, overall_metrics = compute_signal_metrics(
                time_waveform,
                fft_magnitude,
            )

            # ---------- METRICS ----------
            with metrics_placeholder.container():
                st.subheader("Latest Signal Metrics (per axis)")
                st.dataframe(
                    metrics_df.set_index("Axis"),
                    use_container_width=True,
                )

                st.markdown("**Latest Overall Vibration Metrics**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Overall RMS (vector)",
                    f"{overall_metrics['Overall RMS (vector)']:.4g}",
                )
                col2.metric(
                    "RMS X",
                    f"{metrics_df.loc[metrics_df['Axis']=='X','RMS'].item():.4g}",
                )
                col3.metric(
                    "RMS Y",
                    f"{metrics_df.loc[metrics_df['Axis']=='Y','RMS'].item():.4g}",
                )
                col4.metric(
                    "RMS Z",
                    f"{metrics_df.loc[metrics_df['Axis']=='Z','RMS'].item():.4g}",
                )

            # ---------- WAVEFORM + FFT PLOTS ----------
            with charts_placeholder.container():
                st.subheader("Latest Time Waveform and FFT per Axis")

                for i, axis_name in enumerate(axis_labels):
                    st.markdown(f"### {axis_name}-Axis")

                    col_time, col_fft = st.columns(2)

                    # Time waveform
                    with col_time:
                        wf_df = pd.DataFrame(
                            {
                                "Sample": np.arange(time_waveform.shape[0]),
                                "Amplitude": time_waveform[:, i],
                            }
                        )

                        time_chart = (
                            alt.Chart(wf_df)
                            .mark_line()
                            .encode(
                                x=alt.X("Sample:Q", title="Sample index"),
                                y=alt.Y("Amplitude:Q", title="Amplitude (DC-removed)"),
                            )
                            .properties(
                                width="container",
                                height=300,
                                title=f"Time Waveform ({axis_name}-axis)",
                            )
                            .interactive()
                        )

                        st.altair_chart(time_chart, use_container_width=True)

                    # FFT magnitude
                    with col_fft:
                        fft_df = pd.DataFrame(
                            {
                                "Frequency (Hz)": freqs_hz,
                                "Magnitude": fft_magnitude[:, i],
                            }
                        )

                        fft_chart = (
                            alt.Chart(fft_df)
                            .mark_line()
                            .encode(
                                x=alt.X(
                                    "Frequency (Hz):Q",
                                    title="Frequency (Hz)",
                                ),
                                y=alt.Y("Magnitude:Q", title="Magnitude"),
                            )
                            .properties(
                                width="container",
                                height=300,
                                title=f"FFT Magnitude ({axis_name}-axis)",
                            )
                            .interactive()
                        )

                        st.altair_chart(fft_chart, use_container_width=True)

                    st.markdown("---")

            # 3. Call API
            try:
                result = call_fastapi_predict(time_waveform, fft_magnitude)
            except Exception as e:
                with prediction_placeholder.container():
                    st.subheader("Latest Prediction")
                    st.error(f"API call failed: {e}")
                time.sleep(update_interval)
                continue

            # 4. Render prediction
            with prediction_placeholder.container():
                st.subheader("Latest Prediction")
                render_prediction_result(result)

            # Small pause so the UI can update
            time.sleep(update_interval)

    finally:
        try:
            ser.close()
        except Exception:
            pass
        status_placeholder.info("Real-time streaming stopped.")


# ======================================
#            Streamlit UI
# ======================================

st.title("Rotating Machine Fault Classification")

# Mode selection: Individual vs Real-Time
mode = st.radio(
    "Select mode:",
    ("Individual Prediction (CSV upload)", "Continuous Real-Time Prediction"),
)

st.markdown("---")

# -----------------------------------------------------------------------------
#   MODE 1: INDIVIDUAL PREDICTION (existing CSV-upload workflow)
# -----------------------------------------------------------------------------
if mode == "Individual Prediction (CSV upload)":
    st.markdown("#### Data")
    st.write(
        "Upload a CSV file containing **a _single_ X-Y-Z triplet** in the following format:"
    )
    st.code(
        "timestamp,axis,1,2,...,1024\n"
        "2025-11-14 16:34:39,X,-69.38,...,223.14\n"
        "2025-11-14 16:34:39,Y,-57.57,...,421.52\n"
        "2025-11-14 16:34:39,Z,-10.13,...,988.96\n",
        language="text",
    )
    st.info(
        "**Note:** During data collection, each waveform is **DC-removed** (mean subtracted) "
        "before being written to a CSV file."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV
            bytes_data = uploaded_file.read()
            text = bytes_data.decode("utf-8")  # adjust encoding if needed
            df = pd.read_csv(io.StringIO(text))

            st.subheader("Preview of uploaded data")
            st.dataframe(df.head())

            # Extract waveform and FFT
            time_waveform, fft_magnitude = extract_waveform_and_fft_from_df(df)

            st.write("✅ Data parsed successfully!")
            st.write(
                f"Time waveform shape: `{time_waveform.shape}` (expected: `({N_SAMPLES}, 3)`)"
            )
            st.write(
                f"FFT magnitude shape: `{fft_magnitude.shape}` (expected: `({N_FFT_BINS}, 3)`)"
            )
            st.write(
                f"Sampling rate: `{SAMPLE_RATE:.0f} Hz` "
                f"(Nyquist: `{SAMPLE_RATE/2:.0f} Hz`)"
            )

            # Signal metrics + plots
            render_metrics_and_plots(time_waveform, fft_magnitude)

            # ==================================
            #       Prediction section
            # ==================================
            col_left, col_center, col_right = st.columns([1, 1, 1])

            with col_center:
                if st.button("Run Prediction", use_container_width=True):
                    with st.spinner("Calling API and running model..."):
                        result = call_fastapi_predict(
                            time_waveform,
                            fft_magnitude,
                        )
                    render_prediction_result(result)

        except Exception as e:
            st.error(f"Error processing file or calling API: {e}")

# -----------------------------------------------------------------------------
#   MODE 2: CONTINUOUS REAL-TIME PREDICTION (serial streaming)
# -----------------------------------------------------------------------------
else:
    st.markdown("#### Real-Time Streaming")
    st.write(
        "In this mode, the app connects directly to the sensor over **serial**, "
        "continuously reads X-Y-Z triplets in hex format "
        "and sends each triplet to the FastAPI backend for prediction."
    )

    if serial is None:
        st.error(
            "pyserial is not installed. Please install it with `pip install pyserial` "
            "before using real-time mode."
        )
    else:
        # Keep a flag in session_state to control streaming
        if "streaming" not in st.session_state:
            st.session_state["streaming"] = False

        col1, col2, col3 = st.columns(3)
        with col1:
            port = st.text_input("Serial port", value=DEFAULT_SERIAL_PORT)
        with col2:
            baudrate = st.number_input(
                "Baudrate",
                value=DEFAULT_SERIAL_BAUDRATE,
                step=115200,
            )
        with col3:
            update_interval = st.number_input(
                "Update interval (seconds)",
                value=0.5,
                min_value=0.1,
                max_value=5.0,
                step=0.1,
            )

        st.markdown(
            "When streaming is **ON**, the app will:\n"
            "- Read a complete X-Y-Z triplet from the serial line.\n"
            "- Compute time-waveform and FFT (same preprocessing as training).\n"
            "- Call the FastAPI `/predict` endpoint.\n"
            "- Update the metrics and prediction on this page.\n"
        )

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("Start Real-Time Prediction"):
                st.session_state["streaming"] = True
        with stop_col:
            if st.button("Stop"):
                st.session_state["streaming"] = False

        # Run the loop if streaming is enabled
        if st.session_state["streaming"]:
            realtime_prediction_loop(
                port=port,
                baudrate=int(baudrate),
                timeout=float(DEFAULT_SERIAL_TIMEOUT),
                update_interval=float(update_interval),
            )
