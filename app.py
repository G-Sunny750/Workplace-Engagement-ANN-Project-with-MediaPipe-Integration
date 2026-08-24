import os
import time
import threading
from pathlib import Path
from collections import defaultdict

import av
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Workplace Engagement Monitor",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Workplace Engagement Monitoring")

st.caption(
    "Live webcam monitoring → MediaPipe facial features → "
    "V4 ANN prediction → session engagement report"
)


# ============================================================
# PROJECT FILES
# V4 ONLY
#
# Keep these files in the same folder as app.py
# ============================================================

# ============================================================
# MODEL FILES
# ============================================================

# ============================================================
# PROJECT FILES
# V4 ONLY
# ============================================================

# Folder containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Folder containing all trained model artifacts
MODEL_DIR = BASE_DIR / "models"

# MediaPipe task model
TASK_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

# V4 ANN model
V4_MODEL_PATH = (
    MODEL_DIR / "workplace_engagement_ann_v4.keras"
)

# V4 scaler
V4_SCALER_PATH = (
    MODEL_DIR / "workplace_engagement_scaler_v4.pkl"
)

# V4 configuration
V4_CONFIG_PATH = (
    MODEL_DIR / "workplace_engagement_v4_config.pkl"
)

# Recording folder
RECORDING_DIR = BASE_DIR / "recordings"
RECORDING_DIR.mkdir(exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES_DEFAULT = {
    0: "Low Engagement",
    1: "Moderate Engagement",
    2: "High Engagement",
}


# ============================================================
# 17 FEATURES
#
# These are the features extracted from MediaPipe.
# The V4 config determines which saved feature columns
# are actually passed into the V4 scaler/model.
# ============================================================

FEATURE_NAMES_17 = [
    "left_ear",
    "right_ear",
    "average_ear",
    "ear_asymmetry",
    "mar",
    "normalized_mouth_opening",
    "normalized_mouth_width",
    "normalized_eye_distance",
    "face_aspect_ratio",
    "pitch",
    "yaw",
    "roll",
    "gaze_x",
    "gaze_y",
    "gaze_asymmetry",
    "nose_vertical",
    "chin_vertical",
]


# ============================================================
# MEDIAPIPE LANDMARK DEFINITIONS
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144,
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380,
]

LEFT_IRIS = [
    468,
    469,
    470,
    471,
    472,
]

RIGHT_IRIS = [
    473,
    474,
    475,
    476,
    477,
]

NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

UPPER_LIP = 13
LOWER_LIP = 14

MOUTH_LEFT_INNER = 78
MOUTH_RIGHT_INNER = 308


# ============================================================
# SESSION STATE
# ============================================================

if "session_reset_id" not in st.session_state:
    st.session_state.session_reset_id = 0

if "last_recording_path" not in st.session_state:
    st.session_state.last_recording_path = None


# ============================================================
# V4 MODEL LOADING
# ============================================================

@st.cache_resource
def load_pipeline():
    """
    Load ONLY the V4 workplace engagement pipeline.

    V4 artifacts:
        workplace_engagement_ann_v4.keras
        workplace_engagement_scaler_v4.pkl
        workplace_engagement_v4_config.pkl

    No V5 model is loaded or referenced.
    """

    missing_files = []

    if not V4_MODEL_PATH.exists():
        missing_files.append(
            V4_MODEL_PATH.name
        )

    if not V4_SCALER_PATH.exists():
        missing_files.append(
            V4_SCALER_PATH.name
        )

    if not V4_CONFIG_PATH.exists():
        missing_files.append(
            V4_CONFIG_PATH.name
        )

    if missing_files:
        raise FileNotFoundError(
            "Missing V4 deployment file(s):\n\n"
            + "\n".join(
                f"- {name}"
                for name in missing_files
            )
            + "\n\nAll V4 files must be in the same "
              "folder as app.py."
        )

    # --------------------------------------------------------
    # Load V4 ANN model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        V4_MODEL_PATH,
        compile=False,
    )

    # --------------------------------------------------------
    # Load V4 scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        V4_SCALER_PATH
    )

    # --------------------------------------------------------
    # Load V4 configuration
    # --------------------------------------------------------

    config = joblib.load(
        V4_CONFIG_PATH
    )

    if config is None:
        config = {}

    # --------------------------------------------------------
    # Determine expected feature names
    # --------------------------------------------------------

    configured_feature_names = config.get(
        "feature_names",
        FEATURE_NAMES_17,
    )

    if configured_feature_names is None:
        configured_feature_names = FEATURE_NAMES_17

    configured_feature_names = list(
        configured_feature_names
    )

    # --------------------------------------------------------
    # Validate scaler
    # --------------------------------------------------------

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    if scaler_features is not None:
        if scaler_features != len(
            configured_feature_names
        ):
            raise ValueError(
                "V4 configuration/scaler mismatch.\n\n"
                f"Config expects "
                f"{len(configured_feature_names)} features, "
                f"but the V4 scaler expects "
                f"{scaler_features} features.\n\n"
                "Check workplace_engagement_v4_config.pkl "
                "and workplace_engagement_scaler_v4.pkl."
            )

    # --------------------------------------------------------
    # Validate model input dimension
    # --------------------------------------------------------

    try:
        model_input_shape = model.input_shape

        if isinstance(
            model_input_shape,
            list,
        ):
            model_input_shape = (
                model_input_shape[0]
            )

        model_features = (
            model_input_shape[-1]
        )

        if (
            model_features is not None
            and model_features
            != len(configured_feature_names)
        ):
            raise ValueError(
                "V4 configuration/model mismatch.\n\n"
                f"Config expects "
                f"{len(configured_feature_names)} features, "
                f"but the V4 ANN expects "
                f"{model_features} features."
            )

    except Exception as exc:

        # Do not hide an actual validation error.
        if isinstance(
            exc,
            ValueError,
        ):
            raise

    return {
        "version": "V4",
        "model": model,
        "scaler": scaler,
        "config": config,
        "feature_names": configured_feature_names,
    }


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

@st.cache_resource
def load_face_landmarker():

    if not TASK_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model: "
            f"{TASK_MODEL_PATH.name}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(
            TASK_MODEL_PATH
        )
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return (
        vision.FaceLandmarker
        .create_from_options(options)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def point_xy(
    landmarks,
    idx,
):
    return np.array(
        [
            landmarks[idx].x,
            landmarks[idx].y,
        ],
        dtype=np.float64,
    )


def dist(
    a,
    b,
):
    return float(
        np.linalg.norm(a - b)
    )


def calculate_ear(
    landmarks,
    eye_indices,
):

    p1, p2, p3, p4, p5, p6 = [
        point_xy(
            landmarks,
            i,
        )
        for i in eye_indices
    ]

    horizontal = dist(
        p1,
        p4,
    )

    if horizontal < 1e-8:
        return np.nan

    return (
        dist(p2, p6)
        + dist(p3, p5)
    ) / (
        2.0 * horizontal
    )


def calculate_mar(
    landmarks,
):

    left = point_xy(
        landmarks,
        MOUTH_LEFT,
    )

    right = point_xy(
        landmarks,
        MOUTH_RIGHT,
    )

    upper = point_xy(
        landmarks,
        UPPER_LIP,
    )

    lower = point_xy(
        landmarks,
        LOWER_LIP,
    )

    lower_l = point_xy(
        landmarks,
        MOUTH_LEFT_INNER,
    )

    lower_r = point_xy(
        landmarks,
        MOUTH_RIGHT_INNER,
    )

    mouth_width = dist(
        left,
        right,
    )

    if mouth_width < 1e-8:
        return np.nan

    vertical_1 = dist(
        upper,
        lower,
    )

    vertical_2 = dist(
        lower_l,
        lower_r,
    )

    return (
        vertical_1
        + vertical_2
    ) / (
        2.0 * mouth_width
    )


def calculate_head_pose(
    landmarks,
    image_width,
    image_height,
):

    landmark_indices = [
        1,
        152,
        33,
        263,
        61,
        291,
    ]

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )

    image_points = np.array(
        [
            (
                landmarks[idx].x
                * image_width,
                landmarks[idx].y
                * image_height,
            )
            for idx in landmark_indices
        ],
        dtype=np.float64,
    )

    focal_length = float(
        image_width
    )

    camera_matrix = np.array(
        [
            [
                focal_length,
                0,
                image_width / 2,
            ],
            [
                0,
                focal_length,
                image_height / 2,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float64,
    )

    dist_coeffs = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    try:

        success, rvec, _ = (
            cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

        if not success:
            return (
                0.0,
                0.0,
                0.0,
            )

        rotation_matrix, _ = (
            cv2.Rodrigues(
                rvec
            )
        )

        angles, _, _, _, _, _ = (
            cv2.RQDecomp3x3(
                rotation_matrix
            )
        )

        return tuple(
            float(v)
            for v in angles[:3]
        )

    except Exception:
        return (
            0.0,
            0.0,
            0.0,
        )


def iris_center(
    landmarks,
    indices,
):

    pts = np.array(
        [
            point_xy(
                landmarks,
                i,
            )
            for i in indices
        ]
    )

    return pts.mean(
        axis=0
    )


def normalized_gaze(
    landmarks,
    iris_indices,
    eye_left_idx,
    eye_right_idx,
):

    iris = iris_center(
        landmarks,
        iris_indices,
    )

    left = point_xy(
        landmarks,
        eye_left_idx,
    )

    right = point_xy(
        landmarks,
        eye_right_idx,
    )

    horizontal = (
        right - left
    )

    width = np.linalg.norm(
        horizontal
    )

    if width < 1e-8:
        return (
            0.5,
            0.5,
        )

    gaze_x = (
        np.dot(
            iris - left,
            horizontal,
        )
        / (width ** 2)
    )

    center = (
        left + right
    ) / 2.0

    gaze_y = (
        (
            iris[1]
            - center[1]
        )
        / max(
            width,
            1e-8,
        )
        + 0.5
    )

    return (
        float(gaze_x),
        float(gaze_y),
    )


def extract_refined_features(
    landmarks,
    image_width,
    image_height,
):

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE,
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE,
    )

    average_ear = (
        left_ear
        + right_ear
    ) / 2.0

    ear_asymmetry = abs(
        left_ear
        - right_ear
    )

    mouth_left = point_xy(
        landmarks,
        MOUTH_LEFT,
    )

    mouth_right = point_xy(
        landmarks,
        MOUTH_RIGHT,
    )

    mouth_open = dist(
        point_xy(
            landmarks,
            UPPER_LIP,
        ),
        point_xy(
            landmarks,
            LOWER_LIP,
        ),
    )

    interocular = dist(
        point_xy(
            landmarks,
            LEFT_EYE_OUTER,
        ),
        point_xy(
            landmarks,
            RIGHT_EYE_OUTER,
        ),
    )

    face_height = dist(
        point_xy(
            landmarks,
            FOREHEAD,
        ),
        point_xy(
            landmarks,
            CHIN,
        ),
    )

    face_width = interocular

    safe_interocular = max(
        interocular,
        1e-8,
    )

    safe_face_height = max(
        face_height,
        1e-8,
    )

    mar = calculate_mar(
        landmarks
    )

    normalized_mouth_opening = (
        mouth_open
        / safe_interocular
    )

    normalized_mouth_width = (
        dist(
            mouth_left,
            mouth_right,
        )
        / safe_interocular
    )

    normalized_eye_distance = (
        interocular
        / safe_face_height
    )

    face_aspect_ratio = (
        face_width
        / safe_face_height
    )

    pitch, yaw, roll = (
        calculate_head_pose(
            landmarks,
            image_width,
            image_height,
        )
    )

    left_gaze_x, left_gaze_y = (
        normalized_gaze(
            landmarks,
            LEFT_IRIS,
            LEFT_EYE_OUTER,
            LEFT_EYE_INNER,
        )
    )

    right_gaze_x, right_gaze_y = (
        normalized_gaze(
            landmarks,
            RIGHT_IRIS,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
        )
    )

    gaze_x = (
        left_gaze_x
        + right_gaze_x
    ) / 2.0

    gaze_y = (
        left_gaze_y
        + right_gaze_y
    ) / 2.0

    gaze_asymmetry = abs(
        left_gaze_x
        - right_gaze_x
    )

    nose = point_xy(
        landmarks,
        NOSE_TIP,
    )

    forehead = point_xy(
        landmarks,
        FOREHEAD,
    )

    chin = point_xy(
        landmarks,
        CHIN,
    )

    nose_vertical = (
        nose[1]
        - forehead[1]
    ) / safe_face_height

    chin_vertical = (
        chin[1]
        - forehead[1]
    ) / safe_face_height

    return [
        left_ear,
        right_ear,
        average_ear,
        ear_asymmetry,
        mar,
        normalized_mouth_opening,
        normalized_mouth_width,
        normalized_eye_distance,
        face_aspect_ratio,
        pitch,
        yaw,
        roll,
        gaze_x,
        gaze_y,
        gaze_asymmetry,
        nose_vertical,
        chin_vertical,
    ]


# ============================================================
# V4 FEATURE PREPARATION
# ============================================================

def prepare_v4_features(
    features,
    pipeline,
):

    config = pipeline["config"]

    expected_names = pipeline[
        "feature_names"
    ]

    feature_df = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES_17,
    )

    # --------------------------------------------------------
    # Make sure every requested V4 feature exists
    # --------------------------------------------------------

    missing = [
        name
        for name in expected_names
        if name not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            "V4 feature mismatch.\n\n"
            f"Missing feature(s): {missing}\n\n"
            f"Available extracted features:\n"
            f"{FEATURE_NAMES_17}\n\n"
            f"V4 config expects:\n"
            f"{expected_names}"
        )

    # --------------------------------------------------------
    # Force exact V4 feature order
    # --------------------------------------------------------

    raw_features = feature_df[
        expected_names
    ].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Scale using V4 scaler
    # --------------------------------------------------------

    scaled_features = (
        pipeline["scaler"].transform(
            raw_features
        )
    )

    return scaled_features


# ============================================================
# V4 PREDICTION
# ============================================================

def predict_features(
    features,
    pipeline,
):

    if pipeline["version"] != "V4":
        raise ValueError(
            "Invalid pipeline version. "
            "This application supports V4 only."
        )

    config = pipeline[
        "config"
    ]

    class_names = config.get(
        "class_names",
        CLASS_NAMES_DEFAULT,
    )

    class_names = {
        int(k): v
        for k, v in class_names.items()
    }

    bias = np.asarray(
        config.get(
            "class_bias",
            [
                0.0,
                0.0,
                0.0,
            ],
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Prepare V4 input
    # --------------------------------------------------------

    scaled = prepare_v4_features(
        features,
        pipeline,
    )

    # --------------------------------------------------------
    # V4 ANN prediction
    # --------------------------------------------------------

    probabilities = (
        pipeline["model"]
        .predict(
            scaled,
            verbose=0,
        )[0]
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validation-only class bias rule
    #
    # Same logic that was present in the original deployment.
    # --------------------------------------------------------

    if len(bias) != len(
        probabilities
    ):
        bias = np.zeros(
            len(probabilities),
            dtype=np.float32,
        )

    adjusted = (
        np.log(
            np.clip(
                probabilities,
                1e-8,
                1.0,
            )
        )
        + bias
    )

    predicted_class = int(
        np.argmax(adjusted)
    )

    return (
        predicted_class,
        probabilities,
        class_names,
    )


# ============================================================
# SHARED LIVE MONITOR
# ============================================================

class EngagementMonitor:

    def __init__(self):

        self.lock = threading.Lock()

        self.started_at = None
        self.last_prediction_time = None

        self.total_face_seconds = 0.0

        self.class_seconds = (
            defaultdict(float)
        )

        self.current_class = None

        self.current_probabilities = (
            np.zeros(
                3,
                dtype=np.float32,
            )
        )

        self.frames_seen = 0
        self.predictions_made = 0

        self.last_features = None

        self.recording_path = None

        self.last_frame_time = None

    def start(self):

        with self.lock:

            if self.started_at is None:

                now = time.monotonic()

                self.started_at = now

                self.last_prediction_time = (
                    now
                )

                self.last_frame_time = (
                    now
                )

    def update_prediction(
        self,
        predicted_class,
        probabilities,
        features,
    ):

        now = time.monotonic()

        with self.lock:

            if self.started_at is None:
                self.started_at = now

            if self.last_prediction_time is None:
                self.last_prediction_time = now

            interval = max(
                0.0,
                now
                - self.last_prediction_time,
            )

            interval = min(
                interval,
                1.0,
            )

            if self.current_class is not None:

                self.class_seconds[
                    self.current_class
                ] += interval

                self.total_face_seconds += (
                    interval
                )

            self.current_class = (
                predicted_class
            )

            self.current_probabilities = (
                np.asarray(
                    probabilities,
                    dtype=np.float32,
                )
            )

            self.last_features = list(
                features
            )

            self.last_prediction_time = (
                now
            )

            self.predictions_made += 1

    def frame_seen(self):

        with self.lock:
            self.frames_seen += 1

    def snapshot(self):

        with self.lock:

            elapsed = (
                0.0
                if self.started_at is None
                else max(
                    0.0,
                    time.monotonic()
                    - self.started_at,
                )
            )

            return {
                "elapsed": elapsed,
                "total_face_seconds":
                    self.total_face_seconds,
                "class_seconds":
                    dict(
                        self.class_seconds
                    ),
                "current_class":
                    self.current_class,
                "current_probabilities":
                    self.current_probabilities.copy(),
                "frames_seen":
                    self.frames_seen,
                "predictions_made":
                    self.predictions_made,
                "last_features":
                    (
                        None
                        if self.last_features is None
                        else list(
                            self.last_features
                        )
                    ),
                "recording_path":
                    self.recording_path,
            }

    def reset(self):

        with self.lock:

            self.started_at = None
            self.last_prediction_time = None

            self.total_face_seconds = 0.0

            self.class_seconds = (
                defaultdict(float)
            )

            self.current_class = None

            self.current_probabilities = (
                np.zeros(
                    3,
                    dtype=np.float32,
                )
            )

            self.frames_seen = 0
            self.predictions_made = 0

            self.last_features = None

            self.recording_path = None

            self.last_frame_time = None


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class EngagementVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.monitor = (
            EngagementMonitor()
        )

        self.pipeline = None
        self.detector = None

        self.frame_counter = 0

        self.prediction_interval = 5

        self.last_class = None

        self.last_probabilities = (
            np.zeros(
                3,
                dtype=np.float32,
            )
        )

        self.last_features = None

        self.writer = None

        self.writer_lock = (
            threading.Lock()
        )

    def _lazy_load(self):

        if self.pipeline is None:
            self.pipeline = (
                load_pipeline()
            )

        if self.detector is None:
            self.detector = (
                load_face_landmarker()
            )

    def _open_writer(
        self,
        frame,
    ):

        if self.writer is not None:
            return

        height, width = (
            frame.shape[:2]
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        path = (
            RECORDING_DIR
            / (
                f"engagement_session_"
                f"{timestamp}.mp4"
            )
        )

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            20.0,
            (width, height),
        )

        if writer.isOpened():

            self.writer = writer

            self.monitor.recording_path = (
                str(path)
            )

    def _write_frame(
        self,
        frame,
    ):

        if self.writer is None:
            self._open_writer(frame)

        if self.writer is not None:

            with self.writer_lock:
                self.writer.write(
                    frame
                )

    def _annotate(
        self,
        frame,
        class_name,
        probabilities,
    ):

        output = frame.copy()

        label = (
            f"Engagement: {class_name}"
        )

        cv2.rectangle(
            output,
            (15, 15),
            (430, 130),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        names = [
            "Low",
            "Moderate",
            "High",
        ]

        y = 78

        for idx, name in enumerate(
            names
        ):

            if idx >= len(
                probabilities
            ):
                break

            text = (
                f"{name}: "
                f"{probabilities[idx] * 100:.1f}%"
            )

            cv2.putText(
                output,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y += 18

        return output

    def recv(
        self,
        frame,
    ):

        self.monitor.start()

        self.monitor.frame_seen()

        image = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_counter += 1

        # Always record the frame.
        self._write_frame(
            image
        )

        try:

            self._lazy_load()

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            result = (
                self.detector.detect(
                    mp_image
                )
            )

            if result.face_landmarks:

                landmarks = (
                    result.face_landmarks[0]
                )

                if (
                    self.frame_counter
                    % self.prediction_interval
                    == 0
                ):

                    features = (
                        extract_refined_features(
                            landmarks,
                            image.shape[1],
                            image.shape[0],
                        )
                    )

                    features = np.asarray(
                        features,
                        dtype=np.float32,
                    )

                    if (
                        len(features)
                        == len(
                            FEATURE_NAMES_17
                        )
                        and np.isfinite(
                            features
                        ).all()
                    ):

                        (
                            predicted_class,
                            probabilities,
                            class_names,
                        ) = predict_features(
                            features,
                            self.pipeline,
                        )

                        self.last_class = (
                            predicted_class
                        )

                        self.last_probabilities = (
                            probabilities
                        )

                        self.last_features = (
                            features
                        )

                        self.monitor.update_prediction(
                            predicted_class,
                            probabilities,
                            features,
                        )

            if self.last_class is not None:

                class_names = {
                    0: "Low Engagement",
                    1: "Moderate Engagement",
                    2: "High Engagement",
                }

                output = self._annotate(
                    image,
                    class_names[
                        self.last_class
                    ],
                    self.last_probabilities,
                )

            else:

                output = image.copy()

                cv2.putText(
                    output,
                    "Detecting face...",
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        except Exception as exc:

            output = image.copy()

            error_text = (
                f"V4 Pipeline error: "
                f"{str(exc)[:80]}"
            )

            cv2.putText(
                output,
                error_text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24",
        )

    def stop_recording(self):

        with self.writer_lock:

            if self.writer is not None:

                self.writer.release()

                self.writer = None


# ============================================================
# LOAD V4 PIPELINE STATUS
# ============================================================

try:

    pipeline = load_pipeline()

    st.success(
        "✅ V4 ANN model pipeline loaded successfully."
    )

except Exception as exc:

    st.error(
        "❌ V4 model pipeline could not be loaded."
    )

    st.code(
        str(exc)
    )

    st.info(
        "Make sure these V4 files are in the same "
        "folder as app.py:\n\n"
        "• workplace_engagement_ann_v4.keras\n"
        "• workplace_engagement_scaler_v4.pkl\n"
        "• workplace_engagement_v4_config.pkl\n"
        "• face_landmarker.task"
    )

    st.stop()


# ============================================================
# MEDIAPIPE MODEL CHECK
# ============================================================

if not TASK_MODEL_PATH.exists():

    st.error(
        "❌ face_landmarker.task is missing."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Session Controls"
    )

    prediction_interval = st.slider(
        "Prediction interval (frames)",
        min_value=1,
        max_value=15,
        value=5,
        help=(
            "Lower = more predictions but "
            "more CPU usage. Higher = smoother "
            "webcam processing."
        ),
    )

    st.markdown("---")

    st.write(
        "### Model"
    )

    st.write(
        "Pipeline: **V4**"
    )

    st.write(
        "ANN Model: "
        "`workplace_engagement_ann_v4.keras`"
    )

    st.write(
        "### Classes"
    )

    st.write(
        "- 🔴 Low Engagement"
    )

    st.write(
        "- 🟡 Moderate Engagement"
    )

    st.write(
        "- 🟢 High Engagement"
    )


# ============================================================
# WEBCAM
# ============================================================

st.subheader(
    "📷 Live Webcam"
)

st.write(
    "Click **START** below to begin live "
    "engagement monitoring. The V4 ANN model "
    "will predict engagement continuously "
    "and record the session."
)


rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


ctx = webrtc_streamer(
    key=(
        f"engagement-monitor-"
        f"{st.session_state.session_reset_id}"
    ),
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_processor_factory=(
        EngagementVideoProcessor
    ),
    async_processing=True,
)


# Apply selected prediction interval.
if ctx.video_processor:

    ctx.video_processor.prediction_interval = (
        prediction_interval
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Live Engagement Dashboard"
)


if ctx.video_processor:

    snapshot = (
        ctx.video_processor
        .monitor
        .snapshot()
    )

else:

    snapshot = {
        "elapsed": 0.0,
        "total_face_seconds": 0.0,
        "class_seconds": {},
        "current_class": None,
        "current_probabilities": np.zeros(
            3
        ),
        "frames_seen": 0,
        "predictions_made": 0,
        "last_features": None,
        "recording_path": None,
    }


elapsed = snapshot[
    "elapsed"
]

minutes = int(
    elapsed // 60
)

seconds = int(
    elapsed % 60
)


col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Session Duration",
        f"{minutes:02d}:{seconds:02d}",
    )


with col2:

    st.metric(
        "Frames Processed",
        snapshot[
            "frames_seen"
        ],
    )


with col3:

    st.metric(
        "Predictions",
        snapshot[
            "predictions_made"
        ],
    )


with col4:

    current_class = snapshot[
        "current_class"
    ]

    current_label = (
        "Waiting..."
        if current_class is None
        else CLASS_NAMES_DEFAULT[
            int(current_class)
        ]
    )

    st.metric(
        "Current Status",
        current_label,
    )


# ============================================================
# LIVE PROBABILITIES
# ============================================================

probabilities = snapshot[
    "current_probabilities"
]


p1, p2, p3 = (
    st.columns(3)
)


with p1:

    st.metric(
        "🔴 Low",
        f"{probabilities[0] * 100:.2f}%",
    )


with p2:

    st.metric(
        "🟡 Moderate",
        f"{probabilities[1] * 100:.2f}%",
    )


with p3:

    st.metric(
        "🟢 High",
        f"{probabilities[2] * 100:.2f}%",
    )


# ============================================================
# SESSION REPORT
# ============================================================

st.markdown("---")

st.subheader(
    "📋 Engagement Session Report"
)


class_seconds = snapshot[
    "class_seconds"
]


low_seconds = float(
    class_seconds.get(
        0,
        0.0,
    )
)

moderate_seconds = float(
    class_seconds.get(
        1,
        0.0,
    )
)

high_seconds = float(
    class_seconds.get(
        2,
        0.0,
    )
)


tracked_seconds = (
    low_seconds
    + moderate_seconds
    + high_seconds
)


if tracked_seconds > 0:

    low_pct = (
        low_seconds
        / tracked_seconds
        * 100
    )

    moderate_pct = (
        moderate_seconds
        / tracked_seconds
        * 100
    )

    high_pct = (
        high_seconds
        / tracked_seconds
        * 100
    )

else:

    low_pct = 0.0
    moderate_pct = 0.0
    high_pct = 0.0


def format_duration(
    total_seconds,
):

    total_seconds = max(
        0,
        int(
            round(
                total_seconds
            )
        ),
    )

    h = (
        total_seconds
        // 3600
    )

    m = (
        total_seconds
        % 3600
    ) // 60

    s = (
        total_seconds
        % 60
    )

    if h > 0:

        return (
            f"{h}h "
            f"{m}m "
            f"{s}s"
        )

    return (
        f"{m}m "
        f"{s}s"
    )


r1, r2, r3 = (
    st.columns(3)
)


with r1:

    st.metric(
        "🔴 Low Engagement",
        format_duration(
            low_seconds
        ),
    )

    st.write(
        f"{low_pct:.1f}% "
        f"of tracked time"
    )


with r2:

    st.metric(
        "🟡 Moderate Engagement",
        format_duration(
            moderate_seconds
        ),
    )

    st.write(
        f"{moderate_pct:.1f}% "
        f"of tracked time"
    )


with r3:

    st.metric(
        "🟢 High Engagement",
        format_duration(
            high_seconds
        ),
    )

    st.write(
        f"{high_pct:.1f}% "
        f"of tracked time"
    )


# ============================================================
# OVERALL ENGAGEMENT
# ============================================================

if tracked_seconds > 0:

    percentages = np.array(
        [
            low_pct,
            moderate_pct,
            high_pct,
        ]
    )

    overall_class = int(
        np.argmax(
            percentages
        )
    )

    overall_label = (
        CLASS_NAMES_DEFAULT[
            overall_class
        ]
    )

    st.success(
        f"Overall Engagement: "
        f"**{overall_label}**"
    )

    st.write(
        f"You spent approximately "
        f"**{format_duration(high_seconds)}** "
        f"in High Engagement, "
        f"**{format_duration(moderate_seconds)}** "
        f"in Moderate Engagement, and "
        f"**{format_duration(low_seconds)}** "
        f"in Low Engagement."
    )


# ============================================================
# ENGAGEMENT DISTRIBUTION CHART
# ============================================================

if tracked_seconds > 0:

    chart_df = pd.DataFrame(
        {
            "Engagement": [
                "Low",
                "Moderate",
                "High",
            ],
            "Percentage": [
                low_pct,
                moderate_pct,
                high_pct,
            ],
        }
    ).set_index(
        "Engagement"
    )

    st.bar_chart(
        chart_df
    )


# ============================================================
# LAST EXTRACTED FEATURES
# ============================================================

last_features = snapshot[
    "last_features"
]


with st.expander(
    "🔎 View latest 17 extracted features"
):

    if last_features is not None:

        feature_df = pd.DataFrame(
            {
                "Feature":
                    FEATURE_NAMES_17,
                "Value":
                    last_features,
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Features will appear after "
            "a face is detected."
        )


# ============================================================
# RECORDING INFO
# ============================================================

st.markdown("---")

st.subheader(
    "🎥 Session Recording"
)


recording_path = snapshot[
    "recording_path"
]


if recording_path:

    st.info(
        "Recording is being saved to:\n"
        f"`{recording_path}`"
    )

    if Path(
        recording_path
    ).exists():

        with open(
            recording_path,
            "rb",
        ) as video_file:

            st.download_button(
                "⬇️ Download Session Video",
                data=video_file,
                file_name=(
                    Path(
                        recording_path
                    ).name
                ),
                mime="video/mp4",
            )

else:

    st.caption(
        "A recording file will be created "
        "when the webcam session starts."
    )


# ============================================================
# RESET SESSION
# ============================================================

st.markdown("---")


if st.button(
    "🔄 Reset Session",
    use_container_width=True,
):

    if ctx.video_processor:

        ctx.video_processor.stop_recording()

    st.session_state.session_reset_id += 1

    st.session_state.last_recording_path = (
        None
    )

    st.rerun()


# ============================================================
# AUTO REFRESH WHILE WEBCAM IS ACTIVE
# ============================================================

if ctx.state.playing:

    time.sleep(1.0)

    st.rerun()