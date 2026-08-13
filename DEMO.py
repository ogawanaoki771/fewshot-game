import random
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.interpolate import splprep, splev
from shapely.geometry import LineString

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 基本設定
# =========================================================

IMG_SIZE = 50
SEED = 2026

# 再現性
random.seed(SEED)
np.random.seed(SEED)

rng = np.random.default_rng(SEED)

MAX_CELLS = 105
MIN_CELLS = 10

INITIAL_CELLS = 15

NUM_EPOCHS = 1

TRAIN_STEPS = 28
TRAIN_FEATURE_STEPS = 28
TEST_TTA_STEPS = 28

RESEARCH_TRAJECTORY_STEPS = 28


# =========================================================
# 2. 多様な図形生成器
#
# Train:
#   shape_type = 0..4
#
# Test:
#   shape_type = 5..9
#
# Closed / Open も family を分離
# =========================================================

def generate_diverse_shapes(
    num_samples,
    img_size,
    seed,
    family_type="train"
):

    np.random.seed(seed)
    random.seed(seed)

    X_coords = []
    y = []

    for i in range(num_samples):

        # -------------------------------------------------
        # 0: Closed
        # 1: Open
        # -------------------------------------------------

        label = i % 2
        is_closed = (label == 0)

        valid_curve_found = False
        attempts = 0

        while not valid_curve_found and attempts < 100:

            attempts += 1

            img = np.zeros(
                (img_size, img_size),
                dtype=np.uint8
            )

            cx = random.randint(
                15,
                img_size - 15
            )

            cy = random.randint(
                15,
                img_size - 15
            )

            pts_list = None

            # =================================================
            # Family assignment
            # =================================================

            if family_type == "train":

                shape_type = (
                    i + random.randint(0, 2)
                ) % 5

            else:

                shape_type = (
                    i % 5
                ) + 5

            # =================================================
            # CLOSED CURVES
            # =================================================

            if is_closed:

                # -------------------------------------------------
                # Shape 0: Ellipse
                # -------------------------------------------------

                if shape_type == 0:

                    a = random.uniform(8, 16)
                    b = random.uniform(6, 14)

                    angle = random.uniform(
                        0,
                        2 * np.pi
                    )

                    n_pts = random.randint(
                        40,
                        80
                    )

                    thetas = np.linspace(
                        0,
                        2 * np.pi,
                        n_pts,
                        endpoint=False
                    )

                    x_raw = (
                        a * np.cos(thetas)
                        + np.random.normal(
                            0,
                            0.6,
                            n_pts
                        )
                    )

                    y_raw = (
                        b * np.sin(thetas)
                        + np.random.normal(
                            0,
                            0.6,
                            n_pts
                        )
                    )

                    x_rot = (
                        x_raw * np.cos(angle)
                        - y_raw * np.sin(angle)
                        + cx
                    )

                    y_rot = (
                        x_raw * np.sin(angle)
                        + y_raw * np.cos(angle)
                        + cy
                    )

                    x_pts = np.append(
                        x_rot,
                        x_rot[0]
                    )

                    y_pts = np.append(
                        y_rot,
                        y_rot[0]
                    )

                    tck, _ = splprep(
                        [x_pts, y_pts],
                        s=random.uniform(
                            0.1,
                            1.0
                        ),
                        per=True,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            120
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 1: Polygon
                # -------------------------------------------------

                elif shape_type == 1:

                    n_v = random.randint(
                        4,
                        9
                    )

                    thetas = np.sort(
                        np.random.uniform(
                            0,
                            2 * np.pi,
                            n_v
                        )
                    )

                    radii = np.random.uniform(
                        6,
                        16,
                        n_v
                    )

                    x_v = (
                        cx
                        + radii * np.cos(thetas)
                        + np.random.normal(
                            0,
                            0.4,
                            n_v
                        )
                    )

                    y_v = (
                        cy
                        + radii * np.sin(thetas)
                        + np.random.normal(
                            0,
                            0.4,
                            n_v
                        )
                    )

                    x_pts = np.append(
                        x_v,
                        x_v[0]
                    )

                    y_pts = np.append(
                        y_v,
                        y_v[0]
                    )

                    tck, _ = splprep(
                        [x_pts, y_pts],
                        s=0.5,
                        per=True,
                        k=1
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            100
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 2: Star
                # -------------------------------------------------

                elif shape_type == 2:

                    num_teeth = random.randint(
                        3,
                        8
                    )

                    n_pts = num_teeth * 4

                    thetas = np.linspace(
                        0,
                        2 * np.pi,
                        n_pts,
                        endpoint=False
                    )

                    r_outer = random.uniform(
                        10,
                        16
                    )

                    r_inner = random.uniform(
                        4,
                        9
                    )

                    radii = np.where(
                        np.arange(n_pts) % 2 == 0,
                        r_outer,
                        r_inner
                    )

                    radii += np.random.normal(
                        0,
                        0.3,
                        n_pts
                    )

                    x_v = (
                        cx
                        + radii * np.cos(thetas)
                    )

                    y_v = (
                        cy
                        + radii * np.sin(thetas)
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 3: Smooth irregular contour
                # -------------------------------------------------

                elif shape_type == 3:

                    n_ctrl = random.randint(
                        5,
                        10
                    )

                    thetas = np.linspace(
                        0,
                        2 * np.pi,
                        n_ctrl,
                        endpoint=False
                    )

                    radii = np.random.uniform(
                        7,
                        15,
                        n_ctrl
                    )

                    x_ctrl = (
                        cx
                        + radii * np.cos(thetas)
                    )

                    y_ctrl = (
                        cy
                        + radii * np.sin(thetas)
                    )

                    x_pts = np.append(
                        x_ctrl,
                        x_ctrl[0]
                    )

                    y_pts = np.append(
                        y_ctrl,
                        y_ctrl[0]
                    )

                    tck, _ = splprep(
                        [x_pts, y_pts],
                        s=1.0,
                        per=True,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            140
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 4: Wavy contour
                # -------------------------------------------------

                elif shape_type == 4:

                    n_waves = random.randint(
                        3,
                        6
                    )

                    thetas = np.linspace(
                        0,
                        2 * np.pi,
                        80,
                        endpoint=False
                    )

                    r0 = random.uniform(
                        8,
                        12
                    )

                    amp = random.uniform(
                        2,
                        5
                    )

                    radii = (
                        r0
                        + amp
                        * np.sin(
                            n_waves * thetas
                        )
                        + np.random.normal(
                            0,
                            0.2,
                            80
                        )
                    )

                    x_ctrl = (
                        cx
                        + radii * np.cos(thetas)
                    )

                    y_ctrl = (
                        cy
                        + radii * np.sin(thetas)
                    )

                    x_pts = np.append(
                        x_ctrl,
                        x_ctrl[0]
                    )

                    y_pts = np.append(
                        y_ctrl,
                        y_ctrl[0]
                    )

                    tck, _ = splprep(
                        [x_pts, y_pts],
                        s=0.2,
                        per=True,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            140
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 5: Rose-like irregular contour
                # -------------------------------------------------

                elif shape_type == 5:

                    thetas = np.linspace(
                        0,
                        2 * np.pi,
                        90,
                        endpoint=False
                    )

                    freq = random.randint(
                        4,
                        7
                    )

                    radii = (
                        11
                        + 4
                        * np.sin(freq * thetas)
                        * np.cos(2 * thetas)
                    )

                    x_v = (
                        cx
                        + radii * np.cos(thetas)
                        + np.random.normal(
                            0,
                            0.5,
                            90
                        )
                    )

                    y_v = (
                        cy
                        + radii * np.sin(thetas)
                        + np.random.normal(
                            0,
                            0.5,
                            90
                        )
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 6: Cardioid-like
                # -------------------------------------------------

                elif shape_type == 6:

                    thetas = np.linspace(
                        0,
                        2 * np.pi,
                        80,
                        endpoint=False
                    )

                    a_card = random.uniform(
                        6,
                        10
                    )

                    radii = (
                        a_card
                        * (
                            1
                            - np.cos(thetas)
                        )
                        + random.uniform(
                            0,
                            3
                        )
                    )

                    x_v = (
                        cx
                        + radii * np.cos(thetas)
                    )

                    y_v = (
                        cy
                        + radii * np.sin(thetas)
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 7: Butterfly-like
                # -------------------------------------------------

                elif shape_type == 7:

                    t = np.linspace(
                        0,
                        2 * np.pi,
                        80
                    )

                    scale = random.uniform(
                        10,
                        15
                    )

                    denom = (
                        1.0
                        + np.sin(t) ** 2
                    )

                    x_raw = (
                        scale
                        * np.cos(t)
                        / denom
                    )

                    y_raw = (
                        scale
                        * np.sin(t)
                        * np.cos(t)
                        / denom
                    )

                    x_v = (
                        cx
                        + x_raw
                        + np.random.normal(
                            0,
                            0.3,
                            80
                        )
                    )

                    y_v = (
                        cy
                        + y_raw
                        + np.random.normal(
                            0,
                            0.3,
                            80
                        )
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 8: Superellipse
                # -------------------------------------------------

                elif shape_type == 8:

                    t = np.linspace(
                        0,
                        2 * np.pi,
                        80,
                        endpoint=False
                    )

                    denominator = (
                        np.abs(np.cos(t)) ** 3
                        +
                        np.abs(np.sin(t)) ** 3
                    )

                    radii = (
                        12
                        /
                        np.maximum(
                            denominator ** (1 / 3),
                            1e-6
                        )
                    )

                    radii = np.clip(
                        radii,
                        5,
                        16
                    )

                    radii += np.random.normal(
                        0,
                        0.3,
                        80
                    )

                    x_v = (
                        cx
                        + radii * np.cos(t)
                    )

                    y_v = (
                        cy
                        + radii * np.sin(t)
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 9: Spiral-like closed contour
                # -------------------------------------------------

                else:

                    t = np.linspace(
                        0.5,
                        2 * np.pi,
                        80
                    )

                    radii = (
                        4
                        + 2 * t
                    )

                    x_v = (
                        cx
                        + radii
                        * np.cos(2 * t)
                    )

                    y_v = (
                        cy
                        + radii
                        * np.sin(2 * t)
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

            # =================================================
            # OPEN CURVES
            # =================================================

            else:

                # -------------------------------------------------
                # Shape 0: Spiral
                # -------------------------------------------------

                if shape_type == 0:

                    turns = random.uniform(
                        1.2,
                        2.5
                    )

                    theta_max = (
                        turns * 2 * np.pi
                    )

                    n_pts = 60

                    thetas = np.linspace(
                        0.2,
                        theta_max,
                        n_pts
                    )

                    r0 = random.uniform(
                        2,
                        5
                    )

                    b = random.uniform(
                        1.0,
                        2.0
                    )

                    radii = (
                        r0
                        + b * thetas
                        + np.random.normal(
                            0,
                            0.3,
                            n_pts
                        )
                    )

                    x_ctrl = (
                        cx
                        + radii * np.cos(thetas)
                    )

                    y_ctrl = (
                        cy
                        + radii * np.sin(thetas)
                    )

                    tck, _ = splprep(
                        [x_ctrl, y_ctrl],
                        s=0.5,
                        per=False,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            100
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 1: Arc
                # -------------------------------------------------

                elif shape_type == 1:

                    span = random.uniform(
                        np.pi,
                        1.8 * np.pi
                    )

                    start_a = random.uniform(
                        0,
                        2 * np.pi
                    )

                    n_pts = 50

                    thetas = np.linspace(
                        start_a,
                        start_a + span,
                        n_pts
                    )

                    r = random.uniform(
                        8,
                        15
                    )

                    x_v = (
                        cx
                        + r * np.cos(thetas)
                        + np.random.normal(
                            0,
                            0.3,
                            n_pts
                        )
                    )

                    y_v = (
                        cy
                        + r * np.sin(thetas)
                        + np.random.normal(
                            0,
                            0.3,
                            n_pts
                        )
                    )

                    tck, _ = splprep(
                        [x_v, y_v],
                        s=0.2,
                        per=False,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            90
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 2: Sine
                # -------------------------------------------------

                elif shape_type == 2:

                    n_pts = 50

                    t = np.linspace(
                        -2.0,
                        2.0,
                        n_pts
                    )

                    x_raw = (
                        t
                        * random.uniform(
                            5,
                            9
                        )
                    )

                    y_raw = (
                        np.sin(
                            t
                            * random.uniform(
                                1.0,
                                2.5
                            )
                        )
                        * random.uniform(
                            8,
                            14
                        )
                    )

                    angle = random.uniform(
                        0,
                        2 * np.pi
                    )

                    x_v = (
                        x_raw * np.cos(angle)
                        - y_raw * np.sin(angle)
                        + cx
                        + np.random.normal(
                            0,
                            0.3,
                            n_pts
                        )
                    )

                    y_v = (
                        x_raw * np.sin(angle)
                        + y_raw * np.cos(angle)
                        + cy
                        + np.random.normal(
                            0,
                            0.3,
                            n_pts
                        )
                    )

                    tck, _ = splprep(
                        [x_v, y_v],
                        s=0.5,
                        per=False,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            90
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 3: Smooth open curve
                # -------------------------------------------------

                elif shape_type == 3:

                    n_pts = random.randint(
                        4,
                        8
                    )

                    x_ctrl = (
                        np.linspace(
                            cx - 14,
                            cx + 14,
                            n_pts
                        )
                        + np.random.normal(
                            0,
                            2.0,
                            n_pts
                        )
                    )

                    y_ctrl = (
                        cy
                        + np.random.uniform(
                            -12,
                            12,
                            n_pts
                        )
                    )

                    tck, _ = splprep(
                        [x_ctrl, y_ctrl],
                        s=1.0,
                        per=False,
                        k=2
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            90
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 4: Random spline
                # -------------------------------------------------

                elif shape_type == 4:

                    n_pts = random.randint(
                        6,
                        12
                    )

                    x_v = np.random.uniform(
                        cx - 15,
                        cx + 15,
                        n_pts
                    )

                    y_v = np.random.uniform(
                        cy - 15,
                        cy + 15,
                        n_pts
                    )

                    tck, _ = splprep(
                        [x_v, y_v],
                        s=1.5,
                        per=False,
                        k=3
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            100
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 5: Multi-wave open curve
                # -------------------------------------------------

                elif shape_type == 5:

                    n_pts = 50

                    t = np.linspace(
                        -np.pi,
                        np.pi,
                        n_pts
                    )

                    x_raw = t * 8

                    y_raw = (
                        10 * np.cos(t)
                        + 4 * np.sin(3 * t)
                    )

                    angle = random.uniform(
                        0,
                        np.pi
                    )

                    x_v = (
                        x_raw * np.cos(angle)
                        - y_raw * np.sin(angle)
                        + cx
                    )

                    y_v = (
                        x_raw * np.sin(angle)
                        + y_raw * np.cos(angle)
                        + cy
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 6: Parabola
                # -------------------------------------------------

                elif shape_type == 6:

                    t = np.linspace(
                        -3,
                        3,
                        50
                    )

                    x_raw = t * 5

                    y_raw = (
                        (t ** 2)
                        * random.uniform(
                            1.5,
                            3.0
                        )
                        - 10
                    )

                    angle = random.uniform(
                        0,
                        2 * np.pi
                    )

                    x_v = (
                        x_raw * np.cos(angle)
                        - y_raw * np.sin(angle)
                        + cx
                    )

                    y_v = (
                        x_raw * np.sin(angle)
                        + y_raw * np.cos(angle)
                        + cy
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 7: Zigzag
                # -------------------------------------------------

                elif shape_type == 7:

                    n_pts = 7

                    x_ctrl = np.linspace(
                        cx - 12,
                        cx + 12,
                        n_pts
                    )

                    y_ctrl = (
                        cy
                        + np.array(
                            [
                                0,
                                10,
                                -10,
                                12,
                                -12,
                                8,
                                0
                            ]
                        )
                        * random.uniform(
                            0.8,
                            1.2
                        )
                    )

                    tck, _ = splprep(
                        [x_ctrl, y_ctrl],
                        s=0.0,
                        per=False,
                        k=1
                    )

                    x_curve, y_curve = splev(
                        np.linspace(
                            0,
                            1,
                            80
                        ),
                        tck
                    )

                    pts_list = np.vstack(
                        (x_curve, y_curve)
                    ).T

                # -------------------------------------------------
                # Shape 8: Cycloid-like
                # -------------------------------------------------

                elif shape_type == 8:

                    t = np.linspace(
                        0.5,
                        3.5,
                        50
                    )

                    r_c = random.uniform(
                        3,
                        6
                    )

                    x_raw = (
                        r_c
                        * (
                            t
                            - np.sin(t)
                        )
                        * 3
                        - 15
                    )

                    y_raw = (
                        r_c
                        * (
                            1
                            - np.cos(t)
                        )
                        * 3
                        - 5
                    )

                    x_v = (
                        cx
                        + x_raw
                        + np.random.normal(
                            0,
                            0.2,
                            50
                        )
                    )

                    y_v = (
                        cy
                        + y_raw
                        + np.random.normal(
                            0,
                            0.2,
                            50
                        )
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

                # -------------------------------------------------
                # Shape 9: Sinusoidal open curve
                # -------------------------------------------------

                else:

                    t = np.linspace(
                        0,
                        2 * np.pi,
                        60
                    )

                    x_raw = (
                        t * 6
                        - 18
                    )

                    y_raw = (
                        8
                        * np.sin(t)
                        * np.cos(t * 0.5)
                    )

                    x_v = (
                        cx
                        + x_raw
                    )

                    y_v = (
                        cy
                        + y_raw
                    )

                    pts_list = np.vstack(
                        (x_v, y_v)
                    ).T

            # =================================================
            # Curve validation
            # =================================================

            if (
                pts_list is not None
                and len(pts_list) > 2
            ):

                jitter = np.random.normal(
                    0,
                    0.35,
                    pts_list.shape
                )

                pts_list = (
                    pts_list
                    + jitter
                )

                try:

                    line = LineString(
                        pts_list
                    )

                    if line.is_simple:

                        valid_curve_found = True

                        pts_cv = (
                            pts_list.astype(
                                np.int32
                            )
                            .reshape(
                                (-1, 1, 2)
                            )
                        )

                        thickness = random.choice(
                            [1, 1, 2]
                        )

                        cv2.polylines(
                            img,
                            [pts_cv],
                            isClosed=is_closed,
                            color=255,
                            thickness=thickness
                        )

                except Exception:

                    valid_curve_found = False

        # =====================================================
        # Image -> coordinate set
        # =====================================================

        py_indices, px_indices = np.where(
            img > 0
        )

        current_sample_nodes = [
            (
                float(px + 0.5),
                float(py + 0.5)
            )
            for px, py in zip(
                px_indices,
                py_indices
            )
        ]

        if len(current_sample_nodes) > 0:

            X_coords.append(
                current_sample_nodes
            )

            y.append(
                label
            )

    return (
        X_coords,
        np.asarray(
            y,
            dtype=int
        )
    )


# =========================================================
# 3. Line Adaptive Cell
# =========================================================

class LineAdaptiveCell:

    def __init__(self, cell_id):

        self.id = cell_id

        # -------------------------------------------------
        # Anchor
        # -------------------------------------------------

        self.anchor_x = rng.uniform(
            5,
            45
        )

        self.anchor_y = rng.uniform(
            5,
            45
        )

        # -------------------------------------------------
        # Geometry
        # -------------------------------------------------

        self.geom_sensitivity = rng.uniform(
            1.0,
            5.0
        )

        # -------------------------------------------------
        # Dynamic parameters
        # -------------------------------------------------

        self.g_relational_bias = rng.uniform(
            0.0,
            1.0
        )

        self.g_attention_bias = rng.uniform(
            0.0,
            1.0
        )

        self.g_resonance_gain = rng.uniform(
            0.5,
            3.5
        )

        self.g_synaptic_delay = rng.choice(
            [1, 2, 3]
        )

        self.g_learning_rate = rng.uniform(
            0.05,
            0.35
        )

        self.g_mutation_rate = rng.uniform(
            0.05,
            0.25
        )

        # -------------------------------------------------
        # Energy / age / label
        # -------------------------------------------------

        self.energy = rng.uniform(
            0.85,
            1.0
        )

        self.age = 0

        self.target_label = rng.choice(
            [0, 1]
        )

        # -------------------------------------------------
        # Links
        # -------------------------------------------------

        self.links = {}

        # -------------------------------------------------
        # Anchor trajectory
        # -------------------------------------------------

        self.anchor_trajectory = []

        self.reset_dynamic_state()

    # =====================================================
    # Dynamic reset
    # =====================================================

    def reset_dynamic_state(self):

        self.activation_history = [
            0.0
        ] * 5

        self.last_activation = 0.0

        self.anchor_trajectory = [
            (
                float(self.anchor_x),
                float(self.anchor_y)
            )
        ]

    # =====================================================
    # Access
    # =====================================================

    def access(
        self,
        coordinate_points,
        id_to_cell,
        enable_links=True
    ):

        if len(coordinate_points) == 0:
            return 0.0

        pts = np.asarray(
            coordinate_points,
            dtype=float
        )

        # =================================================
        # 1. Distance to input
        # =================================================

        dists = np.hypot(
            pts[:, 0] - self.anchor_x,
            pts[:, 1] - self.anchor_y
        )

        near_mask = (
            dists < 10.0
        )

        near_line_count = int(
            np.sum(near_mask)
        )

        # =================================================
        # 2. Visual response
        # =================================================

        visual_response = np.tanh(
            near_line_count
            /
            (
                self.geom_sensitivity
                * 8.0
                + 1e-9
            )
        )

        # =================================================
        # 3. Attraction to input
        # =================================================

        if near_line_count > 0:

            near_pts = pts[
                near_mask
            ]

            line_mean = np.mean(
                near_pts,
                axis=0
            )

            attraction_strength = (
                0.12
                + 0.38
                * visual_response
                + 0.10
                * self.g_attention_bias
            )

            attraction_strength = float(
                np.clip(
                    attraction_strength,
                    0.10,
                    0.55
                )
            )

            self.anchor_x += (
                attraction_strength
                * (
                    line_mean[0]
                    - self.anchor_x
                )
            )

            self.anchor_y += (
                attraction_strength
                * (
                    line_mean[1]
                    - self.anchor_y
                )
            )

        # =================================================
        # 4. Cell-cell repulsion
        # =================================================

        REPULSION_RADIUS = 5.0
        REPULSION_STRENGTH = 0.8

        repulsion_x = 0.0
        repulsion_y = 0.0

        for other in id_to_cell.values():

            if other.id == self.id:
                continue

            dx = (
                self.anchor_x
                - other.anchor_x
            )

            dy = (
                self.anchor_y
                - other.anchor_y
            )

            dist = (
                np.sqrt(
                    dx * dx
                    + dy * dy
                )
                + 1e-6
            )

            if dist < REPULSION_RADIUS:

                force = (
                    REPULSION_RADIUS
                    - dist
                ) / REPULSION_RADIUS

                repulsion_x += (
                    dx / dist
                ) * force

                repulsion_y += (
                    dy / dist
                ) * force

        self.anchor_x += (
            REPULSION_STRENGTH
            * repulsion_x
        )

        self.anchor_y += (
            REPULSION_STRENGTH
            * repulsion_y
        )

        # =================================================
        # 5. Active exploration
        # =================================================

        exploration_strength = (
            0.30
            + 0.40
            * (
                1.0
                - visual_response
            )
        )

        self.anchor_x += rng.normal(
            0,
            exploration_strength
        )

        self.anchor_y += rng.normal(
            0,
            exploration_strength
        )

        # =================================================
        # 6. Occasional larger jump
        # =================================================

        if rng.random() < 0.08:

            jump_strength = 2.5

            self.anchor_x += rng.normal(
                0,
                jump_strength
            )

            self.anchor_y += rng.normal(
                0,
                jump_strength
            )

        # =================================================
        # 7. Boundary
        # =================================================

        self.anchor_x = float(
            np.clip(
                self.anchor_x,
                0.0,
                IMG_SIZE
            )
        )

        self.anchor_y = float(
            np.clip(
                self.anchor_y,
                0.0,
                IMG_SIZE
            )
        )

        # =================================================
        # 8. Record trajectory
        # =================================================

        self.anchor_trajectory.append(
            (
                float(self.anchor_x),
                float(self.anchor_y)
            )
        )

        # =================================================
        # 9. Relational response
        # =================================================

        relational_response = 0.0

        if enable_links:

            neighbors = []
            weights = []

            for nid, weight in self.links.items():

                if nid in id_to_cell:

                    neighbors.append(
                        id_to_cell[nid]
                    )

                    weights.append(
                        weight
                    )

            if neighbors:

                delayed_acts = []

                for n in neighbors:

                    delay = int(
                        self.g_synaptic_delay
                    )

                    # 本当に delay させる
                    delayed_index = max(
                        0,
                        len(
                            n.activation_history
                        )
                        - 1
                        - delay
                    )

                    delayed_acts.append(
                        n.activation_history[
                            delayed_index
                        ]
                    )

                relational_response = (
                    np.sum(
                        np.asarray(
                            delayed_acts
                        )
                        *
                        np.asarray(
                            weights
                        )
                    )
                    /
                    (
                        np.sum(
                            weights
                        )
                        + 1e-9
                    )
                )

        # =================================================
        # 10. Visual + relational
        # =================================================

        g_r = (
            self.g_relational_bias
            if enable_links
            else 0.0
        )

        final_score = (
            (
                1.0
                - g_r
            )
            * visual_response
            +
            g_r
            * relational_response
        )

        return float(
            np.clip(
                final_score,
                0.0,
                1.0
            )
        )

    # =====================================================
    # Update activation history
    # =====================================================

    def update_history(
        self,
        act
    ):

        self.activation_history.append(
            float(act)
        )

        if len(
            self.activation_history
        ) > 5:

            self.activation_history.pop(
                0
            )

        self.last_activation = float(
            act
        )

    # =====================================================
    # Adapt
    # =====================================================

    def adapt(
        self,
        coordinate_points,
        act_strength,
        current_label=None
    ):

        self.age += 1

        if len(coordinate_points) == 0:
            return

        # -------------------------------------------------
        # Correct / unlabeled adaptation
        # -------------------------------------------------

        if (
            current_label is None
            or
            current_label
            ==
            self.target_label
        ):

            pts = np.asarray(
                coordinate_points,
                dtype=float
            )

            dists = np.hypot(
                pts[:, 0] - self.anchor_x,
                pts[:, 1] - self.anchor_y
            )

            if np.min(dists) < 20.0:

                closest_pt = pts[
                    np.argmin(dists)
                ]

                lr = (
                    self.g_learning_rate
                    *
                    (
                        0.5
                        + 1.5
                        * act_strength
                    )
                )

                lr = float(
                    np.clip(
                        lr,
                        0.02,
                        0.25
                    )
                )

                self.anchor_x += (
                    lr
                    * (
                        closest_pt[0]
                        - self.anchor_x
                    )
                )

                self.anchor_y += (
                    lr
                    * (
                        closest_pt[1]
                        - self.anchor_y
                    )
                )

                self.anchor_x = float(
                    np.clip(
                        self.anchor_x,
                        0.0,
                        IMG_SIZE
                    )
                )

                self.anchor_y = float(
                    np.clip(
                        self.anchor_y,
                        0.0,
                        IMG_SIZE
                    )
                )

                # 学習による移動も記録
                self.anchor_trajectory.append(
                    (
                        float(self.anchor_x),
                        float(self.anchor_y)
                    )
                )

            # -------------------------------------------------
            # Energy reinforcement
            # -------------------------------------------------

            if current_label is not None:

                self.energy = min(
                    1.5,
                    self.energy
                    + act_strength
                    * 0.3
                )

        # -------------------------------------------------
        # Wrong label
        # -------------------------------------------------

        else:

            self.energy -= 0.05

    # =====================================================
    # Divide
    # =====================================================

    def divide(
        self,
        next_id
    ):

        child = LineAdaptiveCell(
            next_id
        )

        child.anchor_x = float(
            np.clip(
                self.anchor_x
                + rng.normal(
                    0,
                    3.0
                ),
                0.0,
                IMG_SIZE
            )
        )

        child.anchor_y = float(
            np.clip(
                self.anchor_y
                + rng.normal(
                    0,
                    3.0
                ),
                0.0,
                IMG_SIZE
            )
        )

        child.geom_sensitivity = float(
            max(
                0.5,
                self.geom_sensitivity
                + rng.normal(
                    0,
                    0.2
                )
            )
        )

        m_rate = (
            self.g_mutation_rate
        )

        child.g_relational_bias = float(
            np.clip(
                self.g_relational_bias
                + rng.normal(
                    0,
                    m_rate
                ),
                0.0,
                1.0
            )
        )

        child.g_attention_bias = float(
            np.clip(
                self.g_attention_bias
                + rng.normal(
                    0,
                    m_rate
                ),
                0.0,
                1.0
            )
        )

        child.g_resonance_gain = float(
            np.clip(
                self.g_resonance_gain
                + rng.normal(
                    0,
                    m_rate
                ),
                0.1,
                5.0
            )
        )

        child.g_synaptic_delay = (
            rng.choice(
                [1, 2, 3]
            )
            if rng.random() < m_rate
            else self.g_synaptic_delay
        )

        child.g_learning_rate = float(
            np.clip(
                self.g_learning_rate
                + rng.normal(
                    0,
                    m_rate * 0.2
                ),
                0.01,
                0.5
            )
        )

        child.g_mutation_rate = float(
            np.clip(
                self.g_mutation_rate
                + rng.normal(
                    0,
                    0.02
                ),
                0.01,
                0.4
            )
        )

        child.target_label = (
            self.target_label
            if rng.random() > 0.08
            else
            1 - self.target_label
        )

        # -------------------------------------------------
        # Division cost
        # -------------------------------------------------

        self.energy *= 0.50

        return child


# =========================================================
# 4. 生態系管理
# =========================================================

class StochasticMetaEcosystem:

    def __init__(
        self,
        init_count=INITIAL_CELLS
    ):

        self.pool = [
            LineAdaptiveCell(i)
            for i in range(init_count)
        ]

        self.max_id = init_count

        # -------------------------------------------------
        # Link sparsity
        # -------------------------------------------------

        self.max_links_per_cell = 6

        self.min_link_weight = 0.12

    # =====================================================
    # Link pruning
    # =====================================================

    def prune_links(self):

        valid_ids = {
            c.id
            for c in self.pool
        }

        for c in self.pool:

            # -------------------------------------------------
            # Remove invalid / weak links
            # -------------------------------------------------

            c.links = {
                nid: weight
                for nid, weight
                in c.links.items()
                if (
                    nid in valid_ids
                    and
                    weight >= self.min_link_weight
                )
            }

            # -------------------------------------------------
            # Keep strongest links
            # -------------------------------------------------

            if (
                len(c.links)
                >
                self.max_links_per_cell
            ):

                sorted_links = sorted(
                    c.links.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                c.links = dict(
                    sorted_links[
                        :self.max_links_per_cell
                    ]
                )

    # =====================================================
    # Cycle
    # =====================================================

    def cycle(
        self,
        sample_coords,
        label,
        steps=28,
        is_training=True
    ):

        # =================================================
        # Reset dynamic state
        # =================================================

        for c in self.pool:

            c.reset_dynamic_state()

        # =================================================
        # Repeated perception
        # =================================================

        for t in range(steps):

            id_map = {
                c.id: c
                for c in self.pool
            }

            # -------------------------------------------------
            # Perception + anchor movement
            # -------------------------------------------------

            for c in self.pool:

                score = c.access(
                    sample_coords,
                    id_map,
                    enable_links=True
                )

                c.update_history(
                    score
                )

            # =================================================
            # Hebbian learning
            # =================================================

            for i, c1 in enumerate(
                self.pool
            ):

                for j, c2 in enumerate(
                    self.pool
                ):

                    if i == j:
                        continue

                    if (
                        c1.last_activation
                        > 0.2
                        and
                        c2.last_activation
                        > 0.2
                    ):

                        old_weight = (
                            c1.links.get(
                                c2.id,
                                0.1
                            )
                        )

                        new_weight = (
                            old_weight
                            +
                            0.25
                            *
                            (
                                c1.last_activation
                                *
                                c2.last_activation
                            )
                        )

                        c1.links[
                            c2.id
                        ] = float(
                            np.clip(
                                new_weight,
                                0.0,
                                1.0
                            )
                        )

            # -------------------------------------------------
            # Prevent all-to-all connectivity
            # -------------------------------------------------

            self.prune_links()

        # =====================================================
        # Training adaptation / evolution
        # =====================================================

        if is_training:

            # -------------------------------------------------
            # Determine division slots BEFORE modifying pool
            # -------------------------------------------------

            available_slots = max(
                0,
                MAX_CELLS
                - len(self.pool)
            )

            candidates = [
                c
                for c in self.pool
                if c.energy > 0.90
            ]

            rng.shuffle(
                candidates
            )

            candidates = candidates[
                :available_slots
            ]

            # -------------------------------------------------
            # Division
            # -------------------------------------------------

            new_cells = []

            for c in candidates:

                self.max_id += 1

                child = c.divide(
                    self.max_id
                )

                new_cells.append(
                    child
                )

            # -------------------------------------------------
            # Slow adaptation
            # -------------------------------------------------

            for c in self.pool:

                c.adapt(
                    sample_coords,
                    c.last_activation,
                    label
                )

                c.energy *= 0.95

            # -------------------------------------------------
            # Add new cells
            # -------------------------------------------------

            self.pool.extend(
                new_cells
            )

            # -------------------------------------------------
            # Remove weak cells
            # -------------------------------------------------

            self.pool = [
                c
                for c in self.pool
                if (
                    c.energy > 0.05
                    or
                    c.age < 4
                )
            ]

            # -------------------------------------------------
            # Minimum population
            # -------------------------------------------------

            while len(
                self.pool
            ) < MIN_CELLS:

                self.max_id += 1

                self.pool.append(
                    LineAdaptiveCell(
                        self.max_id
                    )
                )

            # -------------------------------------------------
            # Strong population safety limit
            # -------------------------------------------------

            if len(self.pool) > MAX_CELLS:

                ranked = sorted(
                    self.pool,
                    key=lambda c: c.energy,
                    reverse=True
                )

                self.pool = ranked[
                    :MAX_CELLS
                ]

            # -------------------------------------------------
            # Link pruning
            # -------------------------------------------------

            self.prune_links()

        # =====================================================
        # Non-destructive TTA
        # =====================================================

        else:

            for c in self.pool:

                c.adapt(
                    sample_coords,
                    c.last_activation,
                    current_label=None
                )

                if (
                    c.last_activation
                    > 0.3
                ):

                    c.energy = min(
                        1.2,
                        c.energy + 0.02
                    )


# =========================================================
# 5. Feature Extraction
# =========================================================

def extract_features(
    self,
    X_coords_data,
    steps=28,
    is_training=False
):

    features = []

    for sample in X_coords_data:

        # =================================================
        # ALWAYS use a copy
        #
        # Feature extraction itself must not permanently
        # modify the learned ecosystem.
        # =================================================

        working_eco = copy.deepcopy(
            self
        )

        working_eco.cycle(
            sample,
            label=None,
            steps=steps,
            is_training=False
        )

        target_pool = (
            working_eco.pool
        )

        # -------------------------------------------------
        # Activation values
        # -------------------------------------------------

        act_now = np.asarray(
            [
                c.last_activation
                for c in target_pool
            ],
            dtype=float
        )

        act_prev = np.asarray(
            [
                (
                    c.activation_history[-2]
                    if len(
                        c.activation_history
                    ) >= 2
                    else 0.0
                )
                for c in target_pool
            ],
            dtype=float
        )

        # -------------------------------------------------
        # Basic statistics
        # -------------------------------------------------

        feat_row = [

            np.mean(
                act_now
            ),

            np.max(
                act_now
            ),

            np.std(
                act_now
            ),

            np.mean(
                act_prev
            ),

            np.max(
                act_prev
            ),

            np.std(
                act_prev
            )
        ]

        # -------------------------------------------------
        # Percentiles
        # -------------------------------------------------

        for q in [
            10,
            25,
            50,
            75,
            90
        ]:

            feat_row.append(
                np.percentile(
                    act_now,
                    q
                )
            )

            feat_row.append(
                np.percentile(
                    act_prev,
                    q
                )
            )

        # -------------------------------------------------
        # Top-k activation
        # -------------------------------------------------

        for k_val in [
            3,
            8,
            15
        ]:

            k_now = min(
                k_val,
                len(act_now)
            )

            k_prev = min(
                k_val,
                len(act_prev)
            )

            feat_row.append(
                np.mean(
                    np.partition(
                        act_now,
                        -k_now
                    )[-k_now:]
                )
            )

            feat_row.append(
                np.mean(
                    np.partition(
                        act_prev,
                        -k_prev
                    )[-k_prev:]
                )
            )

        # -------------------------------------------------
        # Histograms
        # -------------------------------------------------

        counts_now, _ = np.histogram(
            act_now,
            bins=8,
            range=(0.0, 1.0)
        )

        counts_prev, _ = np.histogram(
            act_prev,
            bins=8,
            range=(0.0, 1.0)
        )

        feat_row.extend(
            counts_now
            /
            (
                len(act_now)
                + 1e-9
            )
        )

        feat_row.extend(
            counts_prev
            /
            (
                len(act_prev)
                + 1e-9
            )
        )

        features.append(
            feat_row
        )

    return np.asarray(
        features,
        dtype=float
    )


# attach as class method
StochasticMetaEcosystem.extract_features = (
    extract_features
)


# =========================================================
# 6. 評価
# =========================================================

def evaluate_concept_learning(
    eco,
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print(
        "\n[Feature Extraction] Training..."
    )

    X_tr_feats = eco.extract_features(
        X_tr,
        steps=TRAIN_FEATURE_STEPS,
        is_training=False
    )

    print(
        "[Feature Extraction] "
        "Testing with TTA..."
    )

    X_te_feats = eco.extract_features(
        X_te,
        steps=TEST_TTA_STEPS,
        is_training=False
    )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_tr_scaled = scaler.fit_transform(
        X_tr_feats
    )

    X_te_scaled = scaler.transform(
        X_te_feats
    )

    print(
        "Training feature matrix:",
        X_tr_scaled.shape
    )

    print(
        "Testing feature matrix :",
        X_te_scaled.shape
    )

    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    clf = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=SEED
    )

    clf.fit(
        X_tr_scaled,
        y_tr
    )

    y_pred = clf.predict(
        X_te_scaled
    )

    return {
        "Accuracy":
            accuracy_score(
                y_te,
                y_pred
            ),

        "Precision":
            precision_score(
                y_te,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_te,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_te,
                y_pred,
                zero_division=0
            )
    }, clf, scaler


# =========================================================
# 7. Anchor Trajectory Statistics
# =========================================================

def get_anchor_statistics(
    eco
):

    records = []

    for c in eco.pool:

        trajectory = np.asarray(
            c.anchor_trajectory,
            dtype=float
        )

        if len(trajectory) < 2:
            continue

        # -------------------------------------------------
        # Step-by-step movement
        # -------------------------------------------------

        diffs = np.diff(
            trajectory,
            axis=0
        )

        step_distances = np.linalg.norm(
            diffs,
            axis=1
        )

        # -------------------------------------------------
        # Total path
        # -------------------------------------------------

        total_path = float(
            np.sum(
                step_distances
            )
        )

        # -------------------------------------------------
        # Start -> final displacement
        # -------------------------------------------------

        displacement = float(
            np.linalg.norm(
                trajectory[-1]
                -
                trajectory[0]
            )
        )

        # -------------------------------------------------
        # Mean step
        # -------------------------------------------------

        mean_step = float(
            np.mean(
                step_distances
            )
        )

        # -------------------------------------------------
        # Max step
        # -------------------------------------------------

        max_step = float(
            np.max(
                step_distances
            )
        )

        # -------------------------------------------------
        # Exploration ratio
        #
        #   Path / displacement
        #
        # 1.0に近い:
        #   直線的移動
        #
        # 大きい:
        #   迂回・探索的軌跡
        # -------------------------------------------------

        exploration_ratio = (
            total_path
            /
            (
                displacement
                + 1e-9
            )
        )

        records.append({

            "cell_id":
                c.id,

            "total_path":
                total_path,

            "displacement":
                displacement,

            "mean_step":
                mean_step,

            "max_step":
                max_step,

            "exploration_ratio":
                exploration_ratio,

            "num_steps":
                len(step_distances),

            "target_label":
                c.target_label
        })

    return records


def print_anchor_statistics(
    eco,
    title="Anchor Movement Statistics"
):

    records = get_anchor_statistics(
        eco
    )

    if not records:

        print(
            "No anchor trajectory recorded."
        )

        return

    total_path = np.asarray(
        [
            r["total_path"]
            for r in records
        ],
        dtype=float
    )

    displacement = np.asarray(
        [
            r["displacement"]
            for r in records
        ],
        dtype=float
    )

    mean_step = np.asarray(
        [
            r["mean_step"]
            for r in records
        ],
        dtype=float
    )

    max_step = np.asarray(
        [
            r["max_step"]
            for r in records
        ],
        dtype=float
    )

    exploration = np.asarray(
        [
            r["exploration_ratio"]
            for r in records
        ],
        dtype=float
    )

    print(
        "\n========================================================="
    )

    print(
        title
    )

    print(
        "========================================================="
    )

    print(
        f"Number of Cells        : "
        f"{len(records)}"
    )

    print(
        f"Mean Total Path        : "
        f"{np.mean(total_path):.3f}"
    )

    print(
        f"Mean Displacement      : "
        f"{np.mean(displacement):.3f}"
    )

    print(
        f"Mean Step Distance     : "
        f"{np.mean(mean_step):.3f}"
    )

    print(
        f"Mean Exploration Ratio : "
        f"{np.mean(exploration):.3f}"
    )

    print(
        f"Max Total Path         : "
        f"{np.max(total_path):.3f}"
    )

    print(
        f"Max Displacement       : "
        f"{np.max(displacement):.3f}"
    )

    print(
        f"Max Step Distance      : "
        f"{np.max(max_step):.3f}"
    )

    print(
        f"Max Exploration Ratio  : "
        f"{np.max(exploration):.3f}"
    )

    print(
        "========================================================="
    )

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    print(
        "\nExploration Ratio Distribution:"
    )

    for q in [
        10,
        25,
        50,
        75,
        90
    ]:

        print(
            f"  P{q:02d}: "
            f"{np.percentile(exploration, q):.3f}"
        )

    print(
        "========================================================="
    )


# =========================================================
# 8. Anchor Statistics by Cell Label
# =========================================================

def print_anchor_statistics_by_label(
    eco
):

    records = get_anchor_statistics(
        eco
    )

    if not records:
        return

    print(
        "\n========================================================="
    )

    print(
        "Anchor Movement Statistics by Target Label"
    )

    print(
        "========================================================="
    )

    for label in [
        0,
        1
    ]:

        subset = [
            r
            for r in records
            if r["target_label"] == label
        ]

        if not subset:
            continue

        total_path = np.asarray(
            [
                r["total_path"]
                for r in subset
            ],
            dtype=float
        )

        displacement = np.asarray(
            [
                r["displacement"]
                for r in subset
            ],
            dtype=float
        )

        exploration = np.asarray(
            [
                r["exploration_ratio"]
                for r in subset
            ],
            dtype=float
        )

        label_name = (
            "Closed"
            if label == 0
            else "Open"
        )

        print(
            f"\nTarget Label: "
            f"{label} ({label_name})"
        )

        print(
            f"  Cells              : "
            f"{len(subset)}"
        )

        print(
            f"  Mean Total Path    : "
            f"{np.mean(total_path):.3f}"
        )

        print(
            f"  Mean Displacement  : "
            f"{np.mean(displacement):.3f}"
        )

        print(
            f"  Mean Exploration   : "
            f"{np.mean(exploration):.3f}"
        )

    print(
        "\n========================================================="
    )


# =========================================================
# 9. Anchor Trajectory Plot
# =========================================================

def plot_anchor_trajectories(
    eco,
    sample_coords,
    title="Dynamic Anchor Trajectories"
):

    plt.figure(
        figsize=(9, 9)
    )

    # -----------------------------------------------------
    # Input shape
    # -----------------------------------------------------

    if len(sample_coords) > 0:

        pts = np.asarray(
            sample_coords,
            dtype=float
        )

        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            c="lightgray",
            s=10,
            alpha=0.35,
            label="Input Shape"
        )

    # -----------------------------------------------------
    # Trajectories
    # -----------------------------------------------------

    for c in eco.pool:

        trajectory = np.asarray(
            c.anchor_trajectory,
            dtype=float
        )

        if len(trajectory) < 2:
            continue

        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            alpha=0.35,
            linewidth=1
        )

        # -------------------------------------------------
        # Start
        # -------------------------------------------------

        plt.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            s=20,
            alpha=0.4
        )

        # -------------------------------------------------
        # Final
        # -------------------------------------------------

        plt.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            s=50,
            alpha=0.85
        )

    plt.xlim(
        0,
        IMG_SIZE
    )

    plt.ylim(
        IMG_SIZE,
        0
    )

    plt.gca().set_aspect(
        "equal"
    )

    plt.xlabel(
        "X"
    )

    plt.ylabel(
        "Y"
    )

    plt.title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.4
    )

    plt.legend(
        loc="upper right"
    )

    plt.show()


# =========================================================
# 10. Research Dashboard
# =========================================================

def plot_bp015_research_dashboard(
    eco,
    sample_coords,
    sample_label,
    eco_before=None,
    X_test_coords=None,
    y_test=None
):

    # =====================================================
    # IMPORTANT:
    #
    # Visualization also uses a copy.
    # It does not modify the learned ecosystem.
    # =====================================================

    demo_eco = copy.deepcopy(
        eco
    )

    demo_eco.cycle(
        sample_coords,
        label=None,
        steps=28,
        is_training=False
    )

    fig = plt.figure(
        figsize=(14, 12)
    )

    plt.subplots_adjust(
        wspace=0.3,
        hspace=0.3
    )

    # =====================================================
    # 1. Anchor positions
    # =====================================================

    ax1 = plt.subplot(
        2,
        2,
        1
    )

    x_after = [
        c.anchor_x
        for c in demo_eco.pool
    ]

    y_after = [
        c.anchor_y
        for c in demo_eco.pool
    ]

    colors_after = [
        "navy"
        if c.target_label == 0
        else
        "crimson"
        for c in demo_eco.pool
    ]

    # -----------------------------------------------------
    # Before
    # -----------------------------------------------------

    if eco_before is not None:

        x_before = [
            c.anchor_x
            for c in eco_before.pool
        ]

        y_before = [
            c.anchor_y
            for c in eco_before.pool
        ]

        ax1.scatter(
            x_before,
            y_before,
            c="gray",
            alpha=0.35,
            s=30,
            label="Before",
            zorder=1
        )

    # -----------------------------------------------------
    # After
    # -----------------------------------------------------

    ax1.scatter(
        x_after,
        y_after,
        c=colors_after,
        s=60,
        edgecolors="k",
        alpha=0.85,
        label="After (TTA)",
        zorder=2
    )

    ax1.set_xlim(
        0,
        IMG_SIZE
    )

    ax1.set_ylim(
        IMG_SIZE,
        0
    )

    ax1.set_aspect(
        "equal"
    )

    ax1.set_title(
        "1. Cell Anchors with TTA\n"
        "(Navy: Closed, Crimson: Open)",
        fontsize=11,
        fontweight="bold"
    )

    ax1.legend(
        loc="upper right"
    )

    ax1.grid(
        True,
        linestyle=":",
        alpha=0.4
    )

    # =====================================================
    # 2. Activation Map
    # =====================================================

    ax2 = plt.subplot(
        2,
        2,
        2
    )

    if len(sample_coords) > 0:

        pts = np.asarray(
            sample_coords,
            dtype=float
        )

        ax2.scatter(
            pts[:, 0],
            pts[:, 1],
            c="gray",
            s=15,
            alpha=0.4,
            label="Input Shape"
        )

    # -----------------------------------------------------
    # Access on a copy
    # -----------------------------------------------------

    id_map = {
        c.id: c
        for c in demo_eco.pool
    }

    acts = []

    for c in demo_eco.pool:

        act = c.access(
            sample_coords,
            id_map,
            enable_links=True
        )

        acts.append(
            act
        )

    x_current = [
        c.anchor_x
        for c in demo_eco.pool
    ]

    y_current = [
        c.anchor_y
        for c in demo_eco.pool
    ]

    sc = ax2.scatter(
        x_current,
        y_current,
        c=acts,
        cmap="plasma",
        s=70,
        edgecolors="k",
        zorder=3
    )

    plt.colorbar(
        sc,
        ax=ax2,
        label="Activation Value"
    )

    label_name = (
        "Closed"
        if sample_label == 0
        else
        "Open"
    )

    ax2.set_xlim(
        0,
        IMG_SIZE
    )

    ax2.set_ylim(
        IMG_SIZE,
        0
    )

    ax2.set_aspect(
        "equal"
    )

    ax2.set_title(
        f"2. Activation Map on Sample "
        f"({label_name})",
        fontsize=11,
        fontweight="bold"
    )

    ax2.grid(
        True,
        linestyle=":",
        alpha=0.4
    )

    # =====================================================
    # 3. Link Network
    # =====================================================

    ax3 = plt.subplot(
        2,
        2,
        3
    )

    G = nx.Graph()

    pos = {}

    for c in demo_eco.pool:

        G.add_node(
            c.id,
            label=c.target_label
        )

        pos[c.id] = (
            c.anchor_x,
            -c.anchor_y
        )

    for c in demo_eco.pool:

        for target_id, weight in (
            c.links.items()
        ):

            if (
                weight > 0.15
                and
                target_id in G
            ):

                G.add_edge(
                    c.id,
                    target_id,
                    weight=weight
                )

    node_colors = [
        "navy"
        if G.nodes[n].get(
            "label",
            0
        ) == 0
        else
        "crimson"
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=50,
        ax=ax3,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.3,
        edge_color="gray",
        ax=ax3
    )

    ax3.set_title(
        "3. Sparse Cell Synaptic Link Network",
        fontsize=11,
        fontweight="bold"
    )

    ax3.axis(
        "off"
    )

    # =====================================================
    # 4. PCA
    # =====================================================

    ax4 = plt.subplot(
        2,
        2,
        4
    )

    if (
        X_test_coords is not None
        and
        y_test is not None
        and
        len(X_test_coords) >= 2
    ):

        X_feats = eco.extract_features(
            X_test_coords,
            steps=TEST_TTA_STEPS,
            is_training=False
        )

        # PCA requires at least 2 features
        if X_feats.shape[1] >= 2:

            pca = PCA(
                n_components=2
            )

            X_pca = pca.fit_transform(
                X_feats
            )

            for lbl, col, name in zip(
                [0, 1],
                ["navy", "crimson"],
                ["Closed", "Open"]
            ):

                mask = (
                    y_test == lbl
                )

                if np.any(mask):

                    ax4.scatter(
                        X_pca[mask, 0],
                        X_pca[mask, 1],
                        c=col,
                        label=name,
                        alpha=0.8,
                        edgecolors="k",
                        s=60
                    )

            ax4.set_xlabel(
                "PCA Component 1"
            )

            ax4.set_ylabel(
                "PCA Component 2"
            )

            explained = (
                np.sum(
                    pca.explained_variance_ratio_
                )
                * 100
            )

            ax4.set_title(
                f"4. Feature Space PCA "
                f"(Explained Var: "
                f"{explained:.1f}%)",
                fontsize=11,
                fontweight="bold"
            )

            ax4.legend()

            ax4.grid(
                True,
                linestyle=":",
                alpha=0.4
            )

    plt.suptitle(
        "Line Adaptive Ecosystem Internal State Dashboard\n"
        "(Dynamic Anchors + Repulsion + "
        "Exploration + Sparse Links)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.show()


# =========================================================
# 11. Pixel Count vs Logistic Output
# =========================================================

def plot_pixel_vs_logistic_output(
    eco,
    X_te,
    y_te,
    X_tr,
    y_tr
):

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    X_tr_feats = eco.extract_features(
        X_tr,
        steps=TRAIN_FEATURE_STEPS,
        is_training=False
    )

    X_te_feats = eco.extract_features(
        X_te,
        steps=TEST_TTA_STEPS,
        is_training=False
    )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_tr_scaled = scaler.fit_transform(
        X_tr_feats
    )

    X_te_scaled = scaler.transform(
        X_te_feats
    )

    # -----------------------------------------------------
    # Logistic regression
    # -----------------------------------------------------

    clf = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=SEED
    )

    clf.fit(
        X_tr_scaled,
        y_tr
    )

    # -----------------------------------------------------
    # Probability of Open
    # -----------------------------------------------------

    y_probs = clf.predict_proba(
        X_te_scaled
    )[:, 1]

    # -----------------------------------------------------
    # Pixel counts
    # -----------------------------------------------------

    pixel_counts = np.asarray(
        [
            len(sample)
            for sample in X_te
        ],
        dtype=float
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    closed_mask = (
        y_te == 0
    )

    plt.scatter(
        pixel_counts[
            closed_mask
        ],
        y_probs[
            closed_mask
        ],
        marker="o",
        color="navy",
        s=60,
        alpha=0.8,
        label="Closed Curve"
    )

    open_mask = (
        y_te == 1
    )

    plt.scatter(
        pixel_counts[
            open_mask
        ],
        y_probs[
            open_mask
        ],
        marker="x",
        color="crimson",
        s=60,
        alpha=0.8,
        label="Open Curve"
    )

    plt.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="Decision Boundary"
    )

    plt.xlabel(
        "Number of Pixels",
        fontsize=12,
        fontweight="bold"
    )

    plt.ylabel(
        "Probability of Open",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Pixel Count vs. Logistic Classifier Output",
        fontsize=14,
        fontweight="bold"
    )

    plt.legend(
        loc="best"
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    plt.ylim(
        -0.05,
        1.05
    )

    plt.show()


# =========================================================
# 12. Point Order Randomization
#
# np.where() 自体が生成する走査順を使わないように、
# 訓練・テストとも明示的に shuffle。
# =========================================================

def shuffle_point_sets(
    X_coords_data
):

    shuffled = []

    for sample in X_coords_data:

        arr = np.asarray(
            sample,
            dtype=float
        ).copy()

        rng.shuffle(
            arr
        )

        shuffled.append(
            arr
        )

    return shuffled


# =========================================================
# 13. Dataset Visualization
# =========================================================

def plot_dataset_examples(
    X_data,
    y_data,
    title,
    num_examples=6
):

    n = min(
        num_examples,
        len(X_data)
    )

    if n == 0:
        return

    cols = 3
    rows = int(
        np.ceil(
            n / cols
        )
    )

    plt.figure(
        figsize=(12, 4 * rows)
    )

    for i in range(n):

        ax = plt.subplot(
            rows,
            cols,
            i + 1
        )

        pts = np.asarray(
            X_data[i],
            dtype=float
        )

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=5,
            alpha=0.7
        )

        name = (
            "Closed"
            if y_data[i] == 0
            else
            "Open"
        )

        ax.set_title(
            f"Sample {i + 1}: {name}"
        )

        ax.set_xlim(
            0,
            IMG_SIZE
        )

        ax.set_ylim(
            IMG_SIZE,
            0
        )

        ax.set_aspect(
            "equal"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.4
        )

    plt.suptitle(
        title,
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# 14. Main
# =========================================================

if __name__ == "__main__":

    print(
        "========================================================="
    )

    print(
        "[BP015 Diverse Shape Experiment]"
    )

    print(
        "Dynamic Anchor Ecosystem"
    )

    print(
        "Attraction + Repulsion + "
        "Exploration + Sparse Links"
    )

    print(
        "========================================================="
    )

    # =====================================================
    # 1. Generate training data
    # =====================================================

    print(
        "\n[1] Generating training data..."
    )

    X_train, y_train = (
        generate_diverse_shapes(
            num_samples=6,
            img_size=IMG_SIZE,
            seed=101,
            family_type="train"
        )
    )

    # =====================================================
    # 2. Generate test data
    # =====================================================

    print(
        "[2] Generating test data..."
    )

    X_test, y_test = (
        generate_diverse_shapes(
            num_samples=50,
            img_size=IMG_SIZE,
            seed=999,
            family_type="test"
        )
    )

    # =====================================================
    # 3. Explicitly shuffle point order
    # =====================================================

    print(
        "[3] Shuffling point sets..."
    )

    X_train_shuffled = (
        shuffle_point_sets(
            X_train
        )
    )

    X_test_shuffled = (
        shuffle_point_sets(
            X_test
        )
    )

    # =====================================================
    # Dataset information
    # =====================================================

    print(
        f" |- Train Family : "
        f"{len(X_train_shuffled)} shapes "
        f"(Closed: {sum(y_train == 0)}, "
        f"Open: {sum(y_train == 1)})"
    )

    print(
        f" |- Test Family  : "
        f"{len(X_test_shuffled)} shapes "
        f"(Closed: {sum(y_test == 0)}, "
        f"Open: {sum(y_test == 1)})"
    )

    # =====================================================
    # Optional dataset visualization
    # =====================================================

    plot_dataset_examples(
        X_train_shuffled,
        y_train,
        title="Training Dataset",
        num_examples=6
    )

    plot_dataset_examples(
        X_test_shuffled,
        y_test,
        title="Unseen Test Dataset",
        num_examples=6
    )

    # =====================================================
    # 4. Initialize ecosystem
    # =====================================================

    print(
        "\n[4] Initializing ecosystem..."
    )

    eco = StochasticMetaEcosystem(
        init_count=INITIAL_CELLS
    )

    eco_before = copy.deepcopy(
        eco
    )

    print(
        f" |- Initial Cells       : "
        f"{len(eco.pool)}"
    )

    print(
        f" |- Maximum Cells       : "
        f"{MAX_CELLS}"
    )

    print(
        f" |- Maximum Links/Cell : "
        f"{eco.max_links_per_cell}"
    )

    print(
        f" |- Minimum Link Weight : "
        f"{eco.min_link_weight}"
    )

    # =====================================================
    # 5. Training
    # =====================================================

    print(
        "\n[5] Ecosystem Learning Progress..."
    )

    for epoch in range(
        NUM_EPOCHS
    ):

        print(
            f"\nEpoch "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        )

        # -------------------------------------------------
        # Shuffle training order each epoch
        # -------------------------------------------------

        order = np.arange(
            len(X_train_shuffled)
        )

        rng.shuffle(
            order
        )

        for count, idx in enumerate(
            order
        ):

            print(
                f"  Training sample "
                f"{count + 1}/"
                f"{len(order)}"
            )

            eco.cycle(
                X_train_shuffled[idx],
                label=int(
                    y_train[idx]
                ),
                steps=TRAIN_STEPS,
                is_training=True
            )

        print(
            f"  Cells after epoch: "
            f"{len(eco.pool)}"
        )

    print(
        "\nTraining finished."
    )

    print(
        f"Final Cell Count: "
        f"{len(eco.pool)}"
    )

    # =====================================================
    # 6. Learned ecosystem anchor statistics
    #
    # Note:
    # These trajectories correspond to the most recently
    # processed training sample because dynamic state is
    # reset for each cycle.
    # =====================================================

    print(
        "\n[6] Final Training-State "
        "Anchor Statistics..."
    )

    print_anchor_statistics(
        eco,
        title=(
            "Anchor Movement Statistics "
            "(Final Training State)"
        )
    )

    print_anchor_statistics_by_label(
        eco
    )

    # =====================================================
    # 7. Evaluate
    # =====================================================

    print(
        "\n[7] Evaluating on unseen test family..."
    )

    metrics, clf, scaler = (
        evaluate_concept_learning(
            eco,
            X_train_shuffled,
            y_train,
            X_test_shuffled,
            y_test
        )
    )

    # =====================================================
    # 8. Results
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BP015 Evaluation Results"
    )

    print(
        "Dynamic Anchors + Repulsion + "
        "Exploration + Sparse Links"
    )

    print(
        "=" * 60
    )

    print(
        f" |- Accuracy  : "
        f"{metrics['Accuracy'] * 100:.2f} %"
    )

    print(
        f" |- Precision : "
        f"{metrics['Precision'] * 100:.2f} %"
    )

    print(
        f" |- Recall    : "
        f"{metrics['Recall'] * 100:.2f} %"
    )

    print(
        f" |- F1 Score  : "
        f"{metrics['F1'] * 100:.2f} %"
    )

    print(
        "=" * 60
    )

    # =====================================================
    # 9. TTA trajectory on one unseen test sample
    # =====================================================

    print(
        "\n[8] Generating Test-Time "
        "Anchor Trajectory..."
    )

    trajectory_eco = copy.deepcopy(
        eco
    )

    trajectory_eco.cycle(
        X_test_shuffled[0],
        label=None,
        steps=RESEARCH_TRAJECTORY_STEPS,
        is_training=False
    )

    print_anchor_statistics(
        trajectory_eco,
        title=(
            "Anchor Movement Statistics "
            "(Test-Time Adaptation)"
        )
    )

    plot_anchor_trajectories(
        trajectory_eco,
        X_test_shuffled[0],
        title=(
            "Dynamic Anchor Trajectories "
            f"({RESEARCH_TRAJECTORY_STEPS} TTA Steps)"
        )
    )

    # =====================================================
    # 10. Research dashboard
    # =====================================================

    print(
        "\n[9] Generating Research Dashboard..."
    )

    plot_bp015_research_dashboard(
        eco,
        X_test_shuffled[0],
        y_test[0],
        eco_before=eco_before,
        X_test_coords=X_test_shuffled,
        y_test=y_test
    )

    # =====================================================
    # 11. Pixel count vs logistic output
    # =====================================================

    print(
        "\n[10] Generating Pixel Count "
        "vs Logistic Output Plot..."
    )

    plot_pixel_vs_logistic_output(
        eco,
        X_test_shuffled,
        y_test,
        X_train_shuffled,
        y_train
    )

    # =====================================================
    # 12. Final summary
    # =====================================================

    print(
        "\n========================================================="
    )

    print(
        "Experiment completed."
    )

    print(
        "========================================================="
    )

    print(
        f"Final cells: {len(eco.pool)}"
    )

    print(
        f"Accuracy: "
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{metrics['F1'] * 100:.2f}%"
    )

    print(
        "========================================================="
    )
