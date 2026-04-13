"""
ROI Extraction — CASIA-MS / MPDv2 / Generic palmprint datasets
==============================================================
All user-facing settings are in the CONFIG block at the top.
No edits needed below the  ═══ END OF CONFIG ═══  line.
"""

# ══════════════════════════════════════════════════════════════
#  USER CONFIG  — edit only this block
# ══════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────
DIR_SOURCE = "/home/pai-ng/Jamal/MPDv2"
DIR_OUTPUT = "/home/pai-ng/Jamal/MPDv2_eralign_roi"

# ── Dataset preset to use as primary config ────────────────────
#    "CASIA-MS"  |  "MPDv2"  |  "Generic"
DATASET_PRESET = "MPDv2"

# ── Debug mode ────────────────────────────────────────────────
#    True  → print per-image detail + save intermediate images
#    False → quiet, only print failures
DEBUG = True

# ── Save intermediate debug images for ONE specific image ──────
#    Set to a full path string to inspect a tricky image, or None
DEBUG_SINGLE_IMAGE = None
# DEBUG_SINGLE_IMAGE = "/home/pai-ng/Jamal/MPDv2/subject01/img001.jpg"

# ── Dataset-specific primary configs ──────────────────────────
DATASET_CONFIGS = {
    "CASIA-MS": {
        # segmentation
        "THRESHOLD_SEG"           : 90,
        "BI_THRESHOLD_SEG"        : True,    # True=fixed thresh, False=Otsu
        # edge detection
        "THRESHOLD_EDGE"          : 10,
        # blurring (localization stage)
        "BLUR"                    : False,
        "BLUR_SIGMA"              : 0.05,
        # erosion kernel — None = auto from image size
        "KERNEL_SIZE"             : (45, 45),
        # rotation stage
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 0.05,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 90,
        # resize ratios
        "ratio_rotate"            : 1.0,
        "ratio"                   : 1.0,
    },
    "MPDv2": {
        "THRESHOLD_SEG"           : 0,       # 0 = use Otsu
        "BI_THRESHOLD_SEG"        : False,
        "THRESHOLD_EDGE"          : 5,
        "BLUR"                    : True,
        "BLUR_SIGMA"              : 1.0,
        "KERNEL_SIZE"             : None,    # auto
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 1.0,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 0,
        "ratio_rotate"            : 2.0,
        "ratio"                   : 1.0,
    },
    "Generic": {
        "THRESHOLD_SEG"           : 0,
        "BI_THRESHOLD_SEG"        : False,
        "THRESHOLD_EDGE"          : 7,
        "BLUR"                    : True,
        "BLUR_SIGMA"              : 1.5,
        "KERNEL_SIZE"             : None,
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 1.0,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 0,
        "ratio_rotate"            : 2.0,
        "ratio"                   : 1.0,
    },
}

# ── Fallback configs tried in order when primary config fails ──
#    Each entry overrides only the keys it specifies;
#    the rest are inherited from the primary config.
FALLBACK_CONFIGS = [
    # fallback 1 — Otsu + more blur
    {
        "THRESHOLD_SEG"    : 0,
        "BI_THRESHOLD_SEG" : False,
        "THRESHOLD_EDGE"   : 5,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 2.0,
        "KERNEL_SIZE"      : None,
    },
    # fallback 2 — fixed low threshold + moderate blur
    {
        "THRESHOLD_SEG"    : 30,
        "BI_THRESHOLD_SEG" : True,
        "THRESHOLD_EDGE"   : 5,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 1.5,
        "KERNEL_SIZE"      : None,
    },
    # fallback 3 — higher threshold + aggressive blur
    {
        "THRESHOLD_SEG"    : 60,
        "BI_THRESHOLD_SEG" : True,
        "THRESHOLD_EDGE"   : 3,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 3.0,
        "KERNEL_SIZE"      : None,
    },
    # fallback 4 — very high threshold, minimal blur
    {
        "THRESHOLD_SEG"    : 120,
        "BI_THRESHOLD_SEG" : True,
        "THRESHOLD_EDGE"   : 3,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 0.5,
        "KERNEL_SIZE"      : None,
    },
]

# ══════════════════════════════════════════════════════════════
#  END OF CONFIG  — do not edit below this line
# ══════════════════════════════════════════════════════════════

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
from shapely.geometry import Polygon


# ──────────────────────────────────────────────────────────────
#  GaborFilter helpers
# ──────────────────────────────────────────────────────────────

def GaborFilter_cc(len_filter, sigma=4.6, delta=2.6, num_direction=6):
    assert num_direction % 2 == 0, 'num_direction should be an even number!'
    half_len = int(len_filter / 2)
    Filter = np.zeros((num_direction, len_filter, len_filter))
    for a in range(num_direction):
        theta   = np.pi / 2 - np.pi * a / num_direction
        kappa   = np.sqrt(2 * np.log(2)) * (delta + 1) / (delta - 1)
        w       = kappa / sigma
        fFactor1 = -w / (np.sqrt(2 * np.pi) * kappa)
        fFactor2 = -(w * w) / (8 * kappa * kappa)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for c in range(len_filter):
            x = c - half_len
            for r in range(len_filter):
                y  = r - half_len
                x1 = x * cos_theta + y * sin_theta
                y1 = y * cos_theta - x * sin_theta
                f_comp = fFactor1 * np.exp(fFactor2 * (4 * x1 * x1 + y1 * y1))
                Filter[a, r, c] = f_comp * np.cos(w * x1)
        Filter[a, :, :] -= Filter[a, :, :].mean()
    return Filter


def GaborFilter(ksize, num_direction, sigma, lambd, gamma):
    assert num_direction % 2 == 0, 'num_direction should be an even number!'
    half_size = int(ksize / 2)
    sigma2    = 2 * sigma ** 2
    Filter    = np.zeros((num_direction, ksize, ksize))
    l_min     = -half_size
    l_max     = half_size
    x, y      = np.meshgrid(range(l_min, l_max + 1), range(l_min, l_max + 1))
    for a in range(num_direction):
        theta     = np.pi * a / num_direction
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_theta   = x * cos_theta + y * sin_theta
        y_theta   = y * cos_theta - x * sin_theta
        term1     = np.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / sigma2)
        term2     = np.cos(2 * np.pi * x_theta / lambd)
        Filter[a] = term1 * term2
        Filter[a] -= Filter[a].mean()
    return Filter


def GaborArray(sigma=4.85, wavelength=14.1, ratio=1.92):
    halfLength = 17
    xmax, xmin = halfLength, -halfLength
    ymax, ymin = halfLength, -halfLength
    x, y       = np.meshgrid(range(xmin, xmax + 1), range(ymin, ymax + 1))
    mask       = np.ones((35, 35))
    for row in range(1, 36):
        for col in range(1, 36):
            if (row - 18) ** 2 + (col - 18) ** 2 > 289:
                mask[row - 1, col - 1] = 0
    gb_r = np.zeros((6, 35, 35))
    for oriIndex in range(1, 7):
        theta   = np.pi / 6 * (oriIndex - 1)
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb      = np.exp(-.5 * (x_theta ** 2 / sigma ** 2 +
                                y_theta ** 2 / (ratio * sigma) ** 2)) \
                  * np.cos(2 * np.pi / wavelength * x_theta)
        total      = (gb * mask).sum()
        meanInner  = total / mask.sum()
        gb         = gb - meanInner.mean()
        gb         = gb * mask
        gb_r[oriIndex - 1] = gb
    return gb_r


# ──────────────────────────────────────────────────────────────
#  PalmBasic
# ──────────────────────────────────────────────────────────────

class PalmBasic:

    def __init__(self):
        pass

    def gaussian_blur(self, img, kernel_size=(5, 5), sigma=2, blur=True):
        if blur:
            return cv2.GaussianBlur(img, kernel_size, sigma)
        return img

    def threshold_image(self, img, threshold_val=20, bi_threshold=True):
        if bi_threshold:
            _, thresholded = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            _, thresholded = cv2.threshold(img, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def compute_edist(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def find_largest_component(self, img):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        if len(stats) <= 1:
            return np.zeros_like(img), None, np.zeros_like(img)
        largest_component_label = np.argmax(stats[1:, -1]) + 1
        max_comp_img = np.zeros_like(img)
        max_comp_img[labels == largest_component_label] = 255
        max_comp_contour_coord = self.find_contour(max_comp_img)
        max_comp_contour_img   = cv2.drawContours(
            np.zeros_like(img, dtype=np.uint8),
            [max_comp_contour_coord], -1, (255), thickness=1)
        return max_comp_img, max_comp_contour_coord, max_comp_contour_img

    def find_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        return cnt

    def fill_contour(self, img, contour, filled=False):
        img_contour = np.zeros_like(img, dtype=np.uint8)
        thickness   = cv2.FILLED if filled else 1
        cv2.drawContours(img_contour, [contour], -1, (255), thickness=thickness)
        return img_contour

    def erode_binary_image(self, img, kernel_size=(7, 7),
                           iterations=1, th_binary=180):
        kernel  = np.ones(kernel_size, np.uint8)
        eroded  = cv2.erode(img, kernel, iterations=iterations)
        _, binaried = cv2.threshold(eroded, th_binary, 255, cv2.THRESH_BINARY)
        return binaried

    def resize_image(self, img, ratio):
        return cv2.resize(img, (img.shape[1] // ratio, img.shape[0] // ratio))

    def hull_image(self, img, img_contour):
        epsilon   = 0.01 * cv2.arcLength(img_contour, True)
        approx    = cv2.approxPolyDP(img_contour, epsilon, True)
        hull_coord = cv2.convexHull(approx, clockwise=False, returnPoints=True)
        hull_img  = cv2.drawContours(
            np.zeros_like(img, dtype=np.uint8),
            [hull_coord], -1, 255, thickness=cv2.FILLED)
        return hull_coord, hull_img

    def get_skeleton(self, img):
        img      = img == 255
        skeleton = skeletonize(img) * 255
        return skeleton

    def euclidean_distance(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sqrt(np.sum(np.square(point1 - point2)))

    def find_closest_path_graph(self, matrix):
        cleaned_matrix = np.zeros_like(matrix)
        graph          = self.build_undirected_graph(matrix)
        nodes_deg1     = [n for n in graph.nodes() if graph.degree[n] == 1]

        leftmost_node = None
        min_x = float('inf')
        for node in nodes_deg1:
            if node[1] < min_x:
                min_x = node[1]
                leftmost_node = node
        source_node = leftmost_node

        longest_path = []
        for node in nodes_deg1:
            if node != source_node:
                path = nx.shortest_path(graph, source=source_node, target=node)
                if len(path) > len(longest_path):
                    longest_path = path

        for point in longest_path:
            cleaned_matrix[point[0], point[1]] = 255

        return [cleaned_matrix, longest_path]

    def build_undirected_graph(self, matrix):
        non_zero    = np.argwhere(matrix == 255)
        min_row, min_col = np.min(non_zero, axis=0)
        max_row, max_col = np.max(non_zero, axis=0)
        graph      = nx.Graph()
        directions = [(-1, 0), (-1, -1), (0, -1), (1, -1)]
        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if matrix[i, j] == 255:
                    current = (i, j)
                    graph.add_node(current)
                    for d in directions:
                        nb = (i + d[0], j + d[1])
                        if (min_row <= nb[0] <= max_row and
                                min_col <= nb[1] <= max_col and
                                matrix[nb[0], nb[1]] == 255):
                            graph.add_edge(current, nb)
        return graph

    def judge_valley(self, path):
        rightmost_node  = max(path, key=lambda n: n[1])
        rightmost_index = path.index(rightmost_node)
        nodes_between1  = path[rightmost_index + 1:]
        nodes_between2  = path[:rightmost_index + 1]
        nodes_between   = min(len(nodes_between1), len(nodes_between2))
        flag_valley     = nodes_between > (len(path) / 3)
        return flag_valley, rightmost_index

    def find_farthest_point(self, path, th):
        len_path = len(path)
        path     = np.array(path)
        path     = path[:, [1, 0]]

        left_ind  = int(len_path / 5 * 2)
        left_node = path[left_ind]
        right_ind = len_path - 1
        right_node = path[right_ind]
        fix_vect  = left_node - right_node

        estimated_nodes = path[left_ind + 1:-1]
        estimated_vects = estimated_nodes - right_node
        angle_degrees   = self.compute_clockwise_angle_out(estimated_vects, fix_vect)
        flag_inter      = np.abs(np.abs(np.mean(angle_degrees)) -
                                 np.mean(np.abs(angle_degrees))) > 1.5

        while flag_inter and left_ind < (len_path - 1) and right_ind > 0:
            if left_ind < int(len_path / 3):
                left_ind  += 2
                left_node  = path[left_ind]
            elif right_ind > int(len_path / 5 * 4):
                right_ind  -= 2
                right_node  = path[right_ind]
            else:
                break
            fix_vect        = left_node - right_node
            estimated_nodes = path[left_ind + 1:right_ind]
            estimated_vects = estimated_nodes - right_node
            angle_degrees   = self.compute_clockwise_angle_out(estimated_vects, fix_vect)
            flag_inter      = np.abs(np.abs(np.mean(angle_degrees)) -
                                     np.mean(np.abs(angle_degrees))) > 1.5

        path_valid = path[left_ind:right_ind + 1]
        distances  = self.compute_perpendicular_dist(path_valid)
        max_idx    = np.argmax(distances)
        if distances[max_idx] > th:
            path_valid = path_valid[:max_idx + 4]
        return path_valid

    def compute_perpendicular_dist(self, path):
        x1, y1 = path[0]
        x2, y2 = path[-1]
        pv      = path[1:-1]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        distances = (abs(A * pv[:, 0] + B * pv[:, 1] + C) /
                     (np.sqrt(A ** 2 + B ** 2) + 1e-9))
        return distances

    def compute_clockwise_angle_out(self, fix_vect, moved_vect):
        cross_product = np.cross(moved_vect, fix_vect)
        dot_product   = np.dot(moved_vect, fix_vect.T)
        angle_relative = np.arctan2(cross_product, dot_product)
        return np.degrees(angle_relative)

    def find_closest_white_point(self, image, point):
        non_zero_points = cv2.findNonZero(image)
        distances       = np.linalg.norm(non_zero_points - point, axis=2)
        min_index       = np.argmin(distances)
        return tuple(non_zero_points[min_index][0]), distances[min_index][0]


# ──────────────────────────────────────────────────────────────
#  ROI geometry helper
# ──────────────────────────────────────────────────────────────

pad = 0

def extract_roi(p1, p2, img, color, thickness):
    d         = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / (d + 1e-9)
    normal    = np.array([direction[1], -direction[0]])

    s               = int(d / 6 * 7)
    half_extend     = int(d / 12)
    distance_to_sq  = int(d / 6 * 1)

    C = (p1 + (distance_to_sq + s) * normal - half_extend * direction).astype(np.int32)
    D = (C + s * direction).astype(np.int32)
    E = (D - s * normal).astype(np.int32)
    F = (C - s * normal).astype(np.int32)

    img_padded = cv2.copyMakeBorder(img, top=pad, bottom=pad,
                                    left=pad, right=pad,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    C += pad; D += pad; E += pad; F += pad

    pts    = np.array([C, D, E, F], dtype="float32")
    width  = max(np.linalg.norm(E - D), np.linalg.norm(F - C))
    height = max(np.linalg.norm(D - C), np.linalg.norm(E - F))

    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M       = cv2.getPerspectiveTransform(pts, dst)
    warped  = cv2.warpPerspective(img_padded, M, (int(width), int(height)))
    cv2.polylines(img_padded, [np.array([C, D, E, F])],
                  isClosed=True, color=color, thickness=thickness)
    corner_points = [(C[0], C[1]), (D[0], D[1]),
                     (E[0], E[1]), (F[0], F[1])]
    return warped, corner_points, img_padded


# ──────────────────────────────────────────────────────────────
#  GetROI — config-driven
# ──────────────────────────────────────────────────────────────

def _resolve_kernel(cfg, img_shape):
    """Return a concrete kernel size, deriving it from image size if needed."""
    if cfg.get("KERNEL_SIZE") is not None:
        return cfg["KERNEL_SIZE"]
    k = max(15, min(img_shape[0], img_shape[1]) // 10)
    k = k if k % 2 == 1 else k + 1
    return (k, k)


class GetROI(PalmBasic):

    def __init__(self, img, config):
        self.cfg          = config
        self.ratio_rotate = config["ratio_rotate"]
        self.ratio        = config["ratio"]
        self.ori_img      = img
        self.w, self.h    = img.shape
        self.w_rotate     = max(1, int(self.w / self.ratio_rotate))
        self.h_rotate     = max(1, int(self.h / self.ratio_rotate))
        self._kernel_size = _resolve_kernel(config, img.shape)

    # ── rotation stage ────────────────────────────────────────
    def run_rotate(self):
        cfg  = self.cfg
        img  = self.resize_image(self.ori_img, int(self.ratio_rotate))
        gabors = self._get_gabor_filters()

        img_blurred  = self.gaussian_blur(img,
                                          sigma=cfg["BLUR_SIGMA_ROTATE"],
                                          blur=cfg["BLUR_ROTATE"])
        rough_binary = self.threshold_image(img_blurred,
                                            threshold_val=cfg["THRESHOLD_SEG_ROTATE"],
                                            bi_threshold=cfg["BI_THRESHOLD_SEG_ROTATE"])
        max_comp_img, max_comp_contour_coord, _ = self.find_largest_component(rough_binary)
        hull_coord, hull_img = self.hull_image(img, max_comp_contour_coord)
        rough_edges_img      = self._detect_rough_edges(img_blurred & max_comp_img,
                                                        threshold_edge=cfg["THRESHOLD_EDGE"])
        edges_hull_img       = rough_edges_img & hull_img
        palm_labels, palm_sorted = self._select_rough_palm(edges_hull_img)

        self.rotation_angle = self.compute_rough_orientation(
            palm_labels, palm_sorted, hull_img, gabors)
        self.norm_img       = self.rotate_image(self.ori_img, self.rotation_angle)
        h, w                = self.norm_img.shape[:2]
        self.cut_norm_img   = self.norm_img[0:h, 0:int(w / 4 * 3)]

    # ── localization stage ────────────────────────────────────
    def run_localization(self):
        cfg = self.cfg
        img = self.cut_norm_img
        self.w, self.h = img.shape

        img_blurred  = self.gaussian_blur(img,
                                          sigma=cfg["BLUR_SIGMA"],
                                          blur=cfg["BLUR"])
        rough_binary = self.threshold_image(img_blurred,
                                            threshold_val=cfg["THRESHOLD_SEG"],
                                            bi_threshold=cfg["BI_THRESHOLD_SEG"])
        max_comp_img, max_comp_contour_coord, _ = self.find_largest_component(rough_binary)

        if max_comp_contour_coord is None:
            return False

        max_comp_img_small = self.erode_binary_image(max_comp_img,
                                                     kernel_size=self._kernel_size)
        if cv2.countNonZero(max_comp_img_small) == 0:
            max_comp_img_small = self.erode_binary_image(max_comp_img,
                                                         kernel_size=(15, 15))
        if cv2.countNonZero(max_comp_img_small) == 0:
            return False

        hull_coord, hull_img     = self.hull_image(img, max_comp_contour_coord)
        rough_edges_img          = self._detect_rough_edges(img_blurred & max_comp_img,
                                                            threshold_edge=cfg["THRESHOLD_EDGE"])
        hull_img_small           = self.erode_binary_image(hull_img,
                                                           kernel_size=self._kernel_size)
        if cv2.countNonZero(hull_img_small) == 0:
            hull_img_small = self.erode_binary_image(hull_img, kernel_size=(15, 15))
        if cv2.countNonZero(hull_img_small) == 0:
            return False

        hull_contour_coord_small = self.find_contour(hull_img_small)
        hull_contour_img_small   = self.fill_contour(img, hull_contour_coord_small)

        max_comp_contour_img     = cv2.drawContours(
            np.zeros_like(img, dtype=np.uint8),
            [max_comp_contour_coord], -1, 255, 1)
        concave_contour_img      = max_comp_contour_img & hull_img_small

        edges_hull_img           = rough_edges_img & hull_img_small
        self.edges_hull_img      = edges_hull_img

        palm_labels, palm_sorted = self._select_rough_palm(edges_hull_img)
        palm_len = (np.max(hull_coord[:, 0, :][:, 0]) -
                    np.min(hull_coord[:, 0, :][:, 0]))

        self.finger_edges_img = self.process_palm_contour_rough(
            palm_sorted, palm_labels, palm_len,
            hull_contour_img_small, concave_contour_img, max_comp_contour_img)
        self.palm_contour_img, self.palm_contour_coord = self.find_inner_contour(
            self.finger_edges_img)
        self.finger_lines = self.find_concave_finger(
            hull_contour_coord_small, hull_contour_img_small,
            palm_len, np.max(hull_coord[:, 0, :][:, 0]),
            self.palm_contour_coord)
        self.finger_lines_sorted = self.sort_and_surround_lines(self.finger_lines)
        self.keypoints           = self.detect_keypoints(self.finger_lines_sorted)
        self.show_img            = cv2.cvtColor(np.copy(self.norm_img),
                                               cv2.COLOR_GRAY2BGR)

        if self.keypoints is None:
            return False

        sorted_indices = np.argsort(self.keypoints[:, 1])
        self.keypoints = self.keypoints[sorted_indices]
        self.localize_keypoints(self.keypoints)
        return True

    # ── keypoint helpers (unchanged from original) ────────────
    def localize_keypoints(self, keypoints):
        if len(keypoints) == 4:
            d1 = (self.compute_edist(keypoints[0], keypoints[2]) +
                  self.compute_edist(keypoints[0], keypoints[1]))
            d2 = (self.compute_edist(keypoints[1], keypoints[3]) +
                  self.compute_edist(keypoints[2], keypoints[3]))
            if d2 > d1:
                kp1, kp2 = keypoints[0], keypoints[2]
            else:
                kp1, kp2 = keypoints[1], keypoints[3]
        elif len(keypoints) == 2:
            kp1, kp2 = keypoints[0], keypoints[1]
        self.keypoints_localization = [kp1, kp2]

    def detect_keypoints(self, lines_segment):
        def is_on_right(p1, p2, p):
            return ((p2[0] - p1[0]) * (p[1] - p1[1]) -
                    (p2[1] - p1[1]) * (p[0] - p1[0])) < 0

        def find_line_all_right(set1, set2):
            for p1 in set1:
                for p2 in set2:
                    ok = True
                    for p in set1:
                        if p is not p1 and is_on_right(p1, p2, p):
                            ok = False; break
                    if ok:
                        for p in set2:
                            if p is not p2 and is_on_right(p1, p2, p):
                                ok = False; break
                    if ok:
                        return (p1, p2)
            return (set1[np.argmax(set1[:, 0])],
                    set2[np.argmax(set2[:, 0])])

        if len(lines_segment) == 4:
            kps = np.zeros([4, 2], np.int32)
            r1  = find_line_all_right(lines_segment[0], lines_segment[2])
            r2  = find_line_all_right(lines_segment[1], lines_segment[3])
            kps[0], kps[2] = r1[0], r1[1]
            kps[1], kps[3] = r2[0], r2[1]
        elif len(lines_segment) == 3:
            kps = np.zeros([2, 2], np.int32)
            r   = find_line_all_right(lines_segment[0], lines_segment[2])
            kps[0], kps[1] = r[0], r[1]
        elif len(lines_segment) == 2:
            kps = np.zeros([2, 2], np.int32)
            r   = find_line_all_right(lines_segment[0], lines_segment[1])
            kps[0], kps[1] = r[0], r[1]
        else:
            print('error keypoints--------------')
            kps = None
        return kps

    def sort_and_surround_lines(self, finger_lines, num_points=80):
        min_ys_and_lines = [(np.min(line[:, 1]), line) for line in finger_lines]
        sorted_by_y      = sorted(min_ys_and_lines, key=lambda item: item[0])
        sorted_lines     = [item[1] for item in sorted_by_y]
        half_num         = num_points // 2
        new_sets = []
        for line in sorted_lines:
            idx_max  = np.argmax(line[:, 0])
            start    = max(0, idx_max - half_num)
            end      = min(line.shape[0], idx_max + half_num + 1)
            new_sets.append(line[start:end])
        return new_sets

    def find_concave_finger(self, hull_contour_coord_small,
                            hull_contour_img_small, palm_len,
                            hull_coord_right, palm_contour):
        mask = np.array([
            cv2.pointPolygonTest(
                hull_contour_coord_small[:, 0, :].astype(np.float32),
                (float(pt[0]), float(pt[1])), measureDist=True) > 0.5
            for pt in palm_contour])
        transitions = np.where(mask[:-1] != mask[1:])[0] + 1
        if mask[0]:
            transitions = np.insert(transitions, 0, 0)
        if mask[-1]:
            transitions = np.append(transitions, len(palm_contour))

        rough = [palm_contour[s:e]
                 for s, e in zip(transitions[::2], transitions[1::2])
                 if e - s > 5]

        if len(rough) <= 4:
            return rough

        lengths      = np.array([r.shape[0] for r in rough])
        sorted_idx   = np.argsort(lengths)[::-1]
        finger_lines = []
        for idx in sorted_idx:
            if len(finger_lines) >= 4:
                break
            line = rough[idx]
            if line.shape[0] <= 20:
                continue
            right_pt = line[np.argmax(line[:, 0])]
            cp, cd   = self.find_closest_white_point(hull_contour_img_small, right_pt)
            right_x  = np.max(line[:, 0])
            if cd < 20 / self.ratio:
                if cd / line.shape[0] > 1 / 3:
                    finger_lines.append(line)
            else:
                if hull_coord_right - right_x > palm_len / 4:
                    finger_lines.append(line)
        return finger_lines

    def find_inner_contour(self, finger_edges):
        contours_final, hierarchy = cv2.findContours(
            finger_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_sorted = sorted(contours_final, key=cv2.contourArea, reverse=True)
        top_contours    = contours_sorted[:2]
        selected        = top_contours[1] if len(top_contours) > 1 else top_contours[0]
        palm_contour_coord = selected[:, 0, :]
        palm_contour_img   = self.fill_contour(self.ori_img, selected)
        return palm_contour_img, palm_contour_coord

    def _select_rough_palm(self, edges_hull_img):
        num_labels, labels, stats = cv2.connectedComponentsWithStats(
            edges_hull_img, connectivity=8)[:3]
        sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        return labels, sorted_indices

    def _detect_rough_edges(self, img, threshold_edge=10):
        lap   = cv2.Laplacian(img, cv2.CV_64F, ksize=7).clip(min=0)
        edges = cv2.normalize(lap, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, edges = cv2.threshold(edges, threshold_edge, 255, cv2.THRESH_BINARY)
        edges    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                    np.ones((3, 3), np.uint8))
        return edges

    def compute_rough_orientation(self, labels, sorted_indices,
                                  hull_filled_img, gabors):
        edges_img = np.zeros([self.w_rotate, self.h_rotate])
        for index in sorted_indices[1:4]:
            if np.sum(labels == index + 1) > (20 / self.ratio):
                edges_img[labels == index + 1] = 255

        points       = np.argwhere(edges_img > 0)
        centroid_x   = np.mean(points, axis=0)[1]
        centroid_y   = np.mean(points, axis=0)[0]
        finger_hull  = cv2.convexHull(points[:, [1, 0]])
        finger_hull_filled = cv2.drawContours(
            np.zeros_like(edges_img, dtype=np.uint8),
            [finger_hull], -1, 255, thickness=cv2.FILLED)
        pts_remained = np.argwhere(
            (finger_hull_filled & hull_filled_img) ^ hull_filled_img)
        center       = np.mean(pts_remained, axis=0)

        vect         = np.array([centroid_x - center[1], centroid_y - center[0]])
        angle_rough  = np.degrees(np.arctan2(vect[1], vect[0]))
        cx           = edges_img.shape[1] // 2
        cy           = edges_img.shape[0] // 2
        Mrough       = cv2.getRotationMatrix2D((cx, cy), angle_rough, 1)
        edges_rough  = cv2.warpAffine(edges_img, Mrough,
                                      (edges_img.shape[1], edges_img.shape[0]))

        results = np.zeros((12, edges_rough.shape[0], edges_rough.shape[1]))
        for jj, gabor in enumerate(gabors):
            results[jj] = cv2.filter2D(edges_rough.astype(np.float32), -1, gabor)

        valid_ind   = np.argmax(results, axis=0)[edges_rough == 255]
        unique, counts = np.unique(valid_ind, return_counts=True)
        most_common = unique[np.argmax(counts)]
        angle_fine  = 90 - 180 / 12 * most_common
        return angle_rough - angle_fine + 180

    def _get_gabor_filters(self):
        return GaborFilter(15, 12, sigma=3, lambd=10, gamma=0.2)

    def rotate_image(self, image, angle_degrees):
        cx = image.shape[1] // 2
        cy = image.shape[0] // 2
        M  = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def rotate_keypoints(self, image, angle_degrees, kk_data):
        cx = image.shape[1] // 2
        cy = image.shape[0] // 2
        M  = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
        pts = np.hstack([kk_data, np.ones((kk_data.shape[0], 1))])
        return np.round(np.dot(pts, M.T)[:, :2]).astype(int)

    def process_palm_contour_rough(self, sorted_edges_indices, labels,
                                   len_palm, hull_contour_small,
                                   concave_contour_img, hull_contour):
        concave_hull  = concave_contour_img | hull_contour_small
        finger_edges  = concave_contour_img | hull_contour_small
        th_removed    = 6 / self.ratio
        edges_counts  = 0
        th_edge_count = 3
        hull_points   = np.argwhere(hull_contour == 255)

        for ind in sorted_edges_indices:
            if edges_counts > th_edge_count:
                break
            gray_edge = np.zeros((self.w, self.h))
            gray_edge[labels == ind + 1] = 255
            gray_edge_skele = self.get_skeleton(gray_edge)
            if gray_edge_skele.sum() / 255 < 25:
                continue
            results_graph = self.find_closest_path_graph(gray_edge_skele)
            if len(results_graph[1]) == 0:
                continue
            flag_valley, rightnode_index = self.judge_valley(results_graph[1])

            if not flag_valley:
                right_node   = results_graph[1][-1]
                left_node    = results_graph[1][0]
                left_hull    = hull_points[np.argwhere(hull_points[:, 0] == left_node[0])]
                left_dest    = left_node[1] - np.min(left_hull[:, 0, 1])
                if left_dest > (len_palm / 6):
                    continue
                path_valid   = self.find_farthest_point(results_graph[1], th_removed)
                gray_edge_skele = np.zeros_like(self.cut_norm_img)
                cv2.polylines(gray_edge_skele, [path_valid],
                              isClosed=False, color=(255, 255, 255), thickness=1)
                cp, _ = self.find_closest_white_point(concave_hull, path_valid[0])
                cv2.line(gray_edge_skele, cp, tuple(path_valid[0]), 255, thickness=1)
            else:
                dest_node = results_graph[1][-1]
                src_node  = results_graph[1][0]
                dest_hull = hull_points[np.argwhere(hull_points[:, 0] == dest_node[0])]
                dist_dest = dest_node[1] - np.min(dest_hull[:, 0, 1])
                src_hull  = hull_points[np.argwhere(hull_points[:, 0] == src_node[0])]
                src_dest  = src_node[1] - np.min(src_hull[:, 0, 1])
                if min(dist_dest, src_dest) > (len_palm / 5):
                    continue
                gray_edge_skele = results_graph[0].astype(np.uint8)
                cp_dest, _ = self.find_closest_white_point(concave_hull, dest_node[::-1])
                cv2.line(gray_edge_skele, cp_dest, dest_node[::-1], 255, thickness=1)
                cp_src, _  = self.find_closest_white_point(concave_hull, src_node[::-1])
                cv2.line(gray_edge_skele, cp_src, src_node[::-1], 255, thickness=1)

            edges_counts  += 1
            finger_edges   = finger_edges | gray_edge_skele

        return finger_edges.astype(np.uint8)


# ──────────────────────────────────────────────────────────────
#  Core extraction logic with fallback cascade
# ──────────────────────────────────────────────────────────────

def _build_merged_cfg(primary_cfg, override_dict, img_shape):
    """Merge override keys into a copy of primary_cfg and resolve kernel."""
    merged = dict(primary_cfg)
    merged.update(override_dict)
    merged["KERNEL_SIZE"] = _resolve_kernel(merged, img_shape)
    return merged


def try_extract(gray_img, color_img, primary_cfg, fallback_cfgs, debug=False):
    """
    gray_img  — used for all processing (segmentation, keypoint detection)
    color_img — used only for the final warp so the saved ROI is in colour
    Returns (roi_bgr, config_label) or (None, None).
    """
    all_attempts = [("primary", primary_cfg)] + \
                   [(f"fallback_{i+1}", fb) for i, fb in enumerate(fallback_cfgs)]

    for cfg_label, override in all_attempts:
        cfg = _build_merged_cfg(primary_cfg, override, gray_img.shape)
        try:
            get_roi = GetROI(gray_img, cfg)
            get_roi.run_rotate()
            get_roi.norm_img = get_roi.rotate_image(
                get_roi.ori_img, get_roi.rotation_angle)
            h, w = get_roi.norm_img.shape[:2]
            get_roi.cut_norm_img = get_roi.norm_img[0:h, 0:int(w / 4 * 3)]
            success = get_roi.run_localization()
        except Exception as e:
            if debug:
                print(f"      [{cfg_label}] exception: {e}")
            continue

        if success:
            kk  = np.array(get_roi.keypoints_localization)
            kkt = get_roi.rotate_keypoints(get_roi.norm_img,
                                           -get_roi.rotation_angle, kk)
            # ── warp the COLOUR image using the keypoints found on gray ──
            roi, _, _ = extract_roi(kkt[0], kkt[1], color_img,
                                    color=[0, 0, 255], thickness=2)
            return roi, cfg_label

        if debug:
            print(f"      [{cfg_label}] localization returned False")

    return None, None

# ──────────────────────────────────────────────────────────────
#  Debug: save intermediate stages for a single image
# ──────────────────────────────────────────────────────────────

def debug_single_image(img_path, dataset_preset="MPDv2"):
    """
    Saves intermediate processing images next to the source image.
    Useful for tuning thresholds when images are failing.
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Cannot read image: {img_path}")
        return

    cfg = dict(DATASET_CONFIGS[dataset_preset])
    cfg["KERNEL_SIZE"] = _resolve_kernel(cfg, gray.shape)
    out  = os.path.dirname(img_path)

    # 0 — raw
    cv2.imwrite(os.path.join(out, "dbg_0_raw.jpg"), gray)

    # 1 — blurred
    blurred = (cv2.GaussianBlur(gray, (5, 5), cfg["BLUR_SIGMA"])
               if cfg["BLUR"] else gray.copy())
    cv2.imwrite(os.path.join(out, "dbg_1_blurred.jpg"), blurred)

    # 2 — Otsu threshold value (info only)
    otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3 — threshold
    if cfg["BI_THRESHOLD_SEG"]:
        _, thresh = cv2.threshold(blurred, cfg["THRESHOLD_SEG"],
                                  255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(out, "dbg_2_threshold.jpg"), thresh)

    # 4 — largest connected component
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    comp = np.zeros_like(gray)
    if len(stats) > 1:
        lbl = np.argmax(stats[1:, -1]) + 1
        comp[labels == lbl] = 255
    cv2.imwrite(os.path.join(out, "dbg_3_largest_component.jpg"), comp)

    # 5 — eroded
    k      = cfg["KERNEL_SIZE"]
    eroded = cv2.erode(comp, np.ones(k, np.uint8))
    cv2.imwrite(os.path.join(out, "dbg_4_eroded.jpg"), eroded)

    # 6 — edges (Laplacian)
    lap   = cv2.Laplacian(blurred & comp, cv2.CV_64F, ksize=7).clip(min=0)
    edges = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, edges_bin = cv2.threshold(edges, cfg["THRESHOLD_EDGE"],
                                 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(out, "dbg_5_edges.jpg"), edges_bin)

    print(f"\nDebug images saved to: {out}")
    print(f"  Image size         : {gray.shape[1]}×{gray.shape[0]}")
    print(f"  Otsu threshold     : {otsu_val:.1f}")
    print(f"  Kernel size used   : {k}")
    print(f"  Config preset      : {dataset_preset}")


# ──────────────────────────────────────────────────────────────
#  Main extraction runner
# ──────────────────────────────────────────────────────────────

def run_extraction(dir_source, dir_output, dataset_preset="MPDv2", debug=False):
    os.makedirs(dir_output, exist_ok=True)

    if dataset_preset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset preset '{dataset_preset}'. "
                         f"Choose from: {list(DATASET_CONFIGS.keys())}")

    primary_cfg = DATASET_CONFIGS[dataset_preset]

    # collect all images recursively
    all_images = []
    for root, _, files in os.walk(dir_source):
        for filename in sorted(files):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append((root, filename))

    print(f"\n{'='*55}")
    print(f"  Source  : {dir_source}")
    print(f"  Output  : {dir_output}")
    print(f"  Preset  : {dataset_preset}")
    print(f"  Images  : {len(all_images)}")
    print(f"  Debug   : {debug}")
    print(f"{'='*55}\n")

    num_success  = 0
    num_failed   = 0
    usage_counts = {}

    for idx, (root, filename) in enumerate(all_images):
        img_path = os.path.join(root, filename)
        rel_path = os.path.relpath(root, dir_source)
        out_dir  = os.path.join(dir_output, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        color_img = cv2.imread(img_path, cv2.IMREAD_COLOR)   # BGR, 3-channel
        if color_img is None:
            print(f"[{idx:05d}] SKIP  (unreadable) : {filename}")
            num_failed += 1
            continue
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # gray for processing

        if debug:
            print(f"[{idx:05d}] Processing : {filename}  "
                  f"({gray_img.shape[1]}×{gray_img.shape[0]})")

        roi, used = try_extract(gray_img, color_img, primary_cfg,
                                FALLBACK_CONFIGS, debug=debug)

        if roi is not None:
            roi_rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)   # BGR → RGB
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, roi_rgb)
            usage_counts[used] = usage_counts.get(used, 0) + 1
            if debug or used != "primary":
                print(f"[{idx:05d}] OK  [{used}]  →  {out_path}")
            num_success += 1
        else:
            print(f"[{idx:05d}] FAILED all configs : {filename}")
            num_failed += 1

    # ── summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Finished.")
    print(f"  Success : {num_success}")
    print(f"  Failed  : {num_failed}")
    print(f"  Total   : {num_success + num_failed}")
    print(f"\n  Config usage breakdown:")
    for label, count in sorted(usage_counts.items()):
        pct = 100.0 * count / max(num_success, 1)
        print(f"    {label:15s} : {count:5d}  ({pct:.1f}%)")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── optional: inspect a single tricky image first ─────────
    if DEBUG_SINGLE_IMAGE is not None:
        debug_single_image(DEBUG_SINGLE_IMAGE, dataset_preset=DATASET_PRESET)

    # ── run full extraction ───────────────────────────────────
    run_extraction(
        dir_source     = DIR_SOURCE,
        dir_output     = DIR_OUTPUT,
        dataset_preset = DATASET_PRESET,
        debug          = DEBUG,
    )
