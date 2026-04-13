import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
from shapely.geometry import Polygon


# ─────────────────────────────────────────────
# GaborFilter.py (original, unchanged)
# ─────────────────────────────────────────────

def GaborFilter_cc(len_filter, sigma=4.6, delta=2.6, num_direction=6):
    assert num_direction % 2 == 0, 'num_direction should be an even number!'
    half_len = int(len_filter / 2)

    Filter = np.zeros((num_direction, len_filter, len_filter))
    for a in range(num_direction):
        theta = np.pi / 2 - np.pi * a / num_direction
        kappa = np.sqrt(2 * np.log(2)) * (delta + 1) / (delta - 1)
        w = kappa / sigma
        fFactor1 = -w / (np.sqrt(2 * np.pi) * kappa)
        fFactor2 = -(w * w) / (8 * kappa * kappa)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for c in range(len_filter):
            x = c - half_len
            for r in range(len_filter):
                y = r - half_len
                x1 = x * cos_theta + y * sin_theta
                y1 = y * cos_theta - x * sin_theta
                f_comp = fFactor1 * np.exp(fFactor2 * (4 * x1 * x1 + y1 * y1))
                Filter[a, r, c] = f_comp * np.cos(w * x1)

        Filter[a, :, :] -= Filter[a, :, :].mean()

    return Filter


def GaborFilter(ksize, num_direction, sigma, lambd, gamma):
    assert num_direction % 2 == 0, 'num_direction should be an even number!'
    half_size = int(ksize/2)
    sigma2 = 2*sigma**2

    Filter = np.zeros((num_direction, ksize, ksize))
    l_min = -half_size
    l_max = half_size
    x, y = np.meshgrid(range(l_min, l_max + 1), range(l_min, l_max + 1))
    for a in range(num_direction):
        theta = np.pi * a / num_direction
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_theta = x * cos_theta + y * sin_theta
        y_theta = y * cos_theta - x * sin_theta
        term1 = np.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / sigma2)
        term2 = np.cos(2 * np.pi * x_theta / lambd)
        Filter[a] = term1 * term2
        Filter[a] -= Filter[a].mean()

    return Filter


def GaborArray(sigma=4.85, wavelength=14.1, ratio=1.92):
    halfLength = 17

    xmax = halfLength
    xmin = -halfLength
    ymax = halfLength
    ymin = -halfLength
    [x, y] = np.meshgrid(range(xmin, xmax+1), range(ymin, ymax+1))

    mask = np.ones((35, 35))
    for row in range(1,36):
        for col in range(1,36):
            if (row - 18)**2 + (col - 18)**2 > 289:
                mask[row-1, col-1] = 0

    gb_r = np.zeros((6, 35, 35))
    for oriIndex in range(1, 7):
        theta = np.pi / 6 * (oriIndex - 1)

        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5*(x_theta**2/sigma**2 + y_theta**2/(ratio*sigma)**2)) * np.cos(2*np.pi/wavelength*x_theta)

        total = (gb * mask).sum()
        meanInner = total / (mask).sum()

        gb = gb - meanInner.mean()
        gb = gb * mask
        gb_r[oriIndex-1] = gb

    return gb_r


# ─────────────────────────────────────────────
# PalmBasic.py (original, unchanged)
# ─────────────────────────────────────────────

class PalmBasic:

    def __init__(self):
        pass

    def gaussian_blur(self, img, kernel_size=(5, 5), sigma=2, blur=True):
        if blur:
            return cv2.GaussianBlur(img, kernel_size, sigma)
        else:
            return img

    def threshold_image(self, img, threshold_val=20, bi_threshold=True):
        if bi_threshold:
            _, thresholded = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def compute_edist(self, point1, point2):
        return np.sqrt(np.sum((point1-point2)**2))

    def find_largest_component(self, img):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        if len(stats) <= 1:  # only background, no foreground components
            return np.zeros_like(img), None, np.zeros_like(img)
        largest_component_label = np.argmax(stats[1:, -1]) + 1
        max_comp_img = np.zeros_like(img)
        max_comp_img[labels == largest_component_label] = 255
        max_comp_contour_coord = self.find_contour(max_comp_img)
        max_comp_contour_img = cv2.drawContours(np.zeros_like(img, dtype=np.uint8), [max_comp_contour_coord], -1, (255), thickness=1)
        return max_comp_img, max_comp_contour_coord, max_comp_contour_img

    def find_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        return cnt

    def fill_contour(self, img, contour, filled=False):
        img_contour = np.zeros_like(img, dtype=np.uint8)
        thickness = cv2.FILLED if filled else 1
        cv2.drawContours(img_contour, [contour], -1, (255), thickness=thickness)
        return img_contour

    def erode_binary_image(self, img, kernel_size=(7, 7), iterations=1, th_binary=180):
        kernel = np.ones(kernel_size, np.uint8)
        eroded = cv2.erode(img, kernel, iterations=iterations)
        _, binaried = cv2.threshold(eroded, th_binary, 255, cv2.THRESH_BINARY)
        return binaried

    def resize_image(self, img, ratio):
        img_resized = cv2.resize(img, (img.shape[1] // ratio, img.shape[0] // ratio))
        return img_resized

    def hull_image(self, img, img_contour):
        epsilon = 0.01*cv2.arcLength(img_contour, True)
        approx = cv2.approxPolyDP(img_contour, epsilon, True)
        hull_coord = cv2.convexHull(approx, clockwise=False, returnPoints=True)
        hull_img = cv2.drawContours(np.zeros_like(img, dtype=np.uint8), [hull_coord], -1, 255, thickness=cv2.FILLED)
        return hull_coord, hull_img

    def get_skeleton(self, img):
        img = img == 255
        skeleton = skeletonize(img)*255
        return skeleton

    def euclidean_distance(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        difference = point1 - point2
        squared_difference = np.square(difference)
        sum_of_squared_difference = np.sum(squared_difference)
        distance = np.sqrt(sum_of_squared_difference)
        return distance

    def find_closest_path_graph(self, matrix):
        cleaned_matrix = np.zeros_like(matrix)
        graph = self.build_undirected_graph(matrix)
        nodes_with_degree_1 = [node for node in graph.nodes() if graph.degree[node] == 1]

        leftmost_node = None
        min_x = float('inf')
        for node in nodes_with_degree_1:
            x = node[1]
            if x < min_x:
                min_x = x
                leftmost_node = node
        source_node = leftmost_node

        longest_shortest_path = []
        for node in nodes_with_degree_1:
            if node != source_node:
                shortest_path = nx.shortest_path(graph, source=source_node, target=node)
                if len(shortest_path) > len(longest_shortest_path):
                    longest_shortest_path = shortest_path

        for point in longest_shortest_path:
            cleaned_matrix[point[0], point[1]] = 255

        return [cleaned_matrix, longest_shortest_path]

    def build_undirected_graph(self, matrix):
        non_zero_elements = np.argwhere(matrix == 255)
        min_row, min_col = np.min(non_zero_elements, axis=0)
        max_row, max_col = np.max(non_zero_elements, axis=0)

        graph = nx.Graph()
        directions = [(-1, 0), (-1, -1), (0, -1), (1, -1)]

        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if matrix[i, j] == 255:
                    current_node = (i, j)
                    graph.add_node(current_node)
                    for direction in directions:
                        neighbor = (i + direction[0], j + direction[1])
                        if (min_row <= neighbor[0] <= max_row) and (min_col <= neighbor[1] <= max_col) and matrix[neighbor[0], neighbor[1]] == 255:
                            graph.add_edge(current_node, neighbor)

        return graph

    def judge_valley(self, path):
        rightmost_node = max(path, key=lambda node: node[1])
        rightmost_index = path.index(rightmost_node)
        nodes_between1 = path[rightmost_index + 1:]
        nodes_between2 = path[:rightmost_index + 1]
        nodes_between = min(len(nodes_between1), len(nodes_between2))
        flag_valley = False
        if nodes_between > (len(path)/3):
            flag_valley = True
        return flag_valley, rightmost_index

    def find_farthest_point(self, path, th):
        len_path = len(path)
        path = np.array(path)
        path = path[:, [1, 0]]

        left_ind = int(len_path/5*2)
        left_node = path[left_ind]
        right_ind = len_path-1
        right_node = path[right_ind]
        fix_vect = left_node - right_node

        estimated_nodes = path[left_ind+1:-1]
        estimated_vects = estimated_nodes - right_node
        angle_degrees = self.compute_clockwise_angle_out(estimated_vects, fix_vect)
        flag_inter = np.abs(np.abs(np.mean(angle_degrees))-np.mean(np.abs(angle_degrees))) > 1.5

        while flag_inter and left_ind < (len_path-1) and right_ind > 0:
            if left_ind < int(len_path/3):
                left_ind += 2
                left_node = path[left_ind]
            elif right_ind > int((len_path/5*4)):
                right_ind -= 2
                right_node = path[right_ind]
            else:
                break

            fix_vect = left_node - right_node
            estimated_nodes = path[left_ind+1:right_ind]
            estimated_vects = estimated_nodes - right_node
            angle_degrees = self.compute_clockwise_angle_out(estimated_vects, fix_vect)
            flag_inter = np.abs(np.abs(np.mean(angle_degrees))-np.mean(np.abs(angle_degrees))) > 1.5

        path_valid = path[left_ind:right_ind+1]
        distances = self.compute_perpendicular_dist(path_valid)
        max_idx = np.argmax(distances)
        if distances[max_idx] > th:
            path_valid = path_valid[:max_idx+4]

        return path_valid

    def compute_perpendicular_dist(self, path):
        x1, y1 = path[0]
        x2, y2 = path[-1]
        path_valid = path[1:-1]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        distances = abs(A * path_valid[:, 0] + B * path_valid[:, 1] + C) / (np.sqrt(A**2 + B**2)+0.00000000001)
        return distances

    def compute_clockwise_angle_out(self, fix_vect, moved_vect):
        cross_product = np.cross(moved_vect, fix_vect)
        dot_product = np.dot(moved_vect, fix_vect.T)
        angle_relative = np.arctan2(cross_product, dot_product)
        angle_degrees = np.degrees(angle_relative)
        return angle_degrees

    def find_closest_white_point(self, image, point):
        non_zero_points = cv2.findNonZero(image)
        distances = np.linalg.norm(non_zero_points - point, axis=2)
        min_index = np.argmin(distances)
        return tuple(non_zero_points[min_index][0]), distances[min_index][0]


# ─────────────────────────────────────────────
# run_roi.py (original, unchanged)
# ─────────────────────────────────────────────

# Setting for CUHK
THRESHOLD_SEG = 90
THRESHOLD_EDGE = 10
BI_THRESHOLD_SEG = True
BI_THRESHOLD_ROI_CHECK = True
BLUR = False
BLUR_SIGMA = 0.05
KERNEL_SIZE = (45, 45)

BLUR_ROTATE = True
BLUR_SIGMA_ROTATE = 0.05
BI_THRESHOLD_SEG_ROTATE = False
THRESHOLD_SEG_ROTATE = 90

pad = 0

def extract_roi(p1, p2, img, color, thickness):
    d = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / (d+0.000001)
    normal = np.array([direction[1], -direction[0]])

    s = int(d/6*7)
    half_extend = int(d/12)
    distance_to_square = int(d/6*1)

    C = (p1 + (distance_to_square+s) * normal - half_extend * direction).astype(np.int32)
    D = (C + s * direction).astype(np.int32)
    E = (D - s * normal).astype(np.int32)
    F = (C - s * normal).astype(np.int32)

    img_padded = cv2.copyMakeBorder(img, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    C += pad
    D += pad
    E += pad
    F += pad

    pts = np.array([C, D, E, F], dtype="float32")
    width = max(np.linalg.norm(E-D), np.linalg.norm(F-C))
    height = max(np.linalg.norm(D-C), np.linalg.norm(E-F))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_padded, M, (int(width), int(height)))
    cv2.polylines(img_padded, [np.array([C, D, E, F])], isClosed=True, color=color, thickness=thickness)

    corner_points = [(C[0], C[1]), (D[0], D[1]), (E[0], E[1]), (F[0], F[1])]

    return warped, corner_points, img_padded


class GetROI(PalmBasic):

    def __init__(self, img, ratio_rotate=2.0, ratio=1.0):
        self.ratio_rotate = ratio_rotate
        self.ratio = ratio
        self.ori_img = img
        self.w, self.h = img.shape
        self.w_rotate, self.h_rotate = int(self.w/ratio_rotate), int(self.h/ratio_rotate)

    def run_rotate(self):
        img = self.resize_image(self.ori_img, self.ratio_rotate)
        gabors = self._get_gabor_filters()

        img_blurred = self.gaussian_blur(img, sigma=BLUR_SIGMA_ROTATE, blur=BLUR_ROTATE)
        rough_binary = self.threshold_image(img_blurred, threshold_val=THRESHOLD_SEG_ROTATE, bi_threshold=BI_THRESHOLD_SEG_ROTATE)
        max_comp_img, max_comp_contour_coord, max_comp_contour_img = self.find_largest_component(rough_binary)
        hull_coord, hull_img = self.hull_image(img, max_comp_contour_coord)
        rough_edges_img = self._detect_rough_edges(img_blurred & max_comp_img)
        edges_hull_img = rough_edges_img & hull_img
        palm_edges_labels, palm_edges_sorted_indices = self._select_rough_palm(edges_hull_img)

        self.rotation_angle = self.compute_rough_orientation(palm_edges_labels, palm_edges_sorted_indices, hull_img, gabors)
        self.norm_img = self.rotate_image(self.ori_img, self.rotation_angle)
        height, width = self.norm_img.shape[:2]
        self.cut_norm_img = self.norm_img[0:height, 0:int(width/4*3)]

    def run_localization(self):
        img = self.cut_norm_img
        self.w, self.h = img.shape
        img_blurred = self.gaussian_blur(img, sigma=BLUR_SIGMA, blur=BLUR)
        rough_binary = self.threshold_image(img_blurred, threshold_val=THRESHOLD_SEG, bi_threshold=BI_THRESHOLD_SEG)
        max_comp_img, max_comp_contour_coord, max_comp_contour_img = self.find_largest_component(rough_binary)

        if max_comp_contour_coord is None:
            return False

        max_comp_img_small = self.erode_binary_image(max_comp_img, kernel_size=KERNEL_SIZE)
        if cv2.countNonZero(max_comp_img_small) == 0:
            max_comp_img_small = self.erode_binary_image(max_comp_img, kernel_size=(15, 15))
        if cv2.countNonZero(max_comp_img_small) == 0:
            return False
        max_comp_contour_coord_small = self.find_contour(max_comp_img_small)
        max_comp_contour_img_small = self.fill_contour(img, max_comp_contour_coord_small)

        hull_coord, hull_img = self.hull_image(img, max_comp_contour_coord)
        rough_edges_img = self._detect_rough_edges(img_blurred & max_comp_img)

        hull_img_small = self.erode_binary_image(hull_img, kernel_size=KERNEL_SIZE)
        if cv2.countNonZero(hull_img_small) == 0:
            hull_img_small = self.erode_binary_image(hull_img, kernel_size=(15, 15))
        if cv2.countNonZero(hull_img_small) == 0:
            return False

        hull_contour_coord_small = self.find_contour(hull_img_small)
        hull_contour_img_small = self.fill_contour(img, hull_contour_coord_small)
        concave_contour_img = max_comp_contour_img & hull_img_small

        edges_hull_img = rough_edges_img & hull_img_small
        self.edges_hull_img = edges_hull_img

        palm_edges_labels, palm_edges_sorted_indices = self._select_rough_palm(edges_hull_img)

        palm_len = np.max(hull_coord[:,0,:][:,0]) - np.min(hull_coord[:,0,:][:,0])

        self.finger_edges_img = self.process_palm_contour_rough(palm_edges_sorted_indices, palm_edges_labels, palm_len, hull_contour_img_small, concave_contour_img, max_comp_contour_img)
        self.palm_contour_img, self.palm_contour_coord = self.find_inner_contour(self.finger_edges_img)
        self.finger_lines = self.find_concave_finger(hull_contour_coord_small, hull_contour_img_small, palm_len, np.max(hull_coord[:,0,:][:,0]), self.palm_contour_coord)
        self.finger_lines_sorted = self.sort_and_surround_lines(self.finger_lines)
        self.keypoints = self.detect_keypoints(self.finger_lines_sorted)
        self.show_img = cv2.cvtColor(np.copy(self.norm_img), cv2.COLOR_GRAY2BGR)

        if self.keypoints is None:
            return False
        else:
            for point in self.keypoints:
                x, y = int(point[0]), int(point[1])
            sorted_indices = np.argsort(self.keypoints[:, 1])
            self.keypoints = self.keypoints[sorted_indices]
            self.localize_keypoints(self.keypoints)
            return True

    def localize_keypoints(self, keypoints):
        if len(keypoints) == 4:
            dist1 = self.compute_edist(keypoints[0], keypoints[2]) + self.compute_edist(keypoints[0], keypoints[1])
            dist2 = self.compute_edist(keypoints[1], keypoints[3]) + self.compute_edist(keypoints[2], keypoints[3])
            if dist2 > dist1:
                keypoint1 = keypoints[0]
                keypoint2 = keypoints[2]
            else:
                keypoint1 = keypoints[1]
                keypoint2 = keypoints[3]
        elif len(keypoints) == 2:
            keypoint1 = keypoints[0]
            keypoint2 = keypoints[1]
        self.keypoints_localization = [keypoint1, keypoint2]

    def detect_keypoints(self, lines_segment):
        def is_on_right(p1, p2, p):
            return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) < 0

        def find_line_with_all_points_on_right(set1, set2):
            for p1 in set1:
                for p2 in set2:
                    all_on_right = True
                    for p in set1:
                        if p is not p1 and is_on_right(p1, p2, p):
                            all_on_right = False
                            break
                    if all_on_right:
                        for p in set2:
                            if p is not p2 and is_on_right(p1, p2, p):
                                all_on_right = False
                                break
                    if all_on_right:
                        return (p1, p2)
            return set1[np.argmax(set1[:, 0])], set2[np.argmax(set2[:, 0])]

        if len(lines_segment) == 4:
            keypoints = np.zeros([4, 2], np.int32)
            points1_result = find_line_with_all_points_on_right(lines_segment[0], lines_segment[2])
            points2_result = find_line_with_all_points_on_right(lines_segment[1], lines_segment[3])
            keypoints[0] = points1_result[0]
            keypoints[2] = points1_result[1]
            keypoints[1] = points2_result[0]
            keypoints[3] = points2_result[1]
        elif len(lines_segment) == 3:
            keypoints = np.zeros([2, 2], np.int32)
            points_result = find_line_with_all_points_on_right(lines_segment[0], lines_segment[2])
            keypoints[0] = points_result[0]
            keypoints[1] = points_result[1]
        elif len(lines_segment) == 2:
            keypoints = np.zeros([2, 2], np.int32)
            points_result = find_line_with_all_points_on_right(lines_segment[0], lines_segment[1])
            keypoints[0] = points_result[0]
            keypoints[1] = points_result[1]
        else:
            keypoints = None
            print('error keypoints--------------')

        return keypoints

    def sort_and_surround_lines(self, finger_lines, num_points=80):
        min_ys_and_lines = [(np.min(line[:, 1]), line) for line in finger_lines]
        sorted_by_y = sorted(min_ys_and_lines, key=lambda item: item[0])
        sorted_lines = [item[1] for item in sorted_by_y]

        new_point_sets = []
        half_num = num_points // 2
        for line in sorted_lines:
            idx_max_x = np.argmax(line[:, 0])
            start_idx = max(0, idx_max_x - half_num)
            end_idx = min(line.shape[0], idx_max_x + half_num + 1)
            extracted_points = line[start_idx:end_idx]
            new_point_sets.append(extracted_points)

        return new_point_sets

    def find_concave_finger(self, hull_contour_coord_small, hull_contour_img_small, palm_len, hull_coord_right, palm_contour):
        finger_lines = []
        mask = np.array([cv2.pointPolygonTest(hull_contour_coord_small[:,0,:].astype(np.float32), (float(pt[0]), float(pt[1])), measureDist=True) > 0.5 for pt in palm_contour])
        transitions = np.where(mask[:-1] != mask[1:])[0] + 1

        if mask[0]:
            transitions = np.insert(transitions, 0, 0)
        if mask[-1]:
            transitions = np.append(transitions, len(palm_contour))

        finger_lines_rough = [palm_contour[start:end] for start, end in zip(transitions[::2], transitions[1::2]) if end - start > 5]

        finger_lines = []
        if len(finger_lines_rough) <= 4:
            finger_lines = finger_lines_rough
        else:
            lines_long = np.zeros(len(finger_lines_rough))
            for ind, finger_line in enumerate(finger_lines_rough):
                lines_long[ind] = finger_line.shape[0]

            sorted_indices = np.argsort(lines_long)[::-1]
            for index in sorted_indices:
                if len(finger_lines) >= 4:
                    break
                finger_line = finger_lines_rough[index]
                if finger_line.shape[0] > 20:
                    right_point = finger_line[np.argmax(finger_line[:,0])]
                    closet_point, closet_dist = self.find_closest_white_point(hull_contour_img_small, right_point)
                    right_most_finger_x = np.max(finger_lines_rough[index][:, 0])
                    if closet_dist < 20/self.ratio:
                        if closet_dist/finger_line.shape[0] > 1/3:
                            finger_lines.append(finger_lines_rough[index])
                    else:
                        if hull_coord_right - right_most_finger_x > palm_len / 4:
                            finger_lines.append(finger_lines_rough[index])

        return finger_lines

    def find_inner_contour(self, finger_edges):
        contours_final, hierarchy = cv2.findContours(finger_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_sorted = sorted(contours_final, key=cv2.contourArea, reverse=True)
        top_contours = contours_sorted[:2]

        selected_contour = None
        for i, contour in enumerate(top_contours):
            parent_contour_exists = False
            if any((np.array_equal(contour, target) for target in contours_final)):
                hierarchy_idx = hierarchy[0][i]
                if hierarchy_idx[3] != -1 and contours_final[hierarchy_idx[3]] in top_contours:
                    parent_contour_exists = True
            if parent_contour_exists:
                selected_contour = contour
            else:
                selected_contour = top_contours[1]

        palm_contour_coord = selected_contour[:, 0, :]
        palm_contour_img = self.fill_contour(self.ori_img, selected_contour)
        return palm_contour_img, palm_contour_coord

    def _select_rough_palm(self, edges_hull_img):
        num_labels, labels, stats = cv2.connectedComponentsWithStats(edges_hull_img, connectivity=8)[:3]
        sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        return labels, sorted_indices

    def _detect_rough_edges(self, img):
        edges = cv2.normalize(cv2.Laplacian(img, cv2.CV_64F, ksize=7).clip(min=0), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, edges = cv2.threshold(edges, THRESHOLD_EDGE, 255, cv2.THRESH_BINARY)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return edges

    def compute_rough_orientation(self, labels, sorted_indices, hull_filled_img, gabors):
        edges_img = np.zeros([self.w_rotate, self.h_rotate])

        for index in sorted_indices[1:4]:
            if np.sum(labels == index + 1) > (20 / self.ratio):
                edges_img[labels == index + 1] = 255

        points = np.argwhere(edges_img > 0)
        centroid_x, centroid_y = np.mean(points, axis=0)[1], np.mean(points, axis=0)[0]

        finger_hull = cv2.convexHull(points[:, [1, 0]])
        finger_hull_filled_img = cv2.drawContours(np.zeros_like(edges_img, dtype=np.uint8), [finger_hull], -1, 255, thickness=cv2.FILLED)
        points_remained = np.argwhere((finger_hull_filled_img & hull_filled_img) ^ hull_filled_img)
        center = np.mean(points_remained, axis=0)

        rough_orien_vect = np.array([centroid_x - center[1], centroid_y - center[0]])
        angle_degrees_rough = np.degrees(np.arctan2(rough_orien_vect[1], rough_orien_vect[0]))
        center_rotate = (edges_img.shape[1] // 2, edges_img.shape[0] // 2)
        rotation_matrix_rough = cv2.getRotationMatrix2D(center_rotate, angle_degrees_rough, 1)
        edges_img_rough = cv2.warpAffine(edges_img, rotation_matrix_rough, (edges_img.shape[1], edges_img.shape[0]))

        results = np.zeros((12, edges_img_rough.shape[0], edges_img_rough.shape[1]))
        for jj, gabor in enumerate(gabors):
            result = cv2.filter2D(edges_img_rough.astype(np.float32), -1, gabor)
            results[jj] = result

        valid_ind = np.argmax(results, axis=0)[edges_img_rough == 255]
        unique, counts = np.unique(valid_ind, return_counts=True)
        most_common_item = unique[np.argmax(counts)]
        angle_degree_fine = 90 - 180/12 * most_common_item

        angle_combined = angle_degrees_rough - angle_degree_fine + 180
        return angle_combined

    def _get_gabor_filters(self) -> list:
        return GaborFilter(15, 12, sigma=3, lambd=10, gamma=0.2)

    def rotate_image(self, image: np.ndarray, angle_degrees: float) -> np.ndarray:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return rotated

    def rotate_keypoints(self, image, angle_degrees, kk_data):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        transformed_points = np.hstack([kk_data, np.ones((kk_data.shape[0], 1))])
        transformed_points = np.dot(transformed_points, rotation_matrix.T)[:, :2]
        transformed_points = np.round(transformed_points).astype(int)
        return transformed_points

    def process_palm_contour_rough(self, sorted_edges_indices, labels, len_palm, hull_contour_small, concave_contour_img, hull_contour):
        concave_hull = concave_contour_img | hull_contour_small
        finger_edges = concave_contour_img | hull_contour_small

        th_removed = 6/self.ratio
        edges_counts = 0
        th_edge_count = 3
        hull_points = np.argwhere(hull_contour == 255)

        for ind in sorted_edges_indices:
            if edges_counts > th_edge_count:
                break
            gray_edge = np.zeros((self.w, self.h))
            gray_edge[labels == ind + 1] = 255
            gray_edge_skele = self.get_skeleton(gray_edge)

            if gray_edge_skele.sum()/255 < 25:
                continue

            results_graph = self.find_closest_path_graph(gray_edge_skele)

            if len(results_graph[1]) == 0:
                continue

            flag_valley, rightnode_index = self.judge_valley(results_graph[1])

            if not flag_valley:
                right_node = results_graph[1][-1]
                left_node = results_graph[1][0]
                left_hull_points = hull_points[np.argwhere(hull_points[:,0] == left_node[0])]
                left_dest = left_node[1] - np.min(left_hull_points[:,0,1])
                if left_dest > (len_palm/6):
                    continue
                path_valid = self.find_farthest_point(results_graph[1], th_removed)
                gray_edge_skele = np.zeros_like(self.cut_norm_img)
                cv2.polylines(gray_edge_skele, [path_valid], isClosed=False, color=(255, 255, 255), thickness=1)
                closest_point, closest_dist = self.find_closest_white_point(concave_hull, path_valid[0])
                cv2.line(gray_edge_skele, closest_point, tuple(path_valid[0]), 255, thickness=1)
            else:
                dest_node = results_graph[1][-1]
                src_node = results_graph[1][0]
                dest_hull_points = hull_points[np.argwhere(hull_points[:,0] == dest_node[0])]
                dist_dest = dest_node[1] - np.min(dest_hull_points[:,0,1])
                src_hull_points = hull_points[np.argwhere(hull_points[:,0] == src_node[0])]
                src_dest = src_node[1] - np.min(src_hull_points[:,0,1])
                dist_node = min(dist_dest, src_dest)
                if dist_node > (len_palm/5):
                    continue
                gray_edge_skele = results_graph[0].astype(np.uint8)
                closest_point_dest, closest_dist = self.find_closest_white_point(concave_hull, dest_node[::-1])
                cv2.line(gray_edge_skele, closest_point_dest, dest_node[::-1], 255, thickness=1)
                closest_point_src, closest_dist = self.find_closest_white_point(concave_hull, src_node[::-1])
                cv2.line(gray_edge_skele, closest_point_src, src_node[::-1], 255, thickness=1)

            edges_counts += 1
            finger_edges = finger_edges | gray_edge_skele

        finger_edges = finger_edges.astype(np.uint8)
        return finger_edges


# ─────────────────────────────────────────────
# Main: ROI extraction only
# ─────────────────────────────────────────────

def run_extraction(dir_source, dir_output):
    os.makedirs(dir_output, exist_ok=True)

    all_images = []
    for root, _, files in os.walk(dir_source):
        for filename in sorted(files):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append((root, filename))

    print(f"Found {len(all_images)} images.")

    num_success = 0
    num_failed = 0

    for idx, (root, filename) in enumerate(all_images):
        img_path = os.path.join(root, filename)

        rel_path = os.path.relpath(root, dir_source)
        out_dir = os.path.join(dir_output, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            print(f"{idx}: Skipped {filename} (not a readable image)")
            num_failed += 1
            continue

        try:
            get_roi = GetROI(gray_img, ratio_rotate=1, ratio=1)
            get_roi.run_rotate()

            get_roi.norm_img = get_roi.rotate_image(get_roi.ori_img, get_roi.rotation_angle)
            height, width = get_roi.norm_img.shape[:2]
            get_roi.cut_norm_img = get_roi.norm_img[0:height, 0:int(width/4*3)]

            bool_result = get_roi.run_localization()
        except Exception as e:
            print(f"{idx}: Error processing {filename}: {e}")
            num_failed += 1
            continue

        if bool_result:
            kk_pred = np.array(get_roi.keypoints_localization)
            kk_pred_transformed = get_roi.rotate_keypoints(get_roi.norm_img, -get_roi.rotation_angle, kk_pred)

            roi_pred, corner_points_pred, _ = extract_roi(
                kk_pred_transformed[0], kk_pred_transformed[1],
                gray_img, color=[0, 0, 255], thickness=2
            )

            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, roi_pred)
            print(f"{idx}: Saved ROI -> {out_path}")
            num_success += 1
        else:
            print(f"{idx}: Failed to extract ROI for {filename}")
            num_failed += 1

    print(f"\nDone. Success: {num_success}, Failed: {num_failed}")


if __name__ == '__main__':
    dir_source = "/home/pai-ng/Jamal/MPDv2"
    dir_output = "/home/pai-ng/Jamal/MPDv2_eralign_roi"

    run_extraction(dir_source, dir_output)
