import numpy as np
from pytorch3d.ops import ball_query
import torch
import open3d as o3d
from utils.geometry import denoise
from torch.nn.utils.rnn import pad_sequence
from scipy.spatial import KDTree
from itertools import chain
from sklearn.cluster import DBSCAN

# COVERAGE_THRESHOLD = 0.3
# DISTANCE_THRESHOLD = 0.03
# FEW_POINTS_THRESHOLD = 25
# DEPTH_TRUNC = 20
# BBOX_EXPAND = 0.1

COVERAGE_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 0.03
FEW_POINTS_THRESHOLD = 25
DEPTH_TRUNC_MAX = 1000000
BBOX_EXPAND = 0.1


def backproject(depth, intrinisc_cam_parameters, extrinsics):
    """
    convert color and depth to view pointcloud
    """
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=DEPTH_TRUNC) # DEPTH_TRUNC
    pcld.transform(extrinsics)
    return pcld


def get_neighbor(valid_points, scene_points, lengths_1, lengths_2):
    _, neighbor_in_scene_pcld, _ = ball_query(valid_points, scene_points, lengths_1, lengths_2, K=20, radius=DISTANCE_THRESHOLD, return_nn=False)
    return neighbor_in_scene_pcld


def get_depth_mask(depth):
    depth_tensor = torch.from_numpy(depth).cuda()
    depth_mask = torch.logical_and(depth_tensor > 0, depth_tensor < DEPTH_TRUNC_MAX).reshape(-1)  # DEPTH_TRUNC 20
    return depth_mask


def crop_scene_points(mask_points, scene_points):
    x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
    y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
    z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

    selected_point_mask = (scene_points[:, 0] > x_min) & (scene_points[:, 0] < x_max) & (scene_points[:, 1] > y_min) & (scene_points[:, 1] < y_max) & (scene_points[:, 2] > z_min) & (scene_points[:, 2] < z_max)
    selected_point_ids = torch.where(selected_point_mask)[0]
    cropped_scene_points = scene_points[selected_point_ids]
    return cropped_scene_points, selected_point_ids


def turn_mask_to_point(dataset, scene_points, mask_image, frame_id, depth_max_pre=0.03):
    intrinisc_cam_parameters = dataset.get_intrinsics(frame_id)
    extrinsics = dataset.get_extrinsic(frame_id)
    if np.sum(np.isinf(extrinsics)) > 0:
        return {}, [], []

    mask_image = torch.from_numpy(mask_image).cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()
    
    depth = dataset.get_depth(frame_id)
    depth_mask = get_depth_mask(depth)
    
    colored_pcld = backproject(depth, intrinisc_cam_parameters, extrinsics)
    view_points = np.asarray(colored_pcld.points)

    mask_points_list = []
    mask_points_num_list = []
    scene_points_list = []
    scene_points_num_list = []
    selected_point_ids_list = []
    initial_valid_mask_ids = []
    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[depth_mask].cpu().numpy()

        mask_pcld = o3d.geometry.PointCloud()
        mask_points = view_points[valid_mask]
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue
        mask_pcld.points = o3d.utility.Vector3dVector(mask_points)

        mask_pcld = mask_pcld.voxel_down_sample(voxel_size=DISTANCE_THRESHOLD)  # DISTANCE_THRESHOLD

        mask_pcld, _ = denoise(mask_pcld)

        mask_points = np.asarray(mask_pcld.points)
        
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue
        
        mask_points = torch.tensor(mask_points).float().cuda()
        cropped_scene_points, selected_point_ids = crop_scene_points(mask_points, scene_points)
        initial_valid_mask_ids.append(mask_id)
        mask_points_list.append(mask_points)
        scene_points_list.append(cropped_scene_points)
        mask_points_num_list.append(len(mask_points))
        scene_points_num_list.append(len(cropped_scene_points))
        selected_point_ids_list.append(selected_point_ids)

    if len(initial_valid_mask_ids) == 0:
        return {}, [], []
    
    mask_points_tensor = pad_sequence(mask_points_list, batch_first=True, padding_value=0)
    scene_points_tensor = pad_sequence(scene_points_list, batch_first=True, padding_value=0)

    lengths_1 = torch.tensor(mask_points_num_list).cuda()
    lengths_2 = torch.tensor(scene_points_num_list).cuda()
    neighbor_in_scene_pcld = get_neighbor(mask_points_tensor, scene_points_tensor, lengths_1, lengths_2)
    # print(neighbor_in_scene_pcld.shape)

    valid_mask_ids = []
    mask_info = {}
    frame_point_ids = set()

    for i, mask_id in enumerate(initial_valid_mask_ids):
        mask_neighbor = neighbor_in_scene_pcld[i] # P, 20
        mask_point_num = mask_points_num_list[i] # Pi
        mask_neighbor = mask_neighbor[:mask_point_num] # Pi, 20

        valid_neighbor = mask_neighbor != -1 # Pi, 20
        neighbor = torch.unique(mask_neighbor[valid_neighbor])
        neighbor_in_complete_scene_points = selected_point_ids_list[i][neighbor].cpu().numpy()
        coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

        if coverage < COVERAGE_THRESHOLD:
            continue
        valid_mask_ids.append(mask_id)
        mask_info[mask_id] = set(neighbor_in_complete_scene_points)
        frame_point_ids.update(mask_info[mask_id])

    return mask_info, valid_mask_ids, list(frame_point_ids)


def cluster_points_with_dbscan(point_cloud, point_ids, eps=0.5, min_samples=5):
    """
    使用DBSCAN算法对三维点云进行聚类
    
    参数:
    point_cloud (np.ndarray): N×3的numpy数组，包含三维坐标
    point_ids (list): 长度N的列表，对应每个点的唯一标识符
    eps (float): DBSCAN邻域半径，默认0.5
    min_samples (int): 核心点所需的最小邻域点数，默认5
    
    返回:
    list: 聚类结果，每个子列表包含属于同一簇的点ID，噪声点会被排除
    """
    # 验证输入数据有效性
    assert len(point_cloud) == len(point_ids), "点云数据与ID数量不匹配"
    
    # 创建DBSCAN实例并进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(point_cloud)

    # 组织聚类结果
    clusters = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            # 获取当前簇的所有点索引
            cluster_indices = np.where(labels == label)[0]
            # 转换为对应的点ID
            cluster_ids = [point_ids[i] for i in cluster_indices]
            if len(cluster_ids) > 10:
                clusters.append(np.array(cluster_ids))
    
    return clusters

def turn_point_to_mask(dataset, scene_points, mask_image, frame_id, depth_max_pre, threshold=0.1):
    intrinisc_cam_parameters = dataset.get_intrinsics(frame_id)
    extrinsics = dataset.get_extrinsic(frame_id)
    if np.sum(np.isinf(extrinsics)) > 0:
        return {}, [], [], 20

    mask_image = torch.from_numpy(mask_image).cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()
    # print(ids)
    depth = dataset.get_depth(frame_id)
    
    DEPTH_TRUNC = 100000
    depth_max = 0.1

    depth_tensor = torch.from_numpy(depth).cuda()
    depth_mask_ = torch.logical_and(depth_tensor > 0, depth_tensor < DEPTH_TRUNC).reshape(-1)
    mask_image = mask_image * (depth_mask_ * 1)

    depth_mask = get_depth_mask(depth)
    
    colored_pcld = backproject(depth, intrinisc_cam_parameters, extrinsics)
    view_points = np.asarray(colored_pcld.points)

    mask_points_list = []
    mask_points_num_list = []
    scene_points_list = []
    scene_points_num_list = []
    selected_point_ids_list = []
    initial_valid_mask_ids = []

    points_a_frame = torch.tensor(view_points).float().cuda()

    if len(points_a_frame) == 0:
        return {}, [], [], 0.1
    
    cropped_scene_points_frame, selected_point_ids_frame = crop_scene_points(points_a_frame, scene_points)

    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id

        valid_mask = segmentation[depth_mask].cpu().numpy()

        mask_points = view_points[valid_mask]

        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue

        mask_pcld = o3d.geometry.PointCloud()
        mask_pcld.points = o3d.utility.Vector3dVector(mask_points)
        
        mask_pcld = mask_pcld.voxel_down_sample(voxel_size=DISTANCE_THRESHOLD)  # 0.03

        mask_pcld, _ = denoise(mask_pcld)

        mask_points = np.asarray(mask_pcld.points)


        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue

        mask_points_list.append(mask_points)
        initial_valid_mask_ids.append(mask_id)
        mask_points_num_list.append(len(mask_points))


    if len(initial_valid_mask_ids) == 0:
        return {}, [], [], 20
    
    valid_mask_ids = []
    mask_info = {}
    frame_point_ids = set()

    def preprocess_B(B):
        """预处理B列表，合并所有点并记录每个点对应的mask索引"""
        B_points = []
        mask_indices = []
        for mask_idx, mask in enumerate(B):
            for point in mask:
                B_points.append(point)
                mask_indices.append(mask_idx)
        return np.array(B_points), np.array(mask_indices)

    def find_closest_masks(A, B, k=1, threshold=0.1):   # 0.1
        """
        为A中的每个点找到B中最接近的mask（基于最近k个点的多数），若最近距离超过阈值则返回-1
        :param A: 点云列表，格式[[x1,y1,z1], [x2,y2,z2], ...]
        :param B: mask列表，每个mask是点的列表
        :param k: 考虑的最近邻数量，默认为100
        :param threshold: 距离阈值，最近邻距离超过此值时忽略该点（返回-1）
        :return: 每个点对应的mask索引列表（-1表示超出阈值）
        """
        # 预处理B中的点并构建KDTree
        B_points, mask_indices = preprocess_B(B)
        if B_points.shape[0] == 0:
            raise ValueError("B列表不能为空")
        if k > B_points.shape[0]:
            k = B_points.shape[0]  # 当B中点数不足k时，调整k为最大值
        tree = KDTree(B_points)
        
        # 转换A为numpy数组以提高效率
        A_array = np.array(A)
        
        # 批量查询所有点的最近k个点（减少循环次数）
        # print(A_array)
        distances, indices = tree.query(A_array, k=k)
        
        # 处理k=1的情况，确保二维数组
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        # 统计每个点的最近k个点的mask分布
        nearest_masks = mask_indices[indices]
        
        # 对每个点进行阈值判断和统计
        result = []
        for i in range(len(A_array)):
            # 检查最近邻距离是否超过阈值
            # print(distances[i, 0])
            if distances[i, 0] > threshold:
                result.append(-1)
                # continue
            else:
                counts = np.bincount(nearest_masks[i], minlength=len(B))
                result.append(np.argmax(counts))
        
        return result

    scene_points_list = cropped_scene_points_frame.cpu().numpy()

    if len(scene_points_list) == 0:
        return {}, [], [], 0.1

    idx = find_closest_masks(scene_points_list, mask_points_list, threshold=threshold)

    selected_point_ids_list = selected_point_ids_frame.cpu().numpy()

    for mask_id in range(len(initial_valid_mask_ids)):
        valid_neighbor = np.where(np.array(idx) == mask_id)[0].tolist()
        neighbor_in_complete_scene_points = selected_point_ids_list[valid_neighbor]
        valid_mask_ids.append(initial_valid_mask_ids[mask_id])
        mask_info[initial_valid_mask_ids[mask_id]] = set(neighbor_in_complete_scene_points)
        frame_point_ids.update(mask_info[initial_valid_mask_ids[mask_id]])

    return mask_info, valid_mask_ids, list(frame_point_ids), depth_max


def frame_backprojection(args, dataset, scene_points, frame_id, depth_max_pre):

    mask_image = dataset.get_segmentation(frame_id, align_with_depth=True)

    # mask_info, _, frame_point_ids = turn_mask_to_point(dataset, scene_points, mask_image, frame_id)
    
    if args.dataset == 'matterport3d':  # we observe that "turn_mask_to_point" in MaskClustering [CVPR'2024] will lead to many undetected points, and thus we give a "turn_point_to_mask" for matterport3d.
        mask_info, _, frame_point_ids, _ = turn_point_to_mask(dataset, 
                                                              scene_points, 
                                                              mask_image, 
                                                              frame_id, 
                                                              depth_max_pre=depth_max_pre, 
                                                              threshold=0.1)
    else:
        mask_info, _, frame_point_ids = turn_mask_to_point(dataset, 
                                                           scene_points, 
                                                           mask_image, 
                                                           frame_id, 
                                                           depth_max_pre=depth_max_pre)
    
    return mask_info, frame_point_ids, 0.1