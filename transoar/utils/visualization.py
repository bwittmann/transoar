"""Helper functions for visualization purposes."""

from collections import defaultdict
import math

import cv2
import torch
import open3d as o3d
import numpy as np
import torch.nn.functional as F

from transoar.utils.io import write_ply

PALETTE = {
    1: [255, 0, 0], # colors for organ point cloud
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 153, 153],
    5: [255, 204, 153],
    6: [255, 255, 153],
    7: [204, 255, 153],
    8: [153, 255, 153],
    9: [153, 255, 255],
    10: [153, 153, 255],
    11: [255, 153, 255],
    12: [204, 0, 0],
    13: [204, 204, 0],
    14: [0, 204, 0],
    15: [160, 160, 160],
    16: [51, 244, 153],
    17: [255, 102, 178],
    18: [255, 204, 204],
    19: [255, 153, 51],
    20: [153, 204, 255],
    1000: [0, 255, 0],  # colors for bboxes (gt and pred)
    1001: [255, 255, 0] 
}


def show_images(images):
    """Displays a series of images.
    
    Args: A list containing arbitrary many 2D np.ndarrays
    """
    for image in images:
        cv2.imshow('Raw NIfTI', image)
        cv2.waitKey()

def normalize(image):
    """Normalizes an image to uint8 to display as greyscale.

    Args:
        image: A np.ndarray of dim 3d representing data or our modality.

    Returns:
        The same image, but converted to uint8.
    """
    norm_img = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return norm_img

def visualize_voxel_grid(data):
    """Visualizes layers of a voxel grid.

    Args:
        data: A np.ndarray corresponding to the voxel grid data.
    """
    images = defaultdict(list)

    data = normalize(data.squeeze())
    assert len(data.shape) == 3, 'Data has to be 3D.'

    for layer in [int(x) for x in np.linspace(0, data.shape[0] - 1 , 5)]:
        images['axis_0'].append(data[layer, :, :])

    for layer in [int(x) for x in np.linspace(0, data.shape[1] - 1, 5)]:
        images['axis_1'].append(data[:, layer, :])

    for layer in [int(x) for x in np.linspace(0, data.shape[2] - 1, 5)]:
        images['axis_2'].append(data[:, :, layer])

    image_axis_0 = normalize(np.concatenate(images['axis_0'], axis=1))
    image_axis_1 = normalize(np.concatenate(images['axis_1'], axis=1))
    image_axis_2 = normalize(np.concatenate(images['axis_2'], axis=1))

    show_images([image_axis_0, image_axis_1, image_axis_2])

def incorporate_bboxes(labels, data=None, seg_map=None, standalone=True, value=None):
    """Incorporate bounding boxes to a 3D volume.

    This makes it possible to visualize bboxes of the format x1, y1, z1, x2, y2, z2
    with a tool like ITK-SNAP.
    
    Args:
        labels: A tuple consisting of a torch.tensor of the shape [N, 6] representing bboxes
            and a list of length N representing the respective classes.
        data: A torch.tensor representing the the image data.
        seg_map: A torch.tensor representing the segmentation labels.
        standalone: If True, add bounding boxes to a np.ndarray full of zeros.
        value: If not None, the bboxes will have this value in the array.

    Returns:
        A np.ndarray containing the bboxes in a visualizable format.
    """
    # Generate volume to add bboxes
    if standalone:
        if data is not None:
            bboxes_volume = np.zeros_like(data).squeeze()
        elif seg_map is not None:
            bboxes_volume = np.zeros_like(seg_map).squeeze()
        else:
            raise RuntimeError('Please input either the data or the seg_map.')
    elif data is not None:
        bboxes_volume = data.numpy().squeeze()
    elif seg_map is not None:
        bboxes_volume = seg_map.numpy().squeeze()

    bboxes = labels[0].split(1)
    classes = labels[1]
    for bbox, class_ in zip(bboxes, classes):
        x1, y1, z1, x2, y2, z2 = bbox[0].tolist()

        if value:
            bbox_val = value
        else:
            bbox_val = class_

        for y, z in [(y1, z1), (y1, z2), (y2, z1), (y2, z2)]:
            for x_val in range(x1, x2 + 1):
                bboxes_volume[x_val, y, z] = bbox_val

        for x, z in [(x1, z1), (x1, z2), (x2, z1), (x2, z2)]:
            for y_val in range(y1, y2 + 1):
                bboxes_volume[x, y_val, z] = bbox_val

        for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
            for z_val in range(z1, z2 + 1):
                bboxes_volume[x, y, z_val] = bbox_val

    return bboxes_volume

def save_pred_visualization(
    pred_boxes, pred_classes, gt_boxes, gt_classes, seg_mask, path, classes, idx, create_mesh=False
):
    """Saves predictions, gt, and, data as .ply files for visualization purposes.

    Args:
        pred_boxes: A np.ndarray of shape [N, 6] in the format [cx, cy, cz, w, h, d].
        pred_classes: A np.ndarray of shape [N,].
        gt_boxes: A np.ndarray of shape [M, 6] in the format [cx, cy, cz, w, h, d].
        gt_classes: A np.ndarray of shape [M,]. 
        seg_mask: A torch.Tensor representing the segmentation map.
        path: A path to the dir of the run to be predicted.
        classes: A dict containing the class id to class name mapping.
        idx: The index of the current case to be predicted.
        create_mesh: If True, create mesh out of seg_mask instead of point cloud.
            This should not be invoked, as it currently works poorly.
    """
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.squeeze().detach().cpu().numpy()

    # Create dir for current instance
    instance_name = 'case_' + str(idx)
    path_to_instance = path / instance_name
    path_to_instance.mkdir(exist_ok=True)

    # Generate point cloud of each class and gt boxes
    for bbox, class_ in zip(gt_boxes, gt_classes):
        class_pc = (seg_mask == class_).nonzero()
        class_pc = np.stack(class_pc).T   # xyz
        class_pc_color = np.repeat((np.array(PALETTE[class_]) / 255)[None], class_pc.shape[0], axis=0)

        # Class point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(class_pc)
        pcd.colors = o3d.utility.Vector3dVector(class_pc_color)

        if create_mesh:    # Create triangle mesh from point cloud
            pcd.estimate_normals()

            poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )[0]

            bbox_mesh = pcd.get_axis_aligned_bounding_box()
            mesh = poisson_mesh.crop(bbox_mesh)

            path_mesh = str(path_to_instance) + f'/{classes[str(class_)]}_mesh.ply'
            o3d.io.write_triangle_mesh(path_mesh, mesh)
        else:
            path_pc = str(path_to_instance) + f'/{classes[str(class_)]}_pc.ply'
            o3d.io.write_point_cloud(path_pc, pcd)

        # Gt bboxes
        bbox = rescale_bbox(bbox, seg_mask.shape)
        bbox = np.concatenate((bbox, np.array([0])), axis=0)
        path_bbox = str(path_to_instance) + f'/{classes[str(class_)]}_bbox_gt.ply'
        write_bbox(bbox, 1000, path_bbox, PALETTE, seg_mask.shape[-1] / 750)

    # Generate pred bboxes
    for bbox, class_ in zip(pred_boxes, pred_classes):
        bbox = rescale_bbox(bbox, seg_mask.shape)

        bbox = np.concatenate((bbox, np.array([0])), axis=0)
        path_bbox = str(path_to_instance) + f'/{classes[str(class_)]}_bbox_pred.ply'
        write_bbox(bbox, 1001, path_bbox, PALETTE, seg_mask.shape[-1] / 750)

def rescale_bbox(bbox, original_shape):
        # Change val of bboxes from sigmoid range back to meaningful values
        bbox[:3] = bbox[:3] * original_shape
        bbox[3:] = bbox[3:] * original_shape
        return bbox

def save_attn_visualization(
    model_out, backbone_features, enc_attn_weights, dec_attn_weights, original_shape, seg_mask
):
    seg_mask = seg_mask.squeeze()

    # Get shape of input the feature map
    d, h, w = backbone_features.shape[-3:]

    # Get all predicted classes and bboxes for all queries
    all_pred_classes = torch.max(F.softmax(model_out['pred_logits'], dim=-1), dim=-1)[1].squeeze()
    all_pred_boxes = model_out['pred_boxes'].squeeze()

    # Decoder cross attn weights, averaged over all heads
    for query_id in range(dec_attn_weights.shape[0]):
        query_dec_attn_weights = dec_attn_weights[query_id].view(d, h, w)                        
        assert torch.isclose(query_dec_attn_weights.sum(), torch.tensor([1.]))

        # Upsample attn weights to original input shape
        query_dec_attn_weights = F.interpolate(query_dec_attn_weights[None, None], original_shape).squeeze()

        # Get class and bbox of current query
        query_class = all_pred_classes[query_id]
        # query_box = rescale_bbox(all_pred_boxes[query_id], original_shape)

        # Generate complete pc and different colored query pc
        pc_rest = torch.nonzero(torch.logical_and(seg_mask != 0, seg_mask != query_class))
        pc_rest_color = (torch.tensor(PALETTE[3]) / 255)[None].repeat(pc_rest.shape[0], 1)

        pc_query = torch.nonzero(seg_mask == query_class)
        pc_query_color = (torch.tensor(PALETTE[1]) / 255)[None].repeat(pc_query.shape[0], 1)

        pc_comb, pc_comb_color = torch.cat((pc_rest, pc_query), dim=0), torch.cat((pc_rest_color, pc_query_color), dim=0)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_comb)
        pc.colors = o3d.utility.Vector3dVector(pc_comb_color)
        pc_organs = pc 

        pc_attn_weights = []

        query_dec_attn_weights = query_dec_attn_weights.flatten()
        mask = torch.ones_like(query_dec_attn_weights).to(dtype=torch.bool)
        mask[::30] = False
        query_dec_attn_weights[mask] = -1
        query_dec_attn_weights = query_dec_attn_weights.view(original_shape) 

        weights = query_dec_attn_weights.unique()[2:]
        for weight in weights:
            pc_attn_weight = torch.nonzero(query_dec_attn_weights == weight)
            color = torch.tensor([1, 1, 1]) - 60 * weight
            color[0] = 1

            pc_attn_color = color[None].repeat(pc_attn_weight.shape[0], 1)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pc_attn_weight)
            pc.colors = o3d.utility.Vector3dVector(pc_attn_color)

            pc_attn_weights.append(pc)

        for idx, pc in enumerate(pc_attn_weights):
            o3d.io.write_point_cloud(f'/home/bastian/Downloads/tester/attn{idx}.ply', pc)

        o3d.io.write_point_cloud('/home/bastian/Downloads/tester/test.ply', pc_organs)
        o3d.visualization.draw_geometries([pc_organs, *pc_attn_weights])

def write_bbox(bbox, mode, output_file, palette, diameter=0.3):
    """Generate a .ply file representing a bbox.

    Args:cxcyczwhd
        bbox: A np.array representig the bbox to visualize in the format [cx, cy, cz, w, h, d, r].
        mode: If 1000: gt, if 1001: pred.
        output_file: A str representing the absolute path to output file.
        palette: A dict containing RGB values as values. Has to include the keys 1000 and 1001.
        diameter: A float representing the diameter of the bbox struts.
    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius * math.sin(theta), height * i / stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if math.fabs(dotx) != 1.0:
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
        ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
        zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0)  # 8 x 3

        return corners

    radius = diameter / 2
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)

    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)


            



        
    
    

