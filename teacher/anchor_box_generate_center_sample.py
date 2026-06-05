import numpy as np

IMAGE_SIZE = 640

MIN_RATIO = 1.0
MAX_RATIO = 4.0
# we might need to change this sig_scale !!
SIG_SCALE = 4.0



''' 
make_random_anchor_shapes()
- input: stride, N_short_sides to sample, N_ratios to sample
- we sample short sides using randint between 10*stride to 20*stride
- the ratios are sampled N_ratio times per each sampled short side, using triangular distribution of [min 1.0, mode 1.0, max 4.0 or max_ratio allowed by image size]

total number of anchor box types : B (= N_short_sides * N_ratios * 2)
- output : (B, 2) shaped integer array where last dim is width and height
'''

# gives shapes of N_short_sides * N_ratios * 2
def make_random_anchor_shapes(stride : int, N_short_sides : int, N_ratios : int):
    # 10 - 20 in short side
    short_min = stride * 10
    short_max = stride * 20

    # sample short sides with randint (WE COULD CHANGE THIS IF NEEDED : Look at all log_dicts)
    short_sides = np.random.randint(short_min, short_max + 1, N_short_sides)

    # ensure 1:4 max ratio doesnt go over whole image
    max_ratio_for_each_shortside = np.minimum(MAX_RATIO, IMAGE_SIZE / short_sides)
    # ratios : (R,)
    rng = np.random.default_rng()

    # sample ratios per short side using its own upper bound (S, R)
    ratios = np.array([
        rng.triangular(left=MIN_RATIO, mode=MIN_RATIO, right=max_r, size = N_ratios)
        for max_r in max_ratio_for_each_shortside
    ])

    # compute long_sides (S, R) -> (S * R,)
    long_sides = (short_sides[:,None] * ratios).astype(int).flatten()

    # make final anchors
    anchor_types = np.stack([np.repeat(short_sides, N_ratios), long_sides], axis = 1)
    total_anchors = np.concatenate([anchor_types, anchor_types[:, [1, 0]]], axis = 0)

    # we set logs, to show ratios per short side length!!
    log_dict = dict([(short_side, ratios[idx]) for idx, short_side in enumerate(short_sides)])
    return total_anchors, log_dict








# target_coords : GT box coordinates for one Image
# - If there are 3 objects in this image it will be shaped (3, 4) (last dim : x1, y1, x2, y2 of integers)
# anchor_boxes : (B, 2) where B is number of anchor boxes (last dim : w, h)
def get_GToverlap_center_regions_SINGLE(target_coords : np.array, anchor_boxes : np.array):
    assert (target_coords.shape[1] == 4) and (anchor_boxes.shape[1] == 2)
    assert target_coords.dtype == np.int32
    assert anchor_boxes.dtype == np.int32
    no_sample_pairs = []

    N_Objects, _ = target_coords.shape
    N_Boxes, _ = anchor_boxes.shape

    x1, y1, x2, y2 = target_coords.T
    b_w, b_h = anchor_boxes.T
    
    # (N_Objects, 2)
    target_centers = np.stack([target_coords[:,[0,2]].mean(-1), target_coords[:,[1,3]].mean(-1)], axis=0).T

    # (B, )
    half_b_w = b_w / 2.0
    half_b_h = b_h / 2.0

    # each (B, T)
    overlap_x_min = np.maximum(np.ceil(x1[:,None] - half_b_w[None, :] + 1), 0).T.astype(np.int32)
    overlap_x_max = np.minimum(np.floor(x2[:,None] + half_b_w[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)
    overlap_y_min = np.maximum(np.ceil(y1[:,None] - half_b_h[None, :] + 1), 0).T.astype(np.int32)
    overlap_y_max = np.minimum(np.floor(y2[:,None] + half_b_h[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    sampled_centers = - np.ones((N_Boxes, N_Objects, 2))

    for b_idx in range(N_Boxes):
        mask[:, :] = 0
        # should follow the shape of anchor box
        sigma_x = b_w[b_idx] / SIG_SCALE
        sigma_y = b_h[b_idx] / SIG_SCALE

        # 1st pass : Make bit mask for this anchor box A, loop for all GT objects
        for t_idx in range(N_Objects):
            x_min = overlap_x_min[b_idx, t_idx]
            x_max = overlap_x_max[b_idx, t_idx]
            y_min = overlap_y_min[b_idx, t_idx]
            y_max = overlap_y_max[b_idx, t_idx]

            mask[x_min : (x_max + 1), y_min : (y_max + 1)] |= (1<<t_idx)
        
        # 2nd pass : For each GT object, sample with Gaussian dist from valid center points
        for t_idx in range(N_Objects):
            x_min = overlap_x_min[b_idx, t_idx]
            x_max = overlap_x_max[b_idx, t_idx]
            y_min = overlap_y_min[b_idx, t_idx]
            y_max = overlap_y_max[b_idx, t_idx]

            # ROI : we only look at overlap region of anchor box & GT object
            ROI = mask[x_min : (x_max + 1), y_min : (y_max + 1)]
            ROI_start = np.array([x_min, y_min])

            # centers that only overlap with this single GT object
            local_anchor_center_candidates = (ROI == 1<<t_idx)
            n_centers = local_anchor_center_candidates.sum()

            # we skip if there are none.
            if n_centers == 0:
                no_sample_pairs.append((b_idx, t_idx))

            # If there is one, we sample using gaussian!
            else:
                # possible center points
                valid_coords = np.argwhere(local_anchor_center_candidates) + ROI_start
    
                # distance from GT object center
                dist_squared = ((valid_coords[:, 0] - target_centers[t_idx, 0]) / sigma_x) ** 2 + ((valid_coords[:, 1] - target_centers[t_idx, 1]) / sigma_y) ** 2
                weights = np.exp(-dist_squared / 2.0)
                weights /= weights.sum()

                sampled = valid_coords[np.random.choice(len(valid_coords), p=weights)]
                sampled_centers[b_idx, t_idx] = sampled

    return sampled_centers, no_sample_pairs


''' 
THIS IS CODE TO BE USED IN boxes.py
It gets all targets, target_index for main target, then compares all other targets to find single overlapping region for main GT
'''
# target_coords : GT box coordinates for one Image
# - If there are 3 objects in this image it will be shaped (3, 4) (last dim : x1, y1, x2, y2 of integers)
# anchor_boxes : (B, 2) where B is number of anchor boxes (last dim : w, h)
def get_GToverlap_center_regions_SINGLE_for_one_GT(target_idx : int, target_coords : np.array, anchor_boxes : np.array):
    assert (target_coords.shape[1] == 4) and (anchor_boxes.shape[1] == 2)
    assert target_coords.dtype == np.int32
    assert anchor_boxes.dtype == np.int32
    no_sample_pairs = []

    N_Objects, _ = target_coords.shape
    N_Boxes, _ = anchor_boxes.shape

    x1, y1, x2, y2 = target_coords.T
    b_w, b_h = anchor_boxes.T
    
    # (N_Objects, 2)
    target_centers = np.stack([target_coords[:,[0,2]].mean(-1), target_coords[:,[1,3]].mean(-1)], axis=0).T

    # (B, )
    half_b_w = b_w / 2.0
    half_b_h = b_h / 2.0

    # each (B, T)
    overlap_x_min = np.maximum(np.ceil(x1[:,None] - half_b_w[None, :] + 1), 0).T.astype(np.int32)
    overlap_x_max = np.minimum(np.floor(x2[:,None] + half_b_w[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)
    overlap_y_min = np.maximum(np.ceil(y1[:,None] - half_b_h[None, :] + 1), 0).T.astype(np.int32)
    overlap_y_max = np.minimum(np.floor(y2[:,None] + half_b_h[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    sampled_centers = - np.ones((N_Boxes, 2))

    for b_idx in range(N_Boxes):
        mask[:, :] = 0
        # should follow the shape of anchor box
        sigma_x = b_w[b_idx] / SIG_SCALE
        sigma_y = b_h[b_idx] / SIG_SCALE

        # 1st pass : Make bit mask for this anchor box A, loop for all GT objects
        for t_idx in range(N_Objects):
            x_min = overlap_x_min[b_idx, t_idx]
            x_max = overlap_x_max[b_idx, t_idx]
            y_min = overlap_y_min[b_idx, t_idx]
            y_max = overlap_y_max[b_idx, t_idx]

            mask[x_min : (x_max + 1), y_min : (y_max + 1)] |= (1<<t_idx)
        
        
        # 2nd pass : For each GT object, sample with Gaussian dist from valid center points
        
        x_min = overlap_x_min[b_idx, target_idx]
        x_max = overlap_x_max[b_idx, target_idx]
        y_min = overlap_y_min[b_idx, target_idx]
        y_max = overlap_y_max[b_idx, target_idx]

        # ROI : we only look at overlap region of anchor box & GT object
        ROI = mask[x_min : (x_max + 1), y_min : (y_max + 1)]
        ROI_start = np.array([x_min, y_min])

        # centers that only overlap with this single GT object
        local_anchor_center_candidates = (ROI == 1<<target_idx)
        n_centers = local_anchor_center_candidates.sum()

        # we skip if there are none.
        if n_centers == 0:
            no_sample_pairs.append((b_idx, target_idx))

        # If there is one, we sample using gaussian!
        else:
            # possible center points
            valid_coords = np.argwhere(local_anchor_center_candidates) + ROI_start

            # distance from GT object center
            dist_squared = ((valid_coords[:, 0] - target_centers[target_idx, 0]) / sigma_x) ** 2 + ((valid_coords[:, 1] - target_centers[target_idx, 1]) / sigma_y) ** 2
            weights = np.exp(-dist_squared / 2.0)
            weights /= weights.sum()

            sampled = valid_coords[np.random.choice(len(valid_coords), p=weights)]
            sampled_centers[b_idx] = sampled

    return sampled_centers, no_sample_pairs




# for multiple GT bboxes sampling! (includes both single, and multiple combinations of 1 each if possible)
# uses independence assumption to multiply each bboxes' probabilities

# <Inputs>
# target_coords : GT box coordinates for one Image
# - If there are 3 objects in this image it will be shaped (3, 4) (last dim : x1, y1, x2, y2 of integers)
# anchor_boxes : (B, 2) where B is number of anchor boxes (last dim : w, h)
# <Outputs>
# final_samples : dict, of the following form
# key : bbox id
# value : list of [(list[overlapping target ids], 2-dim np.array of center x, y )]
# for each bbox, there may be multiple combination of GT objects overlap

def get_GToverlap_center_regions_MULTIPLE_MULTIVARIATE(target_coords : np.array, anchor_boxes : np.array):
    assert (target_coords.shape[1] == 4) and (anchor_boxes.shape[1] == 2)
    assert target_coords.dtype == np.int32
    assert anchor_boxes.dtype == np.int32
    final_samples = {}
    N_Objects, _ = target_coords.shape
    N_Boxes, _ = anchor_boxes.shape

    x1, y1, x2, y2 = target_coords.T
    b_w, b_h = anchor_boxes.T
    
    # (N_Objects, 2)
    target_centers = np.stack([target_coords[:,[0,2]].mean(-1), target_coords[:,[1,3]].mean(-1)], axis=0).T

    # (B, )
    half_b_w = b_w / 2.0
    half_b_h = b_h / 2.0

    # each (B, T)
    overlap_x_min = np.maximum(np.ceil(x1[:,None] - half_b_w[None, :] + 1), 0).T.astype(np.int32)
    overlap_x_max = np.minimum(np.floor(x2[:,None] + half_b_w[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)
    overlap_y_min = np.maximum(np.ceil(y1[:,None] - half_b_h[None, :] + 1), 0).T.astype(np.int32)
    overlap_y_max = np.minimum(np.floor(y2[:,None] + half_b_h[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    for b_idx in range(N_Boxes):
        final_samples[b_idx] = []

        mask[:, :] = 0
        # should follow the shape of anchor box
        sigma_x = b_w[b_idx] / SIG_SCALE
        sigma_y = b_h[b_idx] / SIG_SCALE

        # 1st pass : Make bit mask for this anchor box A, loop for all GT objects
        for t_idx in range(N_Objects):
            x_min = overlap_x_min[b_idx, t_idx]
            x_max = overlap_x_max[b_idx, t_idx]
            y_min = overlap_y_min[b_idx, t_idx]
            y_max = overlap_y_max[b_idx, t_idx]

            mask[x_min : (x_max + 1), y_min : (y_max + 1)] |= (1<<t_idx)
        
        unique_combinations = np.unique(mask)
        unique_combinations = unique_combinations[unique_combinations != 0]

        # 2nd pass : For each GT object, sample with Gaussian dist from valid center points
        for combination in unique_combinations:
            # Decode bit mask to list of overlapping GT objects
            active_t_indices = [t for t in range(N_Objects) if (combination & (1 << t))]
            # Use frirst GT object to reduce region to search
            t_idx_0 = active_t_indices[0]

            x_min = overlap_x_min[b_idx, t_idx_0]
            x_max = overlap_x_max[b_idx, t_idx_0]
            y_min = overlap_y_min[b_idx, t_idx_0]
            y_max = overlap_y_max[b_idx, t_idx_0]

            # ROI : we only look at overlap region of anchor box & frirst GT object to reduce region to search
            ROI = mask[x_min : (x_max + 1), y_min : (y_max + 1)]
            ROI_start = np.array([x_min, y_min])

            # centers that only overlap with this single GT object
            local_anchor_center_candidates = (ROI == combination)

            # possible center points
            valid_coords = np.argwhere(local_anchor_center_candidates) + ROI_start

            '''
            Multiplying probability from each GT objects (Independence assumption -> would lead to center of mass frequentlysampled)
            '''
            dist_squared = np.zeros(len(valid_coords))

            # distance from all center points computation
            for t_idx in active_t_indices:
                dist_squared += ((valid_coords[:, 0] - target_centers[t_idx, 0]) / sigma_x) ** 2 + ((valid_coords[:, 1] - target_centers[t_idx, 1]) / sigma_y) ** 2

            log_weights = -dist_squared / 2.0
            log_weights -= log_weights.max()
            weights = np.exp(log_weights)
            weights /= weights.sum()

            sampled = valid_coords[np.random.choice(len(valid_coords), p=weights)]
            
            final_samples[b_idx].append((active_t_indices, sampled))
            
    return final_samples




# for multiple GT bboxes sampling! (includes both single, and multiple combinations of 1 each if possible)
# uses GMM type of sum of probabilities
# <Inputs>
# target_coords : GT box coordinates for one Image
# - If there are 3 objects in this image it will be shaped (3, 4) (last dim : x1, y1, x2, y2 of integers)
# anchor_boxes : (B, 2) where B is number of anchor boxes (last dim : w, h)
# <Outputs>
# final_samples : dict, of the following form
# key : bbox id
# value : list of [(list[overlapping target ids], 2-dim np.array of center x, y )]
# for each bbox, there may be multiple combination of GT objects overlap

def get_GToverlap_center_regions_MULTIPLE_GMM(target_coords : np.array, anchor_boxes : np.array):
    assert (target_coords.shape[1] == 4) and (anchor_boxes.shape[1] == 2)
    assert target_coords.dtype == np.int32
    assert anchor_boxes.dtype == np.int32
    final_samples = {}

    N_Objects, _ = target_coords.shape
    N_Boxes, _ = anchor_boxes.shape

    x1, y1, x2, y2 = target_coords.T
    b_w, b_h = anchor_boxes.T
    
    # (N_Objects, 2)
    target_centers = np.stack([target_coords[:,[0,2]].mean(-1), target_coords[:,[1,3]].mean(-1)], axis=0).T

    # (B, )
    half_b_w = b_w / 2.0
    half_b_h = b_h / 2.0

    # each (B, T)
    overlap_x_min = np.maximum(np.ceil(x1[:,None] - half_b_w[None, :] + 1), 0).T.astype(np.int32)
    overlap_x_max = np.minimum(np.floor(x2[:,None] + half_b_w[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)
    overlap_y_min = np.maximum(np.ceil(y1[:,None] - half_b_h[None, :] + 1), 0).T.astype(np.int32)
    overlap_y_max = np.minimum(np.floor(y2[:,None] + half_b_h[None, :] - 1), IMAGE_SIZE - 1).T.astype(np.int32)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    for b_idx in range(N_Boxes):
        final_samples[b_idx] = []
        mask[:, :] = 0

        # should follow the shape of anchor box
        sigma_x = b_w[b_idx] / SIG_SCALE
        sigma_y = b_h[b_idx] / SIG_SCALE

        # 1st pass : Make bit mask for this anchor box A, loop for all GT objects
        for t_idx in range(N_Objects):
            x_min = overlap_x_min[b_idx, t_idx]
            x_max = overlap_x_max[b_idx, t_idx]
            y_min = overlap_y_min[b_idx, t_idx]
            y_max = overlap_y_max[b_idx, t_idx]

            mask[x_min : (x_max + 1), y_min : (y_max + 1)] |= (1<<t_idx)
        
        unique_combinations = np.unique(mask)
        unique_combinations = unique_combinations[unique_combinations != 0]

        # 2nd pass : For each GT object, sample with Gaussian dist from valid center points
        for combination in unique_combinations:
            # Decode bit mask to list of overlapping GT objects
            active_t_indices = [t for t in range(N_Objects) if (combination & (1 << t))]
            # Use frirst GT object to reduce region to search
            t_idx_0 = active_t_indices[0]

            x_min = overlap_x_min[b_idx, t_idx_0]
            x_max = overlap_x_max[b_idx, t_idx_0]
            y_min = overlap_y_min[b_idx, t_idx_0]
            y_max = overlap_y_max[b_idx, t_idx_0]

            # ROI : we only look at overlap region of anchor box & frirst GT object to reduce region to search
            ROI = mask[x_min : (x_max + 1), y_min : (y_max + 1)]
            ROI_start = np.array([x_min, y_min])

            # centers that only overlap with this single GT object
            local_anchor_center_candidates = (ROI == combination)

            # possible center points
            valid_coords = np.argwhere(local_anchor_center_candidates) + ROI_start

            '''
            Sum of probability from each GT objects ( -> would lead to softened single GT sampling)
            '''
            prob_summed = np.zeros(len(valid_coords))

            # distance from all center points computation
            for t_idx in active_t_indices:
                log_p = -(((valid_coords[:, 0] - target_centers[t_idx, 0]) / sigma_x) ** 2 + ((valid_coords[:, 1] - target_centers[t_idx, 1]) / sigma_y) ** 2) / 2.0
                log_p -= log_p.max()
                prob_for_one_object = np.exp(log_p)
                prob_for_one_object /= prob_for_one_object.sum()
                prob_summed += prob_for_one_object

            
            prob_summed /= prob_summed.sum()

            sampled = valid_coords[np.random.choice(len(valid_coords), p=prob_summed)]

            final_samples[b_idx].append((active_t_indices, sampled))

    return final_samples