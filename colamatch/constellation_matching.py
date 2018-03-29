from __future__ import print_function, absolute_import, division
import numpy as np
import math
from sklearn.neighbors import KDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import random
import time
import logging

logger = logging.getLogger(__package__)

def _get_furthest_points(pointlist):
    from scipy.spatial.distance import pdist, squareform
    D = pdist(pointlist)
    D = squareform(D)
    [idx1, idx2] = np.unravel_index(np.argmax(D), D.shape)
    return idx1, idx2

def _transform_coordinate_system(pointlist):
    """ Transform points in pointlist so that a is origin and b lies on bisetrix of the coordinate system.
    Args:
        pointlist (array_like): Points a,b,c,d.
    Returns:
        a,b,c,d: Points of given pointlist transformed into new local coordinate system.
    """
    a,b,c,d = pointlist - pointlist[0]
    angle = math.atan2(1,1) - math.atan2(b[1],b[0])  # angle between b and the bisetrix of the coordinate system
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    b,c,d = [rot_matrix.dot(v) for v in [b,c,d]]  # rotate b, c and d
    return [0,0],b,c,d

def _create_hash(pointlist, lamda=1):
    """ Create geometric hash of given pointlist.
    The hash code contains point coordinates (p1_x, p1_y, p2_x), the relative
    positions of p3 and p4 in a local coordinate system with origin=p1 and
    (1,1)=p2 (and an optional feature descriptor). The incoming point list is
    sorted so that p3_x <= p4_x and p3_x + p4_x <= 1 in the local coordinate
    system. If there are several possibilities to sort the pointlist
    considering these constraints, break and return None.

    Args:
        pointlist (array_like): x and y coordinates of 4 (or n) points building a quad (shape (4,2)).
        lamda (float): Weight factor for the coordinates inside the hash.
    Returns:
        tuple: Hash of quad built from given pointlist.
        list: Sorted pointlist.
    """
    #TODO break if there is not a clear longest distance AB or if order of AB / CD cannot be defined clearly? add some tolerance for equality?
    if len(pointlist) != 4:
        raise NotImplementedError("Until now, only pointlists of length 4 can be hashed.")
    pointlist = np.array(pointlist)
    ### sort pointlist so that c_x <= d_x and c_x + d_x <= 1 in the local coordinate system
    idx_a, idx_b = _get_furthest_points(pointlist)
    idx_c, idx_d = [i for i in range(len(pointlist)) if i not in [idx_a,idx_b]]

    # build coordinate system with a=origin, b=k*(1,1) and vice versa
    a,b,c,d = _transform_coordinate_system(pointlist[[idx_a, idx_b, idx_c, idx_d]])
    a_valid = (c[0] + d[0] <= b[0])  # a is correct origin if c_x_rel + d_x_rel <= 1
    b2,a2,c2,d2 = _transform_coordinate_system(pointlist[[idx_b, idx_a, idx_c, idx_d]])
    b_valid = (c2[0] + d2[0] <= a2[0])  # b is correct origin if c_x_rel + d_x_rel <= 1

    if a_valid == b_valid:  # break if both AB and BA are valid orders or none of them
        return None, []  #TODO muss ueberhaupt abgebrochen werden wenn beide valid sind?
    sorted_indices = []
    if a_valid:
        sorted_indices.extend([idx_a, idx_b])
    else:
        sorted_indices.extend([idx_b, idx_a])
        b,c,d = [a2,c2,d2]
    # select c and d so that c_x <= d_x, set p3=c, p4=d
    if c[0] <= d[0]:
        sorted_indices.extend([idx_c, idx_d])
    else:
        sorted_indices.extend([idx_d, idx_c])
        c,d = [d,c]

    # build hash of normalized and scaled coordinates and relative positions of all points:
    hashcode = (lamda*pointlist[sorted_indices[0]][0],
                lamda*pointlist[sorted_indices[0]][1],
                lamda*pointlist[sorted_indices[1]][0],
                c[0]/b[0], c[1]/b[1], d[0]/b[0], d[1]/b[1])

    return hashcode, pointlist[sorted_indices]

class RandomSampler:
    def __init__(self, N, K, num_iterations):
        self.counter = 0
        self.K = K
        self.N = N
        self.maxiterations = num_iterations
    def done(self):
        return self.counter > self.maxiterations
    def __call__(self):
        self.counter += 1
        return random.sample(range(self.N), self.K)

class ExhaustiveSampler:
    # TODO make this work for K!=4
    def __init__(self, N, K):
        self.K = K
        self.N = N
        self.counter = 0
        self.combinations = np.mgrid[0:self.N,0:self.N,0:self.N,0:self.N].reshape((self.K,-1)).T
    def done(self):
        return self.counter > self.combinations.shape[0]
    def __call__(self):
        self.counter += 1
        return self.combinations[self.counter % self.combinations.shape[0],:]

def build_index(landmarks, sampler, lamda=1):
    """ Build an index of point samples with their geometric hash code as key.
    <sample_size> randomly choosen landmarks form a geometric object which is converted
    into a geometric hash. The returned dictionary contains geometric hashes for
    <num_samples> of those geometric objects.
    Args:
        landmarks (array_like): List of XY coordinates of landmarks with shape (N,2).
        sampler (object): Sampler object that provides a __call__ function for querying 4 indices, and a done() function
        lamda (int, optional): Weight factor for the coordinates inside the hash.
    Returns:
        dictionary containing geometric hashes as keys and according landmark coordinates as values.
    """
    hash2coords = {}
    while not sampler.done():
        indices = sampler()
        sample_coords = landmarks[indices]
        geo_hash, sorted_coords = _create_hash(sample_coords, lamda=lamda)
        if geo_hash is None:
            continue
        hash2coords[geo_hash] = sorted_coords
    return hash2coords

def find_similar_hashes(index_fixed, index_moving, radius):
    """ Find similar hashes in index_fixed and index_moving using a KDTree.
    Args:
        index_fixed (dictionary): Index with geometric hashes as keys and according fixed landmark coordinates as values.
        index_moving (dictionary): Index with geometric hashes as keys and according moving landmark coordinates as values.
        radius (float): Distance within which neighbors are returned.
    Returns:
        np.array: Matched XY coordinates of shape (M,2,2) with one match = [[x_fixed,y_fixed],[x_moving,y_moving]].
    """
    kdtree = KDTree(list(index_fixed.keys()), leaf_size=2, metric='minkowski')
    matches = []
    for hashcode, moving_coords in index_moving.items():
        neighbor_indices = kdtree.query_radius([hashcode], radius)[0]
        if len(neighbor_indices) > 0:
            neighbor_hashes = np.array(list(index_fixed.keys()))[neighbor_indices]
            neighbor_coords = [index_fixed[tuple(nh)] for nh in neighbor_hashes]
            for nc in neighbor_coords:
                matches.extend([[ni,si] for (ni,si) in zip(nc, moving_coords)])
    # get only unique matches
    if len(matches) > 0:
        matches = np.unique(np.array(matches), axis=0)
    return np.array(matches)

def _normalize_landmarks(landmarks1, landmarks2):
    """ Normalize given sets of landmarks to range (0,1).
    Normalize by subtracting mininmal x and y values and scaling with maximal
    range of coordinates in x or y direction.
    Args:
        landmarks1 (array_like): XY coordinates of 1st landmark set with shape (M,2).
        landmarks2 (array_like): XY coordinates of 2nd landmark set with shape (N,2).
    Returns:
        landmarks1, landmarks2, both normalized to range(0,1).
        offset (array-like): Offset subtracted from original landmarks.
        scale_factor (int): Used scale factor.
    """
    min_x, min_y = np.min([np.min(landmarks1,0), np.min(landmarks2,0)],0)
    max_x, max_y = np.max([np.max(landmarks1,0), np.max(landmarks2,0)],0)
    scale_factor = 1 / float(max(max_x-min_x, max_y-min_y))
    offset = np.array([min_x, min_y])
    landmarks1 = (np.array(landmarks1) - offset) * scale_factor
    landmarks2 = (np.array(landmarks2) - offset) * scale_factor
    return landmarks1, landmarks2, offset, scale_factor

def _homography_ransac(matches, residual_threshold=0.01):
    logger.info("mts vor RANSAC: {}".format(matches.shape))
    landmarks1, landmarks2 = matches[:,0], matches[:,1]
    findingPosSampleProbability = 0.999
    percentageOutliers = 0.997
    numIterations = int(math.ceil(math.log(1-findingPosSampleProbability)/math.log(1-(1-percentageOutliers))))
    model_robust, inliers = ransac((landmarks1, landmarks2), AffineTransform,
                                   min_samples=3, residual_threshold=residual_threshold, max_trials=numIterations)
    if inliers is None:
        logger.warning("Ransac found no inliers.")
        inliers = list(range(len(matches)))
    result = matches[inliers]
    logger.info("mts nach RANSAC: {}".format(result.shape))
    return result

def match(landmarks_fixed, landmarks_moving, sampler_fixed, sampler_moving, radius, lamda=1, ransac=None):
    """ Match landmarks_fixed with landmarks_moving.
    Args:
        landmarks_fixed (array_like): XY coordinates of fixed landmarks with shape (M,2).
        landmarks_moving (array_like): XY coordinates of moving landmarks with shape (N,2).
        sampler_fixed (object): Sampler object that provides a __call__ function for querying 4 indices, and a done() function.
        sampler_moving (object): Sampler object that provides a __call__ function for querying 4 indices, and a done() function.
        radius (float): Distance within which neighbors (=similar objects) are returned.
        lamda (float, optional): Weight factor for the coordinates inside the hash.
        ransac(float, optional): Residual_threshold for homography ransac. If None: Skip RANSAC.
    Returns:
        np.array: Matched XY coordinates of shape (K,2,2) with one match = [[x_fixed,y_fixed],[x_moving,y_moving]].
    """
    start = time.time()
    landmarks_fixed, landmarks_moving, offset, scale_factor = _normalize_landmarks(landmarks_fixed, landmarks_moving)
    logger.debug("runtime normalization: {}".format(time.time()-start))
    logger.debug("scale factor: {}".format(scale_factor))
    start = time.time()
    index_fixed = build_index(landmarks_fixed, sampler_fixed, lamda)
    logger.debug("runtime index fixed: {}".format(time.time()-start))
    start = time.time()
    index_moving = build_index(landmarks_moving, sampler_moving, lamda)
    logger.debug("runtime index moving: {}".format(time.time()-start))
    start = time.time()
    #TODO calculate proper radius (with regard to used scale_factor and assumed accuracy of prereg)?
    matches = find_similar_hashes(index_fixed, index_moving, radius=radius)
    logger.debug("runtime matching: {}".format(time.time()-start))
    if len(matches) == 0:
        logger.info("No matches found.")
        return []
    if ransac is not None:
        start = time.time()
        matches = _homography_ransac(matches, ransac)
        logger.debug("runtime ransac: {}".format(time.time()-start))
    # return matches in original scale
    return matches/scale_factor+offset
