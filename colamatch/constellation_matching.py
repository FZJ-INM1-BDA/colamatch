from sklearn.neighbors import KDTree

def _sample_landmarks(landmarks, sample_size=4):
    """ Sample n random landmarks out of given landmark list.
    Args:
        landmarks (array_like): List of x and y coordinates of available landmarks.
        sample_size (int): Number of required samples.
    Returns:
        list: XY coordinates of SORTED sampled landmarks with shape (sample_size,2).
    """
    if sample_size != 4:
        raise NotImplementedError("Until now, only sample_size=4 is implemented.")
    pass

def _create_hash(pointlist, lamda=1):
    """ Create geometric hash of given pointlist.
    The hash code contains point coordinates (p1_x, p1_y, p2_x), the relative positions of p3 and p4 in a
    local coordinate system with origin=p1 and (1,1)=p2 (and an optional feature descriptor).
    The incoming point list is sorted so that p3_x <= p4_x and p3_x + p4_x <= 1 in the local coordinate
    system. If there are several possibilities to sort the pointlist considering these constraints,
    break and return None.
    Args:
        pointlist (array_like): x and y coordinates of 4 (or n) points building a quad (shape (4,2)).
        lamda (float): Weight factor for the coordinates inside the hash.
    Returns:
        tuple: Hash of quad built from given pointlist.
        list: Sorted pointlist.
    """
    pass

def build_index(landmarks, num_samples, sample_size=4, lamda=1):
    """ Build an index of point samples with their geometric hash code as key.
    <sample_size> randomly choosen landmarks form a geometric object which is converted
    into a geometric hash. The returned dictionary contains geometric hashes for
    <num_samples> of those geometric objects.
    Args:
        landmarks (array_like): List of XY coordinates of landmarks with shape (N,2).
        num_samples (int): Number of geometric objects to add to the index.
        sample_size (int, optional): Number of sampled landmarks to create one geometric object.
        lamda (int, optional): Weight factor for the coordinates inside the hash.
    Returns:
        dictionary containing geometric hashes as keys and according landmark coordinates as values.
    """
    hash2coords = {}
    for i in range(num_samples):
        sample_coords = _sample_landmarks(landmarks, sample_size=sample_size)
        geo_hash, sorted_coords = _create_hash(sample_coords, lamda=lamda)
        hash2coords[geo_hash] = sorted_coords
    return hash2coords

def find_similar_hashes(index_fixed, index_moving, radius):
    """ Find similar hashes in index_fixed and index_moving using a KDTree.
    Args:
        index_fixed (dictionary): Index with geometric hashes as keys and according fixed landmark coordinates as values.
        index_moving (dictionary): Index with geometric hashes as keys and according moving landmark coordinates as values.
        radius (float): Distance within which neighbors are returned.
    Returns:
        List of matching coordinates.
    """
    kdtree = KDTree(list(index_fixed.keys()), leaf_size=2, metric='minkowski')
    matches = []
    for hashcode, moving_coords in index_moving.items():
        #NOTE Vorsicht, hashcode is tuple!
        neighbor_hashes = kdtree.query_radius([hashcode], radius)
        neighbor_coords = [index_fixed[nh] for nh in neighbor_hashes]
        matches.extend([[ni,si] for (ni,si) in zip(neighbor_coords, [moving_coords]*len(neighbor_coords))])
    return matches

def match(landmarks_fixed, landmarks_moving, num_samples, radius, sample_size=4, lamda=1):
    """ Match landmarks_fixed with landmarks_moving. """
    index_fixed = build_index(landmarks_fixed, num_samples, sample_size, lamda)
    index_moving = build_index(landmarks_moving, num_samples, sample_size, lamda)
    matches = find_similar_hashes(index_fixed, index_moving, radius=radius)
    return matches
