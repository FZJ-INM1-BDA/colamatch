from abc import ABC, abstractmethod
import numpy as np
import logging
import cv2

class Matcher(ABC):
    """ Abstract Matcher class. """

    def __init__(self, max_matching_distance):
        """ Initialize. """
        self.max_matching_distance = max_matching_distance
        self.logger = logging.getLogger(__package__)

    @abstractmethod
    def match(self):
        """ Abstract matching method. """
        pass

    def get_candidates(self, L_fixed, L_moving):
        """ Get landmarks with distance < max_matching_distance.

        Args:
            L_fixed (array-like): Landmarks in fixed image (x,y coordinates).
            L_moving (array-like): Landmarks in moving image (x,y coordinates).
        Returns:
            Array with index pairs of L_fixed and L_moving defining match candidates (shape (N,2)).
        """
        possibleMatches = np.zeros((0,3))
        for i,k0 in enumerate(L_fixed):
            for j,k1 in enumerate(L_moving):
                value = 1-(np.sqrt(((np.array(k0[:2])-np.array(k1[:2]))**2).sum()) / float(self.max_matching_distance))
                if value >= 0:
                    possibleMatches = np.append(possibleMatches, [[i,j,value]], axis=0)
        return possibleMatches

    #TODO needed???
    def homography_ransac(self, l_fixed, l_moving, candidates):
        """ Perform homography ransac. """
        pass


class TemplateMatcher(Matcher):
    """ TemplateMatcher class. """

    def __init__(self, max_matching_distance, patchsize, cv2_matching_method=cv2.TM_CCOEFF_NORMED):
        """ Initialize. """
        super(TemplateMatcher, self).__init__(max_matching_distance)
        self.patchsize = patchsize
        self.cv2_matching_method = cv2_matching_method

    def match(self, L_fixed, L_moving, img_fixed, img_moving, candidates=None, num_best_matches=1):
        """ Perform template matching.
        Args:
            L_fixed (array-like): Landmarks in fixed image (x,y coordinates).
            L_moving (array-like): Landmarks in moving image (x,y coordinates).
            img_fixed (array-like): Fixed image.
            img_moving (array-like): Moving image.
            candidates (array-like, optional): Indices of L_fixed and L_moving defining match candidates.
            num_best_matches (int, optional): Number of best matches returned per landmark.
        Returns:
            Array of matched landmark index pairs of L_fixed and L_moving (shape (N,2)).
        """
        if candidates is None:  # every match is possible
            candidates = np.array(np.where(np.ones((len(L_fixed),len(L_moving))))).T
        scores = []
        for candidate in candidates:
            scores.append(self._template_matching(L_fixed[candidate[0]], L_moving[candidate[1]], img_fixed, img_moving))
        scores = np.array(scores)
        # for squared differences, small scores are best -> sorted scores must be reversed later
        sort_scores = -1 if self.cv2_matching_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else 1
        template_matches = []
        for fixed_idx in np.unique(candidates[:,0]):
            relevant_candidates = np.where(candidates[:,0] == fixed_idx)
            match_indices = np.argsort(scores[relevant_candidates], axis=None)[::sort_scores][-num_best_matches:]
            template_matches.extend(np.vstack(([fixed_idx]*len(match_indices), candidates[relevant_candidates][match_indices][:,1])).T)
        for moving_idx in np.unique(candidates[:,1]):
            relevant_candidates = np.where(candidates[:,1] == moving_idx)
            match_indices = np.argsort(scores[relevant_candidates], axis=None)[::sort_scores][-num_best_matches:]
            template_matches.extend(np.vstack((candidates[relevant_candidates][match_indices][:,0], [moving_idx]*len(match_indices))).T)
        return np.unique(template_matches, axis=0)

    def _template_matching(self, l_fixed, l_moving, img_fixed, img_moving):
        # extract template patches of equal shape
        patch_width = int(min(self.patchsize//2, l_fixed[0], img_fixed.shape[1]-l_fixed[0], l_moving[0], img_moving.shape[1]-l_moving[0]))
        patch_height = int(min(self.patchsize//2, l_fixed[1], img_fixed.shape[0]-l_fixed[1], l_moving[1], img_moving.shape[0]-l_moving[1]))
        fixed_patch = img_fixed[l_fixed[1]-patch_height:l_fixed[1]+patch_height, l_fixed[0]-patch_width:l_fixed[0]+patch_width]
        moving_patch = img_moving[l_moving[1]-patch_height:l_moving[1]+patch_height, l_moving[0]-patch_width:l_moving[0]+patch_width]
        # call cv2 TemplateMatching
        return cv2.matchTemplate(fixed_patch, moving_patch, self.cv2_matching_method)[0,0]


class TriangleMatcher(Matcher):
    """ TriangleMatcher class. """

    def match(self, l_fixed, l_moving, candidates=None):
        """ Perform triangle matching. """
        matched_indices = self._get_triangle_matches(candidates)
        matched_indices = self._triangle_ransac(matched_indices)
        #TODO save landmarks (+images) as member variables or pass to every method?

    def _get_triangle_matches(self, candidates):
        pass

    def _triangle_ransac(self, candidates):
        pass

    def _is_mirrored(self, ABC, DEF, L_fixed, L_moving):
        """ Check if Triangles ABC and DEF are mirrored.

        Args:
            ABC (array_like): Three indices of l_fixed defining a triangle.
            DEF (array_like): Three indices of l_moving defining a triangle.
            L_fixed (array_like): Array of all landmarks of fixed image with at least columns x and y.
            L_moving (array_like): Array of all landmarks of moving image with at least columns x and y.
        Returns:
            True if ABC and DEF are mirrored, else False.
        """
        ABC_Coords = np.array([L_moving[ABC[0]][:2], L_moving[ABC[1]][:2], L_moving[ABC[2]][:2], L_moving[ABC[0]][:2]])
        DEF_Coords = np.array([L_fixed[DEF[0]][:2], L_fixed[DEF[1]][:2], L_fixed[DEF[2]][:2], L_fixed[DEF[0]][:2]])
        sumABC = 0
        sumDEF = 0
        for i in range(1, 4):
            sumABC += (ABC_Coords[i][1] + ABC_Coords[i-1][1]) * (ABC_Coords[i][0] - ABC_Coords[i-1][0])
            sumDEF += (DEF_Coords[i][1] + DEF_Coords[i-1][1]) * (DEF_Coords[i][0] - DEF_Coords[i-1][0])
        return np.sign(sumABC) != np.sign(sumDEF)
