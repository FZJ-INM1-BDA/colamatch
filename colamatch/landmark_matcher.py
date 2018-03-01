from abc import ABC, abstractmethod
import cv2

class Matcher(ABC):
    """ Abstract Matcher class. """

    def __init__(self, max_matching_distance):
        """ Initialize. """
        self.max_matching_distance = max_matching_distance

    @abstractmethod
    def match(self):
        """ Abstract match() method. """
        pass

    def get_candidates(l_fixed, l_moving):
        """ Get landmarks with distance < max_matching_distance. """
        pass

    def homography_ransac(l_fixed, l_moving, candidates):
        """ Perform homography ransac. """
        pass

class TemplateMatcher(Matcher):
    """ TemplateMatcher class. """

    def __init__(self, max_matching_distance, patchsize, cv2_matching_method=cv2.TM_CCOEFF_NORMED):
        super(TemplateMatcher, self).__init__(max_matching_distance)
        self.patchsize = patchsize
        self.cv2_matching_method = cv2_matching_method

    def match(self, l_fixed, l_moving, img_fixed, img_moving, candidates=None):
        """ Perform template matching. """
        print("template matching with max_matching_distance:", self.max_matching_distance)


class TriangleMatcher(Matcher):
    """ TriangleMatcher class. """

    def match(self, l_fixed, l_moving, candidates=None):
        """ Perform triangle matching. """
        print("triangle matching with max_matching_distance:", self.max_matching_distance)
        matched_indices = self._get_triangle_matches(candidates)
        matched_indices = self._triangle_ransac(matched_indices)
        #TODO save landmarks (+images) as member variables or pass to every method?

    def _get_triangle_matches(self, candidates):
        pass

    def _triangle_ransac(self, candidates):
        pass


if __name__ == "__main__":
    tm = TemplateMatcher(5, 7)
    tm.match(1,2,3,4,5)
