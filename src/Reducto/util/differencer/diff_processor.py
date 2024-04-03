import configparser

import cv2
import imutils
import numpy as np

from src.Reducto.util.video_processor import VideoProcessor_Reducto


class DiffProcessor:

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        """
        :param thresh: threshold, frame with diff above which will be sent
        :param fraction: only support first and second, force the fraction
        :param dataset: for loading external config
        """
        self.feature = 'none'
        self.fraction = fraction
        self.thresh = thresh
        self.section = self.get_section(dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_diff_vector(self, video_path):
        """주어진 video_path에 대해, 각 frame별로 특정 feature에 대한 difference를 계산한다. 
        Args:
            video_path (str): 데이터셋 위치
        Returns:
            diff_values: 각 frame간의 feature difference를 계산한 결과가 저장된ㄷ 리스트(=diff vector)를 반환한다.
        """
        diff_values = []
        with VideoProcessor_Reducto(video_path) as video:
            prev_frame = next(video)
            prev_frame = self.get_frame_feature(prev_frame)
            for frame in video:
                frame = self.get_frame_feature(frame)
                diff_value = self.cal_frame_diff(frame, prev_frame)
                diff_values.append(diff_value)
                prev_frame = frame
        return diff_values

    def process_video(self, video_path):
        """주어진 video_path에 위치한 데이터셋을 사용하여, 각 frame별로 feature를 뽑고 feature간의 difference를 계산하여, difference가 threshold보다 높으면 선택하고, 아니면 선택하지 않는다. 
        Args:
            video_path (str): 데이터셋 위치
        Returns:
            dict: processing한 결과를 저장하여 반환한다.
        """
        selected_frames = [1]
        estimations = [1.0]
        with VideoProcessor_Reducto(video_path) as video:
            prev_frame = next(video)
            prev_feat = self.get_frame_feature(prev_frame)
            for frame in video:
                feat = self.get_frame_feature(frame)
                dis = self.cal_frame_diff(feat, prev_feat)
                if dis > self.thresh:
                    selected_frames.append(video.index)
                    prev_feat = feat
                    estimations.append(1.0)
                else:
                    estimations.append((self.thresh - dis) / self.thresh)
            total_frames = video.index
        result = {
            'feature': self.feature,
            'thresh': self.thresh,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames),
            'num_total_frames': total_frames,
            'fraction': len(selected_frames) / total_frames,
            'estimation': sum(estimations) / len(estimations)
        }
        return result

    @staticmethod
    def batch_diff(diff_value, diff_processors):
        """diff vector를 누적합으로 바꾼 뒤, 각 diff_processor에 대해 threhold를 넘으면 선택하고 넘지 않으면 버린다. 
        Args:
            diff_value (_type_): _description_
            diff_processors (_type_): _description_
        Returns:
            dict: 처리 결가
        """
        diff_integral = np.cumsum([0.0] + diff_value).tolist()
        diff_results = {}
        total_frames = 1 + len(diff_value)
        for dp in diff_processors:
            threshold = dp.thresh
            selected_frames = [1]
            estimations = [1.0]
            last, current = 1, 2
            while current < total_frames:
                diff_delta = diff_integral[current] - diff_integral[last]
                if diff_delta >= threshold:
                    selected_frames.append(current)
                    last = current
                    estimations.append(1.0)
                else:
                    estimations.append((threshold - diff_delta) / threshold)
                current += 1
            diff_results[dp.name] = DiffProcessor._format_result(selected_frames, total_frames, estimations)
        return diff_results

    @staticmethod
    def batch_diff_noobj(diff_value, thresholds):
        """DiffProcessor 없이, 직접 threshold를 사용하여 계산한다. 
        Args:
            diff_value (_type_): _description_
            thresholds (_type_): _description_
        Returns:
            dict: 처리결과
        """
        diff_integral = np.cumsum([0.0] + diff_value).tolist()
        diff_results = {}
        total_frames = 1 + len(diff_value)
        
        if len(thresholds) == 1 :
            thresh = thresholds[0]
            selected_frames = [1]
            estimations = [1.0]
            last, current = 1, 2
            while current < total_frames:
                diff_delta = diff_integral[current] - diff_integral[last]
                if diff_delta >= thresh:
                    # 변화가 심한 프레임이니까 선택
                    selected_frames.append(current)
                    last = current
                    estimations.append(1.0)
                else:
                    estimations.append((thresh - diff_delta) / thresh)
                current += 1
            diff_results = DiffProcessor._format_result(selected_frames, total_frames, estimations)
        
        else:
            for thresh in thresholds:
                # thresh가 여러개인이뉴: 가능한 방식을 모두 도전해보려고.
                # print("thresh: ", thresh)
                selected_frames = [1]
                estimations = [1.0]
                last, current = 1, 2
                while current < total_frames:
                    diff_delta = diff_integral[current] - diff_integral[last]
                    if diff_delta >= thresh:
                        selected_frames.append(current)
                        last = current
                        estimations.append(1.0)
                    else:
                        estimations.append((thresh - diff_delta) / thresh)
                    current += 1
                diff_results[thresh] = DiffProcessor._format_result(selected_frames, total_frames, estimations)
        
        return diff_results

    def cal_frame_diff(self, frame, prev_frame):
        """Calculate the different between frames."""
        raise NotImplementedError()

    def get_frame_feature(self, frame):
        """Extract feature of frame."""
        raise NotImplementedError()

    @staticmethod
    def get_section(dataset):
        """diff_config.ini 파일에 저장된 dataset 별, 사용하는 feature 별 저장된 hyper parameter를 읽어온다.
        Args:
            dataset (str): dataset name
        Returns:
            _type_: config. (e.g., PIXEL_THRESH_LOW_BOUND, AREA_BLUR_RAD, EDGE_BLUR_VAR, ...)
        """
        config = configparser.ConfigParser()
        config.read('model/Reducto/config/diff_config.ini')
        return config[dataset if dataset and dataset in config else 'default']

    def _load_section(self, section):
        return

    def __str__(self):
        return self.name

    @staticmethod
    def _format_result(selected_frames, total_frames, estimations):
        """Processing한 결과를 반환하는 format을 지정한다. 
        Args:
            selected_frames (_type_): _description_
            total_frames (_type_): _description_
            estimations (_type_): _description_
        Returns:
            _type_: _description_
        """
        return {
            # 'fps': total_frames / complete_time if complete_time != 0 else -1,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames),
            'num_total_frames': total_frames,
            'fraction': len(selected_frames) / total_frames,
            'estimation': sum(estimations) / len(estimations)
        }

    @staticmethod
    def str2class(feature):
        """alias를 사용해서 해당되는 Class를 반환한다. 
        Args:
            feature (str): low-level feature for calculate difference
        Returns:
            Type[DiffProcessor]: class
        """
        return {
            'pixel': PixelDiff,
            'area': AreaDiff,
            'edge': EdgeDiff,
            'corner': CornerDiff,
            'hist': HistDiff,
            'hog': HOGDiff,
            'sift': SIFTDiff,
            'surf': SURFDiff,
        }[feature]


class PixelDiff(DiffProcessor):

    feature = 'pixel'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        return frame

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_diff = cv2.absdiff(frame, prev_frame)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.threshold(frame_diff, self.pixel_thresh_low_bound,
                                   255, cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.pixel_thresh_low_bound = section.getint('PIXEL_THRESH_LOW_BOUND', 21)


class AreaDiff(DiffProcessor):

    feature = 'area'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.area_blur_rad, self.area_blur_rad),
                                self.area_blur_var)
        return blur

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_delta = cv2.absdiff(frame, prev_frame)
        thresh = cv2.threshold(frame_delta, self.area_thresh_low_bound, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if not contours:
            return 0.0
        return max([cv2.contourArea(c) / total_pixels for c in contours])

    def _load_section(self, section):
        self.area_blur_rad = section.getint('AREA_BLUR_RAD', 11)
        self.area_blur_var = section.getint('EDGE_BLUR_VAR', 0)
        self.area_thresh_low_bound = section.getint('AREA_THRESH_LOW_BOUND', 21)


class EdgeDiff(DiffProcessor):

    feature = 'edge'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.edge_blur_rad, self.edge_blur_rad),
                                self.edge_blur_var)
        edge = cv2.Canny(blur, self.edge_canny_low, self.edge_canny_high)
        return edge

    def cal_frame_diff(self, edge, prev_edge):
        total_pixels = edge.shape[0] * edge.shape[1]
        frame_diff = cv2.absdiff(edge, prev_edge)
        frame_diff = cv2.threshold(frame_diff, self.edge_thresh_low_bound, 255,
                                   cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.edge_blur_rad = section.getint('EDGE_BLUR_RAD', 11)
        self.edge_blur_var = section.getint('EDGE_BLUR_VAR', 0)
        self.edge_canny_low = section.getint('EDGE_CANNY_LOW', 101)
        self.edge_canny_high = section.getint('EDGE_CANNY_HIGH', 255)
        self.edge_thresh_low_bound = section.getint('EDGE_THRESH_LOW_BOUND', 21)


class CornerDiff(DiffProcessor):

    feature = 'corner'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'
        self._load_section(self.section)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner = cv2.cornerHarris(gray, self.corner_block_size,
                                  self.corner_ksize, self.corner_k)
        corner = cv2.dilate(corner, None)
        return corner

    def cal_frame_diff(self, corner, prev_corner):
        total_pixels = corner.shape[0] * corner.shape[1]
        frame_diff = cv2.absdiff(corner, prev_corner)
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.corner_block_size = section.getint('CORNER_BLOCK_SIZE', 5)
        self.corner_ksize = section.getint('CORNER_KSIZE', 3)
        self.corner_k = section.getfloat('CORNER_K', 0.05)


class HistDiff(DiffProcessor):

    feature = 'histogram'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_frame_feature(self, frame):
        nb_channels = frame.shape[-1]
        hist = np.zeros((self.hist_nb_bins * nb_channels, 1), dtype='float32')
        for i in range(nb_channels):
            hist[i * self.hist_nb_bins: (i + 1) * self.hist_nb_bins] = \
                cv2.calcHist(frame, [i], None, [self.hist_nb_bins], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    def cal_frame_diff(self, frame, prev_frame):
        return cv2.compareHist(frame, prev_frame, cv2.HISTCMP_CHISQR)

    def _load_section(self, section):
        self.hist_nb_bins = section.getint('HIST_NB_BINS', 32)


class HOGDiff(DiffProcessor):

    feature = 'HOG'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to speed up
        gray = cv2.resize(gray, (self.hog_resize, self.hog_resize))
        hog = feature.hog(gray, orientations=self.hog_orientations,
                          pixels_per_cell=(self.hog_pixel_cell, self.hog_pixel_cell),
                          cells_per_block=(self.hog_cell_block, self.hog_cell_block)
                          ).astype('float32')
        return hog

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis

    def _load_section(self, section):
        self.hog_resize = section.getint('HOG_RESIZE', 512)
        self.hog_orientations = section.getint('HOG_ORIENTATIONS', 10)
        self.hog_pixel_cell = section.getint('HOG_PIXEL_CELL', 5)
        self.hog_cell_block = section.getint('HOG_CELL_BLOCK', 2)


class SIFTDiff(DiffProcessor):

    feature = 'SIFT'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_frame_feature(self, frame):
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        des = (np.mean(des, axis=0).astype('float32')
               if des is not None else np.zeros(128))
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis


class SURFDiff(DiffProcessor):

    feature = 'SURF'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        super().__init__(thresh, fraction, dataset)
        self._load_section(self.section)
        self.name = f'{self.feature}-{self.thresh}-{self.fraction}'

    def get_frame_feature(self, frame):
        surf = cv2.xfeatures2d.SURF_create()
        surf.setUpright(True)
        surf.setHessianThreshold(self.surf_hessian_thresh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = surf.detectAndCompute(gray, None)
        des = np.zeros(128) if des is None else np.mean(des, axis=0).astype('float32')
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis

    def _load_section(self, section):
        self.surf_hessian_thresh = section.getint('SURF_HESSIAN_THRESH', 400)
