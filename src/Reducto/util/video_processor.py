import cv2


class VideoProcessor_Reducto:

    def __init__(self, video_path, frame_limit=None):
        """Video Processor init
        Args:
            video_path (str): 사용할 영상의 경로
            frame_limit (_type_, optional): 사용할 프레임 개수의 제한. Defaults to None.
        """
        self.video_path = str(video_path)
        self.frame_limit = frame_limit
        self.frame_count = 0
        self.index = 0
        self.progress_bar = None

    def __enter__(self):
        """cap을 정의하고, 프레임 개수를 정의한다.
        Returns:
            VideoProcessor
        """
        self.cap = cv2.VideoCapture(self.video_path)
        # self.frame_count = int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT)))
        self.frame_count = self.cap.get(int(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_limit and self.frame_limit > 0:
            self.frame_count = min(self.frame_count, self.frame_limit)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """사용한 자원(cap)을 반환한다."""
        self.cap.release()
        # cv2.destroyAllWindows()

    def __iter__(self):
        """반복가능한 객체로서 자기자신을 반환한다."""
        return self

    def __next__(self):
        """ frame_count에 도달했거나, 더이상 읽을 프레임이 없을 때 반복을 멈추며, 이외의 경우에는 frame을 반환한다."""
        if self.index == self.frame_count:
            raise StopIteration

        _ret, _frame = self.cap.read()
        if not _ret:
            raise StopIteration

        self.index += 1
        if self.progress_bar:
            self.progress_bar.update()
        return _frame

    def __len__(self):
        """frame 개수를 반환한다. """
        return self.frame_count
