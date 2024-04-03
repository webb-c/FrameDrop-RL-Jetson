from src.Reducto.util.data_loader import load_json
from src.Reducto.util.differencer import DiffProcessor


class DiffComposer:

    def __init__(self, differ_dict=None):
        self.differ_dict = differ_dict

    @staticmethod
    def from_jsonfile(jsonpath, differencer_types=None):
        """주어진 jsonpath에 위치한 파일을 읽어들어서, 사용하고자하는 difference type에 해당하는 threshold 값만 dict에 저장하여 DiffComposer 객체를 생성한다. 
        Args:
            jsonpath (str): threshold/{dataset_name}.json으로 저장된 threshold 파일의 경로
            differencer (List, optional): 4가지 feature중에서 특히 사용하고자하는 feature들이 저장된 list. Defaults to None.
        Returns:
            DiffComposer: feature, 그리고 각 feature마다의 threshold 값을 저장하고 있는 dict를 반환한다.
        """
        differencer_types = differencer_types or ['pixel', 'area', 'corner', 'edge']
        differ_dict = load_json(jsonpath)
        differencers = {
            feature: threshes
            for feature, threshes in differ_dict.items()
            if feature in differencer_types
        }
        # print("differencers: ", differencers)
        return DiffComposer(differencers)

    @staticmethod
    def placeholder(differencer_types=None):
        """저장된 파일 없이, 0으로 초기화된 difference threshold dict를 만든다. #TODO: threshold 학습 후에 저장은 어떻게 함?
        Args:
            differencer_types (List, optional): 4가지 feature중에서 특히 사용하고자하는 feature들이 저장된 list. Defaults to None.
        Returns:
            DiffComposer: feature, 그리고 각 feature마다의 threshold 값을 저장하고 있는 dict를 반환한다.
        """
        differencer_types = differencer_types or ['pixel', 'area', 'corner', 'edge']
        differencers = {
            feature: 0
            for feature in differencer_types
        }
        return DiffComposer(differencers)

    def new_thresholds(self, thresholds):
        """현재 diffComposer 객체가 가지고 있느 threshold dict의 값을 인자로 전달된 값으로 변경한다.
        Args:
            thresholds (dict): 새로 계산한 threshold값이 저장된 dict
        """
        for dp, threshes in thresholds.items():
            self.differ_dict[dp] = threshes

    def process_video(self, filepath, diff_vectors=None):
        """Diff vector 변수나 현재 diff가 저장되어있는 path에서 diff vector를 읽어온 뒤, threshold를 넘는 것만 선택하여 처리한다.
        Args:
            filepath (str): _description_
            diff_vectors (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        if diff_vectors:
            assert all([k in diff_vectors for k in self.differ_dict.keys()]), \
                'not compatible diff-vector list'
        else:
            diff_vectors = {
                k: self.get_diff_vector(k, filepath)
                for k in self.differ_dict.keys()
            }

        results = {}
        for differ_type, thresholds in self.differ_dict.items():
            diff_vector = diff_vectors[differ_type]
            result = self.batch_diff(diff_vector, thresholds)
            results[differ_type] = {
                'diff_vector': diff_vector,
                'result': result,
            }
        return results

    def process_video_in_run(self, threshold, diff_vectors=None):
        """전달한 diff_vectors에 대해, KNN이 정해준 threshold를 기반으로 보낼 프레임을 선택한다. 
        """
        result = self.batch_diff(diff_vectors, [threshold])
        return result

    @staticmethod
    def get_diff_vector(differ_type, filepath):
        """사용하려는 difference feature type으로 주어진 동영상의 difference vector를 계산한다.
        Args:
            differ_type (str): alias of using difference calculate feature
            filepath (str): video path (e.g., segment???)
        Returns:
            _type_: difference vector
        """
        differ = DiffProcessor.str2class(differ_type)() 
        return differ.get_diff_vector(filepath)

    @staticmethod
    def batch_diff(diff_vector, thresholds):
        """batch 단위로 처리.
        Args:
            diff_vector (_type_): _description_
            thresholds (_type_): _description_
        Returns:
            _type_: _description_
        """
        result = DiffProcessor.batch_diff_noobj(diff_vector, thresholds)
        return result
