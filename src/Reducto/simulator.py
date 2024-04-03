from itertools import product
from pathlib import Path
import os
import pickle
import time
import numpy as np

from src.Reducto.util.differencer import DiffComposer
from src.Reducto.util.data_loader import load_json
from src.Reducto.util.hashbuilder import HashBuilder, ThreshMap
from src.Reducto.util.utils import flatten, show_stats

class Simulator:

    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        self.actual_size = kwargs.get('actual_size', 2113284)
        self.root = Path(kwargs.get('result_root', 'data'))
        self.fps = kwargs.get('fps', 30)
        self.segment_duration = kwargs.get('segment_duration', 5.0)
        self.gpu_time = -1
        self.send_all = False
        self.name = 'none'
        self.network_logname = '!!!'

    def simulate(self, query_key='queries', network=None, network_name=None, video_scale=1,
                 verbose=False, dataset_names=None, metric=None, subsets='subsets'):
        """여러 데이터셋을 전달받아, simulation한 결과를 출력한다.
        Args:
            query_key (str, optional): _description_. Defaults to 'queries'.
            network (_type_, optional): _description_. Defaults to None.
            network_name (_type_, optional): _description_. Defaults to None.
            video_scale (int, optional): _description_. Defaults to 1.
            verbose (bool, optional): _description_. Defaults to False.
            dataset_names (_type_, optional): _description_. Defaults to None.
            metric (_type_, optional): _description_. Defaults to None.
            subsets (str, optional): _description_. Defaults to 'subsets'.
        Calls:
            #REVIEW: 
            self.eval() 
            self.frame_latency()
            reducto.utils.show_stats()
            reducto.utils.flatten()
        """
        evaluations = {'fractions': [], 'accuracies': [], 'selected_frames': []}
        latencies = {'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}
        other = {'true_sizes': [], 'sizes': []}

        for video in self.datasets:
            if dataset_names is not None and video['dataset'] not in dataset_names:
                continue
            print('video dataset name: ', video['dataset']) #여기까지 잘됨
            for query in video[query_key]:

                if metric is not None and query['metric'] != metric:
                    continue
                evaluation_summary = self.eval(video['dataset'], video[subsets], query, print_action=True)

                # print("===== eval result =====")
                # print(evaluation_summary)
                for item in evaluation_summary:
                    evaluations['fractions'].append(item['fraction'])
                    evaluations['accuracies'].append(item['evaluation'])
                    evaluations['selected_frames'].append(item['selected_frames'])

                if verbose:
                    new_evaluations = {'fractions': [], 'accuracies': []}
                    for item in evaluation_summary:
                        new_evaluations['fractions'].append(item['fraction'])
                        new_evaluations['accuracies'].append(item['evaluation'])
                    # print(f'{query["split"]:.2f},'
                    #       f'{np.mean(new_evaluations["fractions"]):.4f},'
                    #       f'{1-np.mean(new_evaluations["fractions"]):.4f},'
                    #       f'{np.mean(new_evaluations["accuracies"]):.4f}')

                    # cuts = [.40]
                    # df = pd.DataFrame(new_evaluations)
                    # frac_sent = df.quantile(cuts)['fractions'].to_list()[0]
                    # frac_filtered = 1 - frac_sent
                    # acc = df.quantile(cuts)['accuracies'].to_list()[0]

                    # print(f'{query["split"]:.2f},'
                    #       f'{frac_sent:.4f},'
                    #       f'{frac_filtered:.4f},'
                    #       f'{acc:.4f}')
                    print(video['dataset'])
                    show_stats(new_evaluations, ['fractions'])
                    # print('---')

                if not network:
                    continue
                network_summary = self.frame_latency(
                    evaluation_summary, network['bandwidth'], network['rtt'], network_name, scale=video_scale)
                latencies['latencies'].append(network_summary['latencies'])
                latencies['lat_cam'].append(network_summary['lat_cam'])
                latencies['lat_net'].append(network_summary['lat_net'])
                latencies['lat_inf'].append(network_summary['lat_inf'])
                other['sizes'].append(network_summary['sizes'])
                other['true_sizes'].append(network_summary['true_sizes'])

        # print(self.name)
        print('-' * 41)
        # evaluations['fractions'] = [1 - f for f in evaluations['fractions']]
        # show_stats(evaluations, ['fractions', 'accuracies'])
        evaluations['fractions'] = [f for f in evaluations['fractions']]
        print("fractions:", sum(evaluations["fractions"]) / len(evaluations["fractions"]))
        print("avg accuracy:", sum(evaluations["accuracies"]) / len(evaluations["accuracies"]))
        if not network:
            print()
            return
        latencies['latencies'] = flatten(latencies['latencies'])
        latencies['lat_cam'] = flatten(latencies['lat_cam'])
        latencies['lat_net'] = flatten(latencies['lat_net'])
        latencies['lat_inf'] = flatten(latencies['lat_inf'])
        # show_stats(latencies, ['latencies'])
        show_stats(latencies, ['latencies', 'lat_cam', 'lat_net', 'lat_inf'])

        other['sizes'] = flatten(other['sizes'])
        other['true_sizes'] = flatten(other['true_sizes'])
        size_mean = np.mean(other['sizes'])
        truesize_mean = np.mean(other['true_sizes'])
        print(f'      size: ({1 - size_mean / truesize_mean :.4f})')
        print()


    def eval(self, dataset, subsets, query, print_action):
        """*추상클래스* (하나의) 데이터셋과 쿼리를 전달받아서 evaluation을 진행한다.
        Args:
            dataset (_type_): _description_
            subsets (_type_): _description_
            query (_type_): _description_
            print_action (bool): _description_
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()


    def eval_path(self, seg):
        """해당 seg에 대해 evaluation을 수행한 결과가 저장된 path를 반환한다.
        Args:
            seg (List): [dataset, subset, segment]
        Returns:
            str: path (e.g., data\evaluation\jacksonhole\raw000\segment000.json)
        """
        return self.root / 'evaluation' / seg[0] / seg[1] / f'{seg[2]}.json'


    def diff_path(self, seg):
        """해당 seg에 대해 different를 계산한 결과가 저장된 path를 반환한다.
        Args:
            seg (List): [dataset, subset, segment]
        Returns:
            str: path (e.g., data\diff\jacksonhole\raw000\segment000.json)
        """
        return self.root / 'diff' / seg[0] / seg[1] / f'{seg[2]}.json'


    def infer_path(self, seg): #DEBUG:
        """해당 seg에 대해 inference(=YOLO?)를 수행한 결과가 저장된 path를 반환한다.
        Args:
            seg (List): [dataset, subset, segment]
        Returns:
            str: path (e.g., data\inference\jacksonhole\raw000\segment000.json)
        """
        return self.root / 'inference' / seg[0] / seg[1] / f'{seg[2]}.json'


    @staticmethod
    def get_segments(dataset, subsets, video_list_path=None):
        """현재 데이터셋의이 가지고 있는 subset들을 리스트로 저장하여 반환한다?
        Args:
            dataset (_type_): _description_
            subsets (_type_): _description_
            video_list_path (_type_, optional): _description_. Defaults to None.
        Calls:
            reducto.data_loader.load_json()
            reducto.utils.flatten()
        Returns:
            _type_: _description_
        """
        video_list_path = video_list_path or 'video_list.json'
        video_list = load_json(video_list_path)[dataset]
        segments = [(dataset, i[0], i[1].split('.')[0])
                    for i in flatten([list(product([s], video_list[s])) for s in subsets])]
        
        return segments

    def frame_latency(self, summary, bandwidth, rtt, network_name=None, divided_by=4, scale=1):
        """evaluation 결과를 전달받아서, 현재 가상으로 설정한 네트워크 환경에 맞추어 이론적인 latency를 계산하여 결과를 반환한다. 
        Args:
            summary (_type_): _description_
            bandwidth (_type_): _description_
            rtt (_type_): _description_
            network_name (_type_, optional): _description_. Defaults to None.
            divided_by (int, optional): _description_. Defaults to 4.
            scale (int, optional): _description_. Defaults to 1.
        Calls:
            self.get_segment_size()
        Returns:
            _type_: _description_
        """
        report = {'sizes': [], 'true_sizes': [], 'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}

        bandwidth_Bps = bandwidth / 8 * 1024 * 1024
        rtt_latency = rtt / 2 / 1000
        num_segment_frames = int(self.fps * self.segment_duration) // 5

        for seg in summary:
            selected_frames = seg['selected_frames']
            num_sent_frames = len(selected_frames)
            if self.send_all:
                selected_frames = []
                num_sent_frames = num_segment_frames
            size = self.get_segment_size(
                seg['dataset'], seg['subset'], seg['segment'],
                selected_frames, network_name, scale=scale)
            original_size = self.get_segment_size(seg['dataset'], seg['subset'], seg['segment'], log_name='true')
            report['sizes'].append(size)
            report['true_sizes'].append(original_size)
            for fid in range(num_sent_frames):
                sent_id = fid if self.send_all else selected_frames[fid]
                cam_latency = (1 / self.fps) * (num_segment_frames - sent_id // 5)
                # print(f'camera latency: fid={fid:03d}, lat={cam_latency:.4f}, s/d={sent_id // 5}')
                net_latency = (size / bandwidth_Bps + rtt_latency) / float(divided_by)
                inf_latency = self.gpu_time * (fid // 5 + 1)
                latency = cam_latency + net_latency + inf_latency
                report['latencies'].append(latency)
                report['lat_cam'].append(cam_latency)
                report['lat_net'].append(net_latency)
                report['lat_inf'].append(inf_latency)
        return report


class Reducto(Simulator):

    def __init__(self, datasets, communicator, video_processor, jetson_mode, debug_mode, **kwargs):
        super().__init__(datasets, **kwargs)
        self.inference_fps = 40
        self.profiling_fps = 40
        self.camera_diff_fps = 30
        self.inference_time = 1 / self.inference_fps
        self.profiling_time = 1 / self.profiling_fps
        self.camera_diff_time = 1 / self.camera_diff_fps
        self.len_bootstrapping = 5
        self.gpu_time = 1 / 40
        self.network_logname = 'reducto'
        self.name = 'reducto'
        self.communicator = communicator
        self.video_processor = video_processor
        self.frame_shape = self.video_processor.frame_shape
        self.jetson_mode = jetson_mode
        self.debug_mode = debug_mode


    #! IT'S MINE
    def make_hashmap(self, conf, dataset, subsets):
        if os.path.exists(conf["cluster_path"]):
            with open(conf["cluster_path"], 'rb') as f:
                threshmap_init_dict = pickle.load(f)
        else:
            evals, diff_vectors, diff_results = self.load_result(dataset, subsets, conf["differ"], conf["metric"])
            threshmap_init_dict = HashBuilder().generate_threshmap(evals, diff_vectors, target_acc=conf["target_acc"], safe_zone=conf["safe_zone"])
            with open(conf["cluster_path"], 'wb') as f:
                pickle.dump(threshmap_init_dict, f)

        thresh_map = ThreshMap(threshmap_init_dict[conf["differ"]])
        
        return thresh_map
    
    #! IT'S MINE
    def run(self, conf, dataset, subsets):
        dataset_mapping = {
            'JK-1' : 'JK',
            'JK-2' : 'JK',
            'SD-1' : 'SD',
            'SD-2' : 'SD',
            'JN-1' : 'JN'
        }
        train_dataset = dataset_mapping[dataset]
        
        differ = DiffComposer.from_jsonfile(conf['differ_dict_path'], conf['differ_types'])
        thresh_map = self.make_hashmap(conf, train_dataset, subsets)
        
        p = Path(conf['dataset_dir']) / dataset / subsets[0]
        segments_path = [f for f in sorted(p.iterdir()) if f.match('segment???.mp4')]
        
        diff_results = []
        fraction_change = []
        for i, segment in enumerate(segments_path):
            diff_vector = differ.get_diff_vector(differ_type=conf["differ"], filepath=segment)
            if i == 0:
                time.sleep((1.0 / self.fps) * (len(diff_vector)+1))
            
            thresh, distance = thresh_map.get_thresh(diff_vector)
            distance = np.sum(distance)

            if distance > conf['distance']:
                selected_frames = list(range(1, len(diff_vector)+1))
                diff_result = {
                    'selected_frames' : selected_frames,
                    'fraction': len(selected_frames) / (len(diff_vector) + 1)
                }
                fraction_change.append(1.0) 
            else:
                diff_result = differ.process_video_in_run(thresh, diff_vector) 
                fraction_change.append(diff_result['fraction'])
                selected_frames = diff_result['selected_frames']
                
            diff_results.append(diff_result)
            
            if self.debug_mode:
                print("KNN predicted threshold: ", thresh)
                print("vector distance sum: ", distance)
                print(diff_result)
            
            ### Send Selected_frames ###
            self.video_processor.update_segment(segment_path=str(segment))
            
            for i in range(1, len(diff_vector)+1):
                if self.debug_mode:
                    print("segment:", segment, "\tidx:", i)
                if i in selected_frames:
                    if self.jetson_mode:
                        self.communicator.get_message()
                        self.communicator.send_message("action")
                        self.communicator.get_message()
                        self.communicator.send_message(self.frame_shape)
                    ret = self.video_processor.read_video(skip=False)
                    if not ret:
                        break
                else:
                    ret = self.video_processor.read_video(skip=True)
                    if not ret:
                        break
            
        return diff_results, fraction_change
