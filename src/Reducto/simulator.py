from functools import partial
from itertools import product
from pathlib import Path
import multiprocessing as mp
import functools
from pprint import pprint

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.differencer import DiffComposer
from utils.codec import get_video_size
from utils.data_loader import load_evaluation, load_diff_vector, load_diff_result, load_json, load_inference, \
    dump_json
from utils.evaluator import MetricComposer
from utils.hashbuilder import HashBuilder, ThreshMap
from utils.utils import flatten, show_stats
import os
import pickle
from utils.model import Segment, Inference, InferenceResult, DiffVector, FrameEvaluation

_target_acc = 0.90
_tinyyolo_acc = 0.60

class Simulator:

    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        self.actual_size = kwargs.get('actual_size', 2113284)
        self.video_root = Path(kwargs.get('video_root', '../../DATASET/test/split/'))
        self.root = Path(kwargs.get('result_root', 'data'))
        print(self.root)
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

        print(self.name)
        print('-' * 41)
        #TODO: WHY?????
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


    def load_result(self, dataset, subsets, differ, metric):
        """dataset을 전달받아서, 각 segment에 대해 eval, diff_vector, result 결과를 저장된 경로로부터 불러와서 리스트로 저장해 반환한다. 
        Args:
            dataset (_type_): _description_
            subsets (_type_): _description_
            differ (_type_): _description_
            metric (_type_): _description_
        Calls:
            self.get_segments()
            self.eval_path()
            self.diff_path()
            reducto.data_loader.load_evaluation()
            reducto.data_loader.load_diff_vector()
            reducto.data_loader.load_diff_result()
        Returns:
            _type_: _description_
        """
        segments = self.get_segments(dataset, subsets)
        evals = [load_evaluation(self.eval_path(seg), differ, metric) for seg in segments]
        diff_vectors = [load_diff_vector(self.diff_path(seg), differ) for seg in segments]
        diff_results = [load_diff_result(self.diff_path(seg), differ) for seg in segments]
        assert len(evals) == len(diff_vectors) == len(diff_results)
        
        return evals, diff_vectors, diff_results


    def load_inference(self, dataset, subsets): #DEBUG:
        """dataset을 전달받아서, 각 segment에 대해 inference 결과를 저장된 경로로부터 불러와서 리스트로 저장해 반환한다.
        Args:
            dataset (str): 데이터셋
            subsets (List): 해당 데이터셋이 포함하고 있는 subsets
        Calls:
            self.get_segments()
            self.inter_path()
            reducto.data_loader.load_inference()
        Returns:
            List: 각 segment에 대한 inference 결과가 저장된 리스트
        """
        segments = self.get_segments(dataset, subsets)
        seg = segments[0]
        print(seg)
        print(self.infer_path(seg))
        inference = [load_inference(self.infer_path(seg)) for seg in segments]
        
        return inference


    def get_segment_size(self, dataset, subset, segment, selected_frames=None, log_name=None, scale=1):
        """해당 video에 대하여, 선택한 frame이 주어졌을 때 video의 size(=frame_len)이 얼마나 줄어드는지 계산하여 반환한다.
        Args:
            dataset (_type_): _description_
            subset (_type_): _description_
            segment (_type_): _description_
            selected_frames (_type_, optional): _description_. Defaults to None.
            log_name (_type_, optional): _description_. Defaults to None.
            scale (int, optional): _description_. Defaults to 1.
        Calls:
            reducto.codec.get_video_size()
        Returns:
            _type_: _description_
        """
        selected_frames = selected_frames or []
        selected_frames = selected_frames if len(selected_frames) > 0 else None
        video_path = self.video_root / dataset / subset / f'{segment}.mp4'
        size = get_video_size(video_path, selected_frames, log_name, scale=scale)
        
        return size

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


class Optimal(Simulator):

    def __init__(self, datasets, typ, classes, **kwargs):
        super().__init__(datasets, **kwargs)
        self.gpu_time = 1 / 40
        classes_str = ':'.join([str(c) for c in classes])
        self.network_logname = f'optimal_{typ}_{classes_str}'
        self.type = typ
        self.classes = classes
        self.name = f'optimal {typ} ({classes})'
        if self.type == 'coco':
            self.evaluator = MetricComposer.from_json([{'type': 'coco', 'class': self.classes}])
        else:
            self.evaluator = None


    def eval(self, dataset, subsets, query, print_action=False):
        """하나의 데이터셋을 전달받아 각 query에 대해 inference결과가 있다면 불러오고, 존재하지 않으면 inference결과를 경로에 저장하고, 반환한다.
        Args:
            dataset (_type_): _description_
            subsets (_type_): _description_
            query (_type_): _description_
        Calls:
            self.get_segments()
            self.load_inference()
            reducto.data_loader.load_json()
            reducto.data_loader.dump_json()
        Returns:
            _type_: _description_
        """
        segments = self.get_segments(dataset, subsets)
        target_acc = query['target_acc']
        inferences = self.load_inference(dataset, subsets)
        output_path = f'data/optimal-{target_acc:.1f}/{dataset}_{self.network_logname}.json'
        if Path(output_path).exists():
            loaded_summary = load_json(output_path)
            summary = [item for item in loaded_summary if item['dataset'] == dataset and item['subset'] in subsets]
        else:
            with multiprocessing.Pool() as pool:
                result = pool.map(partial(self.select_frames, target_acc=target_acc), inferences)
            summary = [
                {
                    'dataset': segments[index][0],
                    'subset': segments[index][1],
                    'segment': segments[index][2],
                    'fraction': len(res['selected_frames']) / len(inferences[index].keys()),
                    'evaluation': sum(res['scores']) / len(res['scores']),
                    'selected_frames': res['selected_frames'],
                }
                for res, index in zip(result, range(len(segments)))
            ]
            dump_json(summary, output_path, mkdir=True)
        return summary

    def select_frames(self, inference, target_acc):
        """특정 inference 결과에 대해, target_acc와 지금까지 처리한 결과의 acc를 비교하여 각 frame을 선택할지 말지를 결정하여 선택한 프레임과 프레임별 accuracy의 변화를 기록한 리스트를 반환한다. 
        Args:
            inference (_type_): _description_
            target_acc (_type_): _description_
        Calls:
            Optimal.count_objects()
            Optimal.get_counting_score()
            Optimal.get_tagging_score()
            self.get_detection_score()
        Returns:
            _type_: _description_
        """
        frame_ids = list(inference.keys())
        summary = [
            {
                'fid': int(fid),
                'count': Optimal.count_objects(inference[fid], self.classes),
                'inference': inference[fid],
            }
            for fid in frame_ids
        ]
        selected_frames = [summary[0]['fid']]
        scores = [1.0]
        for fid in range(1, len(summary)):
            last_selected_fid = selected_frames[-1]
            if self.type == 'counting':
                score = Optimal.get_counting_score(summary[fid]['count'], summary[last_selected_fid]['count'])
            elif self.type == 'tagging':
                score = Optimal.get_tagging_score(summary[fid]['count'], summary[last_selected_fid]['count'])
            elif self.type == 'coco':
                score = self.get_detection_score(summary[fid]['inference'], summary[last_selected_fid]['inference'])
            new_scores = scores + [score]
            if sum(new_scores) / len(new_scores) >= target_acc:
                # if score >= target_acc:
                scores.append(score)
            else:
                selected_frames.append(fid)
                scores.append(1.0)
        return {
            'selected_frames': selected_frames,
            'scores': scores,
        }


    @staticmethod
    def count_objects(frame_inference, classes):
        """선택한 class들 안에 있는 class의 길이를 inference 결과로부터 추출하여 반환한다.
        Args:
            frame_inference (_type_): _description_
            classes (_type_): _description_
        Returns:
            _type_: _description_
        """
        count = len([c for c in frame_inference['detection_classes'] if c in classes])
        return count


    @staticmethod
    def get_counting_score(count1, count2):
        """두 개의 counting 결과를 비교하여, 동일하면 1, 다르다면 정의한 방식을 통해 score를 반환한다. 
        Args:
            count1 (_type_): _description_
            count2 (_type_): _description_
        Returns:
            _type_: _description_
        """
        if count1 == count2:
            return 1.0
        return (max(count1, count2) - abs(count1 - count2)) / max(count1, count2)


    @staticmethod
    def get_tagging_score(count1, count2):
        """두 개의 counting 결과를 비교하여 둘 중 하나라도 tagging이 되어있지 않다면 o점, 아니면 1점을 반환한다.
        Args:
            count1 (_type_): _description_
            count2 (_type_): _description_
        Returns:
            _type_: _description_
        """
        if count1 == 0 or count2 == 0:
            return 0.0
        return 1.0


    def get_detection_score(self, inference1, inference2):
        """두 inference 결과를 전달받아서 object detection score를 비교한다. #TODO:
        Args:
            inference1 (_type_): _description_
            inference2 (_type_): _description_
        Calls:
            MetricComposer.evaluate_single_frame() #REVIEW:
        Returns:
            _type_: _description_
        """
        result = self.evaluator.evaluate_single_frame(inference1, inference2)
        if len(self.classes) == 0:
            name = 'mAP-all'
        else:
            name = f'mAP-{":".join(str(i) for i in self.classes)}'
        return result[name]


class Reducto(Simulator):

    def __init__(self, datasets, **kwargs):
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
    def run(self, conf, dataset, subsets, writer):
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
        segments = self.get_segments(dataset, subsets, video_list_path='test_video_list.json')
        
        length = len(segments)
        diff_results = []
        fraction_change = []
        p = Path(conf['dataset_dir']) / dataset / subsets[0]
        segments_path = [f for f in sorted(p.iterdir()) if f.match('segment???.mp4')]
        
        for i, segment in enumerate(segments_path):
            diff_vector = differ.get_diff_vector(differ_type=conf["differ"], filepath=segment)
            thresh, distance = thresh_map.get_thresh(diff_vector)
            distance = np.sum(distance)

            if distance > conf['distance']:
                selected_frames = list(range(1, len(diff_vector)+1))
                diff_result = {
                    'selected_frames' : selected_frames,
                    'fraction': len(selected_frames) / (len(diff_results) + 1)
                }
                fraction_change.append(1.0) ### tensor borad 기록하도록 수정하기
                writer.add_scalar("Network/Fraction", diff_result['fraction'], i)
            else:
                # fraction = diff_results[index][differ][thresh]['fraction']
                # evaluation = evals[index][differ][thresh][metric]
                diff_result = differ.process_video_in_run(thresh, diff_vector) 
                fraction_change.append(diff_result['fraction'])
                writer.add_scalar("Network/Fraction", diff_result['fraction'], i)
            
            diff_results.append(diff_result)
            
            if conf['debug']:
                print("KNN predicted threshold: ", thresh)
                print("vector distance sum: ", distance)
                print(diff_result)
            
        return diff_results, fraction_change
        
    
    # 테스트시에만 로컬에서 작동, 실제 실험에서는 평가 자체는 추후에 만든 영상을 가지고 따로 inference후 F1 score 계산.
    #! IT'S MINE
    def test(self, conf, segments, model, diff_results, evaluator, writer):
        # pipeline running
        f1_score_change = []
        pbar = tqdm(total=len(segments))
        for i, segment in enumerate(segments):
            # -- segment ---------------------------------------------------
            segment_record = Segment.find_or_save(segment.parent.name, segment.name)

            # -- inference -------------------------------------------------
            inference_record = Inference.objects(
                segment=segment_record,
                model=model.name,
            ).first()
            if inference_record:
                inference = inference_record.to_json()
            else:
                inference = model.infer_video(segment)
                inference_record = Inference(
                    segment=segment_record,
                    model=model.name,
                    result=[InferenceResult.from_json(inf) for _, inf in inference.items()],
                )
                inference_record.save()
            dump_json(inference, f'data/inference/{conf["dataset"]}/{segment.parent.name}/{segment.stem}.json', mkdir=True)
            if conf["debug"]: print("In segment", segment, "\nInference: ", inference)

            # -- evaluation ------------------------------------------------
            frame_pairs = evaluator.get_frame_pairs(inference, diff_results[i], run=True)

            per_frame_evaluations = {}
            for metric in evaluator.keys:
                metric_evaluations = FrameEvaluation.objects(segment=segment_record, evaluator=metric)
                pairs = [(me.ground_truth, me.comparision) for me in metric_evaluations]
                pairs_pending = [p for p in frame_pairs if p not in pairs]
                with mp.Pool() as pool:
                    eval_f = functools.partial(evaluator.evaluate_frame_pair, inference=inference, metric=metric)
                    metric_evaluations_new = pool.map(eval_f, pairs_pending)
                pair_evaluations_new = {
                    pair: evaluation
                    for pair, evaluation in zip(pairs_pending, metric_evaluations_new)
                }
                for pair, evaluation in pair_evaluations_new.items():
                    frame_evaluation_record = FrameEvaluation(
                        segment=segment_record,
                        model=model.name,
                        evaluator=metric,
                        ground_truth=pair[0],
                        comparision=pair[1],
                        result=evaluation[metric],
                    )
                    frame_evaluation_record.save()
                for me in metric_evaluations:
                    if not per_frame_evaluations.get((me.ground_truth, me.comparision), None):
                        per_frame_evaluations[(me.ground_truth, me.comparision)] = {}
                    per_frame_evaluations[(me.ground_truth, me.comparision)][metric] = me.result
                for pair, evaluation in pair_evaluations_new.items():
                    if not per_frame_evaluations.get(pair, None):
                        per_frame_evaluations[pair] = {}
                    per_frame_evaluations[pair][metric] = evaluation[metric]

            evaluations = evaluator.evaluate(inference, diff_results[i], per_frame_evaluations, segment, run=True)
            
            f1_score = evaluations["mAP-all"]
            writer.add_scalar("Network/f1_score", f1_score, i)
            f1_score_change.append(f1_score)
            
            dump_json(evaluations, f'data/evaluation/{conf["dataset"]}/{segment.parent.name}/{segment.stem}.json', mkdir=True)
            
            if conf["debug"]: print("In segment", segment, "\nevaluations: ", evaluations)
            
            pbar.update()
            
        return f1_score_change


    def eval(self, dataset, subsets, query, print_action=False):
        """Reducto 모델의 evaluation을 위한 코드
        Args:
            dataset (_type_): _description_
            subsets (_type_): _description_
            query (_type_): _description_
        Calls:
            self.get_segments()
            self.load_result()
            HashBuilder().generate_threshmap #REVIEW:
            ThreshMap().get_thresh() #REVIEW:
        Returns:
            _type_: _description_
        """
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        target_acc = query['target_acc']
        dist_thresh = query['distance']
        safe_zone = query['safe']
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)
        length = len(evals)
        boot = True
        profiled_indexes = []
        summary = {}
        for index in range(length):
            # starting
            summary[index] = {
                'start': index * self.segment_duration,
                'taken': (index + 1) * self.segment_duration,
            }
            # bootstrapping
            if len(profiled_indexes) < self.len_bootstrapping:
                boot = True
                profiled_indexes.append(index)
                distance = -1
                fraction = 1.0
                evaluation = 1.0
                summary[index]['sent'] = summary[index]['taken']
                summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                # print("[bootstrapping phase] current len: ", len(profiled_indexes))
            # dynamic phase
            else:
                profiled_available = [
                    p_index for p_index in profiled_indexes
                    if summary[p_index]['prof_done'] < summary[index]['taken']
                ]
                # if there is not enough profiled segments, still sends everything
                if len(profiled_available) < self.len_bootstrapping:
                    profiled_indexes.append(index)
                    boot = True
                    distance = -1
                    fraction = 1.0
                    evaluation = 1.0
                    summary[index]['sent'] = summary[index]['taken']
                    summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                    summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                    # print("[bootstrapping phase] current len: ", len(profiled_indexes))
                # good to start doing dynamic adoption
                else:
                    boot = False
                    # print("[dynamic phase]")
                    threshmap_init_dict = HashBuilder().generate_threshmap(
                        [evals[i] for i in profiled_indexes],
                        [diff_vectors[i] for i in profiled_indexes],
                        target_acc=target_acc, safe_zone=safe_zone)
                    thresh_map = ThreshMap(threshmap_init_dict[differ])
                    thresh, distance = thresh_map.get_thresh(diff_vectors[index][differ])
                    distance = np.sum(distance)
                    # predicted threshold value and distance threshold checking
                    print("KNN predicted threshold: ", thresh)
                    print("vector distance sum: ", distance)
                    if distance > dist_thresh:
                        print("so far distance")
                        profiled_indexes.append(index)
                        fraction = 1.0
                        evaluation = 1.0
                        summary[index]['sent'] = summary[index]['taken'] + self.camera_diff_time
                        summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                        summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                    else:
                        fraction = diff_results[index][differ][thresh]['fraction']
                        evaluation = evals[index][differ][thresh][metric]
                        summary[index]['sent'] = summary[index]['taken'] + self.camera_diff_time
                        summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time * fraction
                        summary[index]['prof_done'] = -1
                
                print("index:", index, "fraction: ", fraction)

            summary[index]['dataset'] = segments[index][0]
            summary[index]['subset'] = segments[index][1]
            summary[index]['segment'] = segments[index][2]
            summary[index]['profiling?'] = int(index in profiled_indexes)
            summary[index]['diff_vector'] = diff_vectors[index][differ]
            summary[index]['distance'] = distance
            summary[index]['fraction'] = fraction
            summary[index]['evaluation'] = evaluation
            if fraction == 1.0:
                summary[index]['selected_frames'] = []
            elif fraction < 1.0:
                selected_frames = diff_results[index][differ][thresh]['selected_frames']
                summary[index]['selected_frames'] = selected_frames
            
            if print_action and not boot:
                print("in index ", index, "processed frame num ", len(summary[index]['selected_frames']))
                #print("*selected frames\n", summary[index]['selected_frames'])
        
        summary_list = [summary[v] for v in range(self.len_bootstrapping, length)]
        return summary_list


class ReductoOptimal(Simulator):

    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        self.gpu_time = 1 / 40
        self.network_logname = 'reducto_optimal'
        self.name = 'reducto optimal'

    def eval(self, dataset, subsets, query, print_action=False):
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        target_acc = query['target_acc']
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)

        summary = []
        for index in range(len(evals)):
            good_threshes = [th for th, acc_dict in evals[index][differ].items()
                             if acc_dict[metric] > target_acc]
            good_fracs = [(th, diff_results[index][differ][th]['fraction'],
                           evals[index][differ][th][metric]) for th in good_threshes]
            good_fracs_sorted = sorted(good_fracs, key=lambda th_acc: th_acc[1])
            if len(good_fracs_sorted) == 0:
                optimal_fraction = 1.0
                optimal_evaluation = 1.0
            else:
                optimal_fraction = good_fracs_sorted[0][1]
                optimal_evaluation = good_fracs_sorted[0][2]
            summary.append({
                'fraction': optimal_fraction,
                'evaluation': optimal_evaluation,
                'dataset': segments[index][0],
                'subset': segments[index][1],
                'segment': segments[index][2],
                'selected_frames': [],
            })
            
            if print_action:
                print("in index ", index, "processed frame num ", len(summary[index]['selected_frames']))
                print("*selected frames\n", summary[index]['selected_frames'])
            
            
        return summary