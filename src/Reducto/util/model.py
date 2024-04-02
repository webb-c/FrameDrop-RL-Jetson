from mongoengine import *


class Segment(Document):
    """monoDB document: 어떤 segment에 대한 정보"""
    subset = StringField(required=True)
    name = StringField(required=True)

    num_frames = IntField(default=-1)
    width = IntField(default=-1)
    height = IntField(default=-1)
    size = IntField(default=-1)

    meta = {
        'indexes': [{
            'fields': ('subset', 'name'),
            'unique': True
        }]
    }

    @staticmethod
    def find_or_save(subset, name):
        record = Segment.objects(
            subset=subset,
            name=name,
        ).first()
        if record:
            return record
        record = Segment(
            subset=subset,
            name=name,
        )
        record.save()
        return record


class InferenceResult(EmbeddedDocument):
    """monoDB document: 어떤 segment의 frame 하나에 대한 inference 결과"""
    num_detections = IntField()
    detection_scores = ListField()
    detection_classes = ListField()
    detection_boxes = ListField()

    @staticmethod
    def from_json(json_data):
        return InferenceResult(
            num_detections=json_data['num_detections'],
            detection_scores=json_data['detection_scores'],
            detection_classes=json_data['detection_classes'],
            detection_boxes=json_data['detection_boxes'],
        )

    def to_json(self):
        return {
            'num_detections': self.num_detections,
            'detection_scores': self.detection_scores,
            'detection_classes': self.detection_classes,
            'detection_boxes': self.detection_boxes,
        }


class Inference(Document):
    """monoDB document: 어떤 segment를 특정 model을 사용하여 inference한 결과"""
    segment = ReferenceField(Segment, required=True)
    model = StringField(required=True)
    result = ListField(EmbeddedDocumentField(InferenceResult), required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'model'),
            'unique': True
        }]
    }

    def to_json(self):
        return {
            i + 1: self.result[i].to_json()
            for i in range(len(self.result))
        }


class DiffVector(Document):
    """monoDB document: 어떤 segment의 대해, 특정 방식으로 계산된 diff vector 리스트"""
    segment = ReferenceField(Segment, required=True)
    differencer = StringField(required=True)
    vector = ListField(required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'differencer'),
            'unique': True
        }]
    }


class MotionVector(Document):
    """monoDB document: 어떤 segment의 대해, 특정 방식으로 계산된 motion vector 리스트"""
    segment = ReferenceField(Segment, required=True)
    motioner = StringField(required=True)
    vector = ListField(required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'motioner'),
            'unique': True
        }]
    }


class FrameEvaluation(Document):
    """monoDB document: 어떤 segment의 대해, predit와 ground_trurh를 사용하여 주어진 metric으로 평가한 결과"""
    segment = ReferenceField(Segment, required=True)
    model = StringField(required=True)
    evaluator = StringField(required=True)
    ground_truth = IntField(required=True)
    comparision = IntField(required=True)
    result = FloatField(required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'model', 'evaluator', 'ground_truth', 'comparision'),
            'unique': True
        }]
    }
