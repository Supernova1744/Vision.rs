use crate::yolo_result::YOLOResult;
use crate::grpc::{
    YoloResult as ProtoYoloResult,
    Embedding as ProtoEmbedding,
    Bbox as ProtoBbox,
    KeypointSet as ProtoKeypointSet,
    Point2 as ProtoPoint2,
};

/// Converts the internal YOLO result to the gRPC proto message.
pub fn convert_yolo_result(internal: &YOLOResult) -> ProtoYoloResult {
    let mut proto_result = ProtoYoloResult::default();

    if let Some(internal_probs) = &internal.probs {
        let data = internal_probs.data().iter().copied().collect::<Vec<f32>>();
        let shape = internal_probs
            .data()
            .shape()
            .iter()
            .map(|&dim| dim as i32)
            .collect::<Vec<i32>>();
        proto_result.probs = Some(ProtoEmbedding { data, shape });
    }

    if let Some(internal_bboxes) = &internal.bboxes {
        let proto_bboxes = internal_bboxes
            .iter()
            .map(|bbox| ProtoBbox {
                xmin: bbox.xmin(),
                ymin: bbox.ymin(),
                width: bbox.width(),
                height: bbox.height(),
                id: bbox.id() as u32,
                confidence: bbox.confidence(),
            })
            .collect::<Vec<_>>();
        proto_result.bboxes = proto_bboxes;
    }

    if let Some(internal_keypoints) = &internal.keypoints {
        let proto_keypoints = internal_keypoints
            .iter()
            .map(|kp_set| {
                let proto_points = kp_set
                    .iter()
                    .map(|point| ProtoPoint2 {
                        x: point.x(),
                        y: point.y(),
                        confidence: point.confidence(),
                    })
                    .collect::<Vec<_>>();
                ProtoKeypointSet { points: proto_points }
            })
            .collect::<Vec<_>>();
        proto_result.keypoints = proto_keypoints;
    }

    proto_result.masks = internal.masks.clone().unwrap_or_default();
    proto_result
}
