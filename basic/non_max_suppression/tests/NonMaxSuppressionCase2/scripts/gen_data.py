import os
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np

np.random.seed(123)


def gen_onnx_model(shape_boxes,
                   shape_scores,
                   shape_max_output_boxes_per_class,
                   shape_iou_threshold,
                   shape_score_threshold,
                   center_point_box,
                   shape_selected_indices):
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, shape_boxes)
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, shape_scores)
    max_output_boxes_per_class = helper.make_tensor_value_info("max_output_boxes_per_class", TensorProto.INT64,
                                                               shape_max_output_boxes_per_class)
    iou_threshold = helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, shape_iou_threshold)
    score_threshold = helper.make_tensor_value_info("score_threshold", TensorProto.FLOAT, shape_score_threshold)
    selected_indices = helper.make_tensor_value_info("selected_indices", TensorProto.INT64, shape_selected_indices)
    node_def = helper.make_node('NonMaxSuppression',
                                inputs=['boxes',
                                        "scores",
                                        "max_output_boxes_per_class",
                                        "iou_threshold",
                                        "score_threshold"],
                                outputs=['selected_indices'],
                                center_point_box=center_point_box
                                )
    graph = helper.make_graph(
        [node_def],
        "test_NonMaxSuppression_case_1",
        inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        outputs=[selected_indices]
    )

    model = helper.make_model(graph, producer_name="onnx-NonMaxSuppression_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_NonMaxSuppression_v11.onnx")


def run_mode(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
    # 加载ONNX模型
    model_path = 'test_NonMaxSuppression_v11.onnx'  # 替换为你的ONNX模型路径
    sess = ort.InferenceSession(model_path)

    input_boxes_name = sess.get_inputs()[0].name
    input_scores_name = sess.get_inputs()[1].name
    input_max_output_boxes_per_class_name = sess.get_inputs()[2].name
    input_iou_threshold_name = sess.get_inputs()[3].name
    input_score_threshold_name = sess.get_inputs()[4].name
    output_name = sess.get_outputs()[0].name

    input_boxes = boxes
    input_scores = scores
    input_max_output_boxes_per_class = max_output_boxes_per_class
    input_iou_threshold = iou_threshold
    input_score_threshold = score_threshold

    outputs = sess.run([output_name], {
        input_boxes_name: input_boxes,
        input_scores_name: input_scores,
        input_max_output_boxes_per_class_name: input_max_output_boxes_per_class,
        input_iou_threshold_name: input_iou_threshold,
        input_score_threshold_name: input_score_threshold
    })
    return outputs[0]


def gen_golden_data_simple():
    num_batches = 64
    spatial_dimension = 64
    num_classes = 16
    max_output_boxes_per_class = 4
    num_selected_indices = 4096

    center_point_box = 0  # 0 or 1

    shape_boxes = [num_batches, spatial_dimension, 4]
    shape_scores = [num_batches, num_classes, spatial_dimension]
    shape_max_output_boxes_per_class = [1]
    shape_iou_threshold = [1]
    shape_score_threshold = [1]
    shape_selected_indices = [num_selected_indices, 3]

    input_boxes = np.random.uniform(-1, 1, shape_boxes).astype(np.float32)
    input_scores = np.random.uniform(0, 1, shape_scores).astype(np.float32)
    input_max_output_boxes_per_class = np.random.uniform(
        max_output_boxes_per_class,
        max_output_boxes_per_class,
        shape_max_output_boxes_per_class
    ).astype(np.int64)
    input_iou_threshold = np.random.uniform(0, 1, shape_iou_threshold).astype(np.float32)  # 取值范围[0,1]
    input_score_threshold = np.random.uniform(0, 1, shape_score_threshold).astype(np.float32)

    gen_onnx_model(shape_boxes,
                   shape_scores,
                   shape_max_output_boxes_per_class,
                   shape_iou_threshold,
                   shape_score_threshold,
                   center_point_box,
                   shape_selected_indices)
    golden = run_mode(input_boxes,
                      input_scores,
                      input_max_output_boxes_per_class,
                      input_iou_threshold,
                      input_score_threshold)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_max_output_boxes_per_class = input_max_output_boxes_per_class.astype(np.int32)
    golden = golden.astype(np.int32)
    input_boxes.tofile("./input/input_boxes.bin")
    input_scores.tofile("./input/input_scores.bin")
    input_max_output_boxes_per_class.tofile("./input/input_max_output_boxes_per_class.bin")
    input_iou_threshold.tofile("./input/input_iou_threshold.bin")
    input_score_threshold.tofile("./input/input_score_threshold.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
