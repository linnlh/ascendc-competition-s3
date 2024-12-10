/**
* @file main.cpp
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    int64_t num_batches = 64;
    int64_t spatial_dimension = 64;
    int64_t num_classes = 16;
    int64_t num_selected_indices = 4096;
    int64_t center_point_box = 0;

    std::vector<int64_t> shape_boxes{num_batches, spatial_dimension, 4};
    std::vector<int64_t> shape_scores{num_batches, num_classes, spatial_dimension};
    std::vector<int64_t> shape_max_output_boxes_per_class {1};
    std::vector<int64_t> shape_iou_threshold{1};
    std::vector<int64_t> shape_score_threshold{1};
    std::vector<int64_t> shape_selected_indices {num_selected_indices, 3};
    opDesc.center_point_box = 0;
    opDesc.AddInputTensorDesc(ACL_FLOAT, shape_boxes.size(), shape_boxes.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, shape_scores.size(), shape_scores.data(), format);
    opDesc.AddInputTensorDesc(ACL_INT32, shape_max_output_boxes_per_class.size(), shape_max_output_boxes_per_class.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, shape_iou_threshold.size(), shape_iou_threshold.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, shape_score_threshold.size(), shape_score_threshold.data(), format);
    opDesc.AddOutputTensorDesc(ACL_INT32, shape_selected_indices.size(), shape_selected_indices.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_boxes.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_scores.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/input_max_output_boxes_per_class.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    ReadFile("../input/input_iou_threshold.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3));
    ReadFile("../input/input_score_threshold.bin", fileSize, runner.GetInputBuffer<void>(4), runner.GetInputSize(4));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("../output/output.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));

    INFO_LOG("Write output success");
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        }
        else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit("../scripts/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
