#pragma once

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>  //OpenVINO Inference
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

#define NMS_UNION 1
#define NMS_MIN  2

typedef struct FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float area;

	float landmarks[10];
};

class LFFD {
public:
	LFFD(const std::string& model_path, const std::string& targetDeviceName="CPU",
		int scale_num=8,bool  enablePerformanceReport_=false);
	~LFFD();

	int detect(cv::Mat& img, std::vector<FaceInfo>& face_lis,cv::Size resized_size,
		float score_threshold = 0.6, float nms_threshold = 0.4, int top_k = 10000,
		std::vector<int> skip_scale_branch_list = {});

private:
	void preprocess(const cv::Mat& image, float* buffer, std::vector<cv::Mat> input_channels) const;

	void generateBBox(std::vector<FaceInfo>& collection, float * score_map, float * box_map, float score_threshold,
		int fea_w, int fea_h, int cols, int rows, int scale_id);

	void get_topk_bbox(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, int topk);

	void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
		float threshold, int type = NMS_UNION);
private:
	std::string modelPath;
	bool enablePerformanceReport;

	InferenceEngine::InferencePlugin plugin;
	InferenceEngine::CNNNetwork network;
	InferenceEngine::ExecutableNetwork executableNetwork;
	InferenceEngine::InferRequest request;
	InferenceEngine::CNNNetReader netReader;

	cv::Size inputLayerSize;
	int inputLayerChannel;

	int num_output_scales;
	int image_w;
	int image_h;

	std::vector<float> receptive_field_list;
	std::vector<float> receptive_field_stride;
	std::vector<float> receptive_field_center_start;
	std::vector<float> constant;

	std::vector<std::string> output_blob_names;

};
