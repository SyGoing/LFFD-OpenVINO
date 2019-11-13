
#include "OpenVINO_LFFD.h"

const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
const float norm_vals[3] = { 0.0078431373, 0.0078431373, 0.0078431373 };


static std::string fileNameNoExt(const std::string& filepath) {
	auto pos = filepath.rfind('.');
	if (pos == std::string::npos) return filepath;
	return filepath.substr(0, pos);
}

LFFD::LFFD(const std::string& model_path, const std::string& targetDeviceName,int scale_num,bool  enablePerformanceReport_):
	num_output_scales(scale_num), enablePerformanceReport(enablePerformanceReport_),modelPath(model_path)
{
	std::string xml_name;
	std::string bin_name;

	if (num_output_scales == 5) {
		xml_name= modelPath + "/10_320_20L_5scales_v2_deploy.xml";
		bin_name = modelPath + "/10_320_20L_5scales_v2_deploy.bin";
		receptive_field_list = { 20, 40, 80, 160, 320 };
		receptive_field_stride = { 4, 8, 16, 32, 64 };
		receptive_field_center_start = { 3, 7, 15, 31, 63 };

		for (int i = 0; i < receptive_field_list.size(); i++) {
			constant.push_back(receptive_field_list[i] / 2);
		}

		output_blob_names = { "softmax0","conv8_3_bbox",
		                                  "softmax1","conv11_3_bbox",
										  "softmax2","conv14_3_bbox",
		                                  "softmax3","conv17_3_bbox",
		                                  "softmax4","conv20_3_bbox" };
	}
	else if (num_output_scales == 8) {
		xml_name = modelPath + "/symbol_10_560_25L_8scales_v1_deploy.xml";
		bin_name = modelPath + "/symbol_10_560_25L_8scales_v1_deploy.bin";
		receptive_field_list = { 15, 20, 40, 70, 110, 250, 400, 560 };
		receptive_field_stride = { 4, 4, 8, 8, 16, 32, 32, 32 };
		receptive_field_center_start = { 3, 3, 7, 7, 15, 31, 31, 31 };

		for (int i = 0; i < receptive_field_list.size(); i++) {
			constant.push_back(receptive_field_list[i] / 2);
		}

		output_blob_names = { "softmax0","conv8_3_bbox",
			"softmax1","conv10_3_bbox",
			"softmax2","conv13_3_bbox",
			"softmax3","conv15_3_bbox",
			"softmax4","conv18_3_bbox",
			"softmax5","conv21_3_bbox",
			"softmax6","conv23_3_bbox",
		    "softmax7","conv25_3_bbox" };
	}
	
	plugin = InferenceEngine::PluginDispatcher()
		.getPluginByDevice(targetDeviceName);
	if (enablePerformanceReport) {
		plugin.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
						   InferenceEngine::PluginConfigParams::YES} });
	}
	netReader.ReadNetwork(xml_name);
	netReader.ReadWeights(bin_name);
	network = netReader.getNetwork();
	InferenceEngine::InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
	inputLayerSize = cv::Size(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]);
	inputLayerChannel = inputInfo->getTensorDesc().getDims()[1];

	executableNetwork = plugin.LoadNetwork(network, {});
	request = executableNetwork.CreateInferRequest();
}

LFFD::~LFFD()
{

}

int LFFD::detect(cv::Mat& img, std::vector<FaceInfo>& face_list, cv::Size input_size,
	float score_threshold, float nms_threshold, int top_k, std::vector<int> skip_scale_branch_list)
{

	if (img.empty()) {
		std::cout << "image is empty ,please check!" << std::endl;
		return -1;
	}

	image_h = img.rows;
	image_w = img.cols;

    cv::Mat in;
    cv::resize(img,in,input_size);
    float ratio_w=(float)image_w/ input_size.width;
    float ratio_h=(float)image_h/ input_size.height;


	//resize net input and network
	if (input_size.height != inputLayerSize.height ||
		input_size.width != inputLayerSize.width) {

		inputLayerSize.height = input_size.height;
		inputLayerSize.width=   input_size.width;

		auto input_shapes = network.getInputShapes();
		std::string input_name;
		InferenceEngine::SizeVector input_shape;
		std::tie(input_name, input_shape) = *input_shapes.begin();
		input_shape[2] = inputLayerSize.height;
		input_shape[3] = inputLayerSize.width;
		input_shapes[input_name] = input_shape;
		network.reshape(input_shapes);
		executableNetwork = plugin.LoadNetwork(network, {});
		request = executableNetwork.CreateInferRequest();
	}
	//preprocess
	std::vector<cv::Mat> input_channels;
	InferenceEngine::Blob::Ptr input = request.GetBlob(network.getInputsInfo().begin()->first);
	auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
	preprocess(in, static_cast<float*>(buffer), input_channels);

	//Infer
	request.Infer();

	//Post Process
	std::vector<FaceInfo> bbox_collection;
	for (int i = 0; i <num_output_scales; i++) {

		InferenceEngine::Blob::Ptr conf_Blob = request.GetBlob(output_blob_names[2 * i]);
		InferenceEngine::Blob::Ptr reg_Blob = request.GetBlob(output_blob_names[2 * i + 1]);

		InferenceEngine::SizeVector shape_conf=conf_Blob->dims();  //ÔõÃ´shapeÊÇµ¹×ÅµÄÄØ w h c n
		InferenceEngine::SizeVector shape_reg = reg_Blob->dims();

		float * conf_ptr= conf_Blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
		float *reg_ptr= reg_Blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

		generateBBox(bbox_collection, conf_ptr, reg_ptr, score_threshold,
			shape_conf[0], shape_conf[1], in.cols, in.rows, i);
	}
	std::vector<FaceInfo> valid_input;
	get_topk_bbox(bbox_collection, valid_input, top_k);
	nms(valid_input, face_list, nms_threshold);

    for(int i=0;i<face_list.size();i++){
        face_list[i].x1*=ratio_w;
        face_list[i].y1*=ratio_h;
        face_list[i].x2*=ratio_w;
        face_list[i].y2*=ratio_h;

        float w,h,maxSize;
        float cenx,ceny;
        w=face_list[i].x2-face_list[i].x1;
        h=face_list[i].y2-face_list[i].y1;

		maxSize = w > h ? w : h;
        cenx=face_list[i].x1+w/2;
        ceny=face_list[i].y1+h/2;
        face_list[i].x1=cenx-maxSize/2>0? cenx - maxSize / 2:0;
        face_list[i].y1=ceny-maxSize/2>0? ceny - maxSize / 2:0;
        face_list[i].x2=cenx+maxSize/2>image_w? image_w-1: cenx + maxSize / 2;
        face_list[i].y2=ceny+maxSize/2> image_h? image_h-1: ceny + maxSize / 2;

    }
	return 0;
}

void LFFD::preprocess(const cv::Mat& image, float* buffer, std::vector<cv::Mat> input_channels) const
{
	cv::Mat sample_float;
	for (int i = 0; i < inputLayerChannel; ++i) {
		cv::Mat channel(inputLayerSize.height, inputLayerSize.width, CV_32FC1, buffer+i*inputLayerSize.area());
		input_channels.push_back(channel);
	}
	image.convertTo(sample_float, CV_32FC3, 0.0078431373f, -127.5f * 0.0078431373f);
	cv::split(sample_float, input_channels);
}

void LFFD::generateBBox(std::vector<FaceInfo>& bbox_collection, float* score_map, float* box_map, float score_threshold, int fea_w, int fea_h, int cols, int rows, int scale_id)
{
	float* RF_center_Xs = new float[fea_w];
	float* RF_center_Xs_mat = new float[fea_w * fea_h];
	float* RF_center_Ys = new float[fea_h];
	float* RF_center_Ys_mat = new float[fea_h * fea_w];

    for (int x = 0; x < fea_w; x++) {
		RF_center_Xs[x] = receptive_field_center_start[scale_id] + receptive_field_stride[scale_id] * x;
	}
	for (int x = 0; x < fea_h; x++) {
		for (int y = 0; y < fea_w; y++) {
			RF_center_Xs_mat[x * fea_w + y] = RF_center_Xs[y];
		}
	}

	for (int x = 0; x < fea_h; x++) {
		RF_center_Ys[x] = receptive_field_center_start[scale_id] + receptive_field_stride[scale_id] * x;
		for (int y = 0; y < fea_w; y++) {
			RF_center_Ys_mat[x * fea_w + y] = RF_center_Ys[x];
		}
	}

	float* x_lt_mat = new float[fea_h * fea_w];
	float* y_lt_mat = new float[fea_h * fea_w];
	float* x_rb_mat = new float[fea_h * fea_w];
	float* y_rb_mat = new float[fea_h * fea_w];

	

	//x-left-top
	float mid_value = 0;
	int fea_spacial_size = fea_h * fea_w;
	for (int j = 0; j < fea_spacial_size; j++) {
		mid_value = RF_center_Xs_mat[j] - box_map[0*fea_spacial_size+j] * constant[scale_id];
		x_lt_mat[j] = mid_value < 0 ? 0 : mid_value;
	}
	//y-left-top
	for (int j = 0; j < fea_spacial_size; j++) {
		mid_value = RF_center_Ys_mat[j] - box_map[1 * fea_spacial_size + j] * constant[scale_id];
		y_lt_mat[j] = mid_value < 0 ? 0 : mid_value;
	}
	//x-right-bottom
	for (int j = 0; j < fea_spacial_size; j++) {
		mid_value = RF_center_Xs_mat[j] - box_map[2 * fea_spacial_size + j] * constant[scale_id];
		x_rb_mat[j] = mid_value > cols - 1 ? cols - 1 : mid_value;
	}
	//y-right-bottom
	for (int j = 0; j < fea_spacial_size; j++) {
		mid_value = RF_center_Ys_mat[j] - box_map[3 * fea_spacial_size + j] * constant[scale_id];
		y_rb_mat[j] = mid_value > rows - 1 ? rows - 1 : mid_value;
	}


	for (int k = 0; k < fea_spacial_size; k++) {
		if (score_map[k] > score_threshold) {
			FaceInfo faceinfo;
			faceinfo.x1 = x_lt_mat[k];
			faceinfo.y1 = y_lt_mat[k];
			faceinfo.x2 = x_rb_mat[k];
			faceinfo.y2 = y_rb_mat[k];
			faceinfo.score = score_map[k];
			faceinfo.area = (faceinfo.x2 - faceinfo.x1) * (faceinfo.y2 - faceinfo.y1);
			bbox_collection.push_back(faceinfo);
		}
	}

	delete[] RF_center_Xs; RF_center_Xs = NULL;
	delete[] RF_center_Ys; RF_center_Ys = NULL;
	delete[] RF_center_Xs_mat; RF_center_Xs_mat = NULL;
	delete[] RF_center_Ys_mat; RF_center_Ys_mat = NULL;
	delete[] x_lt_mat; x_lt_mat = NULL;
	delete[] y_lt_mat; y_lt_mat = NULL;
	delete[] x_rb_mat; x_rb_mat = NULL;
	delete[] y_rb_mat; y_rb_mat = NULL;
}

void LFFD::get_topk_bbox(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, int top_k)
{
	std::sort(input.begin(), input.end(),
		[](const FaceInfo& a, const FaceInfo& b)
		{
			return a.score > b.score;
		});

	if (input.size() > top_k) {
		for (int k = 0; k < top_k; k++) {
			output.push_back(input[k]);
		}
	}
	else {
		output = input;
	}
}

void LFFD::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float threshold, int type)
{
	if (input.empty()) {
		return;
	}
	std::sort(input.begin(), input.end(),
	[](const FaceInfo& a, const FaceInfo& b)
	{
		return a.score > b.score;
	});

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;


		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;//std::max(input[i].x1, input[j].x1);
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;  //bug fixed ,sorry
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;


			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score= inner_area/area1;

			if (score > threshold)
				merged[j] = 1;
		}

	}
}
