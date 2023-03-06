#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define YOLO_SIZE 320												//yolo3 ��Ʈ��ũ �Է� ������ 320 416 609											

constexpr float CONFIDENCE_THRESHOLD = 0.5;							// ��� �Ӱ谪
constexpr float NMS_THRESHOLD = 0.4;								// ��ħ �Ӱ谪    // non maximum suppresion
constexpr int NUM_CLASSES = 80;										// ���� ��ü ���� // class_names.size()

// colors for bounding boxes
const cv::Scalar colors[] = {										
	{0, 255, 255},
	{255, 255, 0},
	{0, 255, 0},
	{255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main()
{
	std::vector<std::string> class_names;							// Ž�� ��ü ( person, sheep, etc...)

	{
		std::ifstream class_file("coco.names");
		if (!class_file)
		{
			std::cerr << "failed to open classes.txt\n";
			return 0;
		}

		std::string line;
		while (std::getline(class_file, line))
			class_names.push_back(line);
	}
	
	auto net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");		// ȯ�漳��, ����ġ
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);						// GPU ���
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);							// GPU ���
	// net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);					// CPU ���
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);						// CPU ���
	auto output_names = net.getUnconnectedOutLayersNames();						// ������ ���̾��� �̸� (��� ���̾�?) -> yolo_82, yolo_94, yolo_106 ??

	
	cv::Mat frame = cv::imread("./../../data/kids.png", cv::IMREAD_COLOR);		// ������
	//cv::Mat frame = cv::imread("./../../image_human/human2.jpg", cv::IMREAD_COLOR);		// ������
	cv::Mat blob;																// ��ó��
	std::vector<cv::Mat> detections;

	cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(YOLO_SIZE, YOLO_SIZE), cv::Scalar(), true, false, CV_32F);		// ��ó��
	net.setInput(blob);															// �Ű���� (��ó��)�Է�

	net.forward(detections, output_names);										// ������ ���̾� �̸� �ʿ�, ������, ���� ���� -> y = w*x + b
	/*
	detections
	[[85 X 300],
	[85 X 1200,
	[85 X 4800]]
	85 = 5 + 80
	0 -> center_x			��ǥ�� ����� ��ǥ?
	1 -> center_y			y��
	2 -> width				�ʺ�
	3 -> height				����
	4 -> �ڽ��� ��ü�� �ѷ� ������ ���� �ŷڵ�
	5 ~ 84 -> �� Ŭ����(coco.names)�� ���� �ŷڵ�
	*/

	for (auto& i : output_names) std::cout << i << std::endl;
	std::cout << blob.size() << std::endl;
	std::cout << frame.size() << std::endl;
	std::cout << class_names.size() << std::endl;
	std::cout << detections.size() << std::endl;
	std::cout << detections[0].size() << std::endl;
	std::cout << detections[1].size() << std::endl;
	std::cout << detections[2].size() << std::endl;
	//std::cout << detections[0] << std::endl;

	std::vector<int> indices[NUM_CLASSES];					// �� ��ü[class_names] �� 
	std::vector<cv::Rect> boxes[NUM_CLASSES];
	std::vector<float> scores[NUM_CLASSES];

	for (auto& output : detections)
	{
		const auto num_boxes = output.rows;
		//std::cout << num_boxes << std::endl;
		for (int i = 0; i < num_boxes; i++)
		{
			auto x = output.at<float>(i, 0) * frame.cols;
			auto y = output.at<float>(i, 1) * frame.rows;
			auto width = output.at<float>(i, 2) * frame.cols;
			auto height = output.at<float>(i, 3) * frame.rows;
			cv::Rect rect(x - width / 2, y - height / 2, width, height);			

			for (int c = 0; c < NUM_CLASSES; c++)
			{
				auto confidence = *output.ptr<float>(i, 5 + c);
				if (confidence >= CONFIDENCE_THRESHOLD)
				{
					boxes[c].push_back(rect);
					scores[c].push_back(confidence);

					//std::cout << x << ", " << y << ", " << width << ", " << height << std::endl;
				}
			}
		}
	}
	if (0) {
		auto x = detections[0].at<float>(1, 0) * frame.cols;
		auto y = detections[0].at<float>(1, 1) * frame.rows;
		auto width = detections[0].at<float>(1, 2) * frame.cols;
		auto height = detections[0].at<float>(1, 3) * frame.rows;
		cv::Rect rect(x - width / 2, y - height / 2, width, height);
		std::cout << "x - width / 2 : " << x - width / 2 << std::endl;
		std::cout << "y - height / 2 : " << y - height / 2 << std::endl;
		std::cout << "width : " << width << std::endl;
		std::cout << "height : " << height << std::endl;
		std::cout << "confidence : " << detections[0].at<float>(1, 5 + 1) << std::endl;
		std::cout << "Rect : " << rect << std::endl;
		cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 0, 255), 3);
	}

	/*
	if (confidence_threshold ���� 0 �̶��)
	boxes
	scores	// ���� 6300�� ���� (300 + 1200 + 4800) 

	else 
	( �ڽ��� �ŷڵ� > confidence_threshold) �� �ڽ��� �ŷڵ��� push
	*/


	for (int c = 0; c < NUM_CLASSES; c++)
		cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);			// ������ ���� // ���� ��ü�� ���� �ڽ� ��ħ ����	// indices[c] �� �߰� (������??)

	// ��ġ�� �ڽ��߿��� ���� ���� �ŷڵ��� �ڽ���ȣ�� indices[c]�� ����?

	for (int c = 0; c < NUM_CLASSES; c++)
	{
		std::cout << "number of " << std::setw(15) << class_names[c] << " -> " << indices[c].size() << std::endl;
		for (size_t i = 0; i < indices[c].size(); ++i)
		{
			const auto color = colors[c % NUM_COLORS];

			auto idx = indices[c][i];
			const auto& rect = boxes[c][idx];
			cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

			std::ostringstream label_ss;
			//label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
			label_ss << class_names[c] << "(" << i << ")(" << idx << "): " << std::fixed << std::setprecision(2) << scores[c][idx];			// i �����ȣ, idx �ڽ���ȣ
			auto label = label_ss.str();

			int baseline;
			auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
			cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
			cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
		}
	}
	std::cout << "=========================" << std::endl;
	std::cout << indices[0].size() << std::endl;
	std::cout << boxes[0].size() << std::endl;
	std::cout << scores[0].size() << std::endl;
	std::cout << "=========================" << std::endl;
	for (auto& i : indices[0]) { std::cout << i << std::endl; }
	//for (auto& i : boxes[0]) { std::cout << i << std::endl; }
	//for (auto& i : scores[0]) { std::cout << i << std::endl; }

	for (int i = 0; i < boxes[0].size(); i++) {
		std::cout << std::setw(3) << i << " : " << boxes[0][i] << ", " << scores[0][i] << std::endl;
	}

	cv::namedWindow("output");
	cv::imshow("output", frame);
	cv::waitKey(0);
	


	/*
	for (auto& i : class_names) {
		std::cout << i << std::endl;
	}
	*/
	return 0;
}