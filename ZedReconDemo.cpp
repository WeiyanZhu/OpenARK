#include "D435iCamera.h"
#include "OkvisSLAMSystem.h"
#include <iostream>
//#include <direct.h>
#include <sl/Camera.hpp>
#include <thread>
#include "glfwManager.h"
#include "Util.h"
#include "SaveFrame.h"
#include "Types.h"
#include "Open3D/Integration/ScalableTSDFVolume.h"
#include "Open3D/Visualization/Utility/DrawGeometry.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/IO/ClassIO/ImageIO.h"

using namespace ark;
using namespace sl;

// Parameters to change here
float voxel_size = 0.03;
float block_size = 3.0;
int max_depth = 10;

std::shared_ptr<open3d::geometry::RGBDImage> generateRGBDImageFromCV(cv::Mat color_mat, cv::Mat depth_mat) {

	cv::Size s = color_mat.size();
	int height = s.height;
	int width = s.width;

	auto color_im = std::make_shared<open3d::geometry::Image>();
	color_im->Prepare(width, height, 3, sizeof(uint8_t));

	uint8_t *pi = (uint8_t *)(color_im->data_.data());

	for (int i = 0; i < height; i++) {
		for (int k = 0; k < width; k++) {

			cv::Vec3b pixel = color_mat.at<cv::Vec3b>(i, k);

			*pi++ = pixel[0];
			*pi++ = pixel[1];
			*pi++ = pixel[2];
		}
	}

	auto depth_im = std::make_shared<open3d::geometry::Image>();
	depth_im->Prepare(width, height, 1, sizeof(uint16_t));

	uint16_t * p = (uint16_t *)depth_im->data_.data();

	for (int i = 0; i < height; i++) {
		for (int k = 0; k < width; k++) {
			*p++ = depth_mat.at<uint16_t>(i, k);
		}
	}

	auto rgbd_image = open3d::geometry::RGBDImage::CreateFromColorAndDepth(*color_im, *depth_im, 1000.0, max_depth, false);

	return rgbd_image;
}

//convert zed mat to cv mat  https://github.com/stereolabs/zed-opencv/blob/master/cpp/src/main.cpp
cv::Mat slMat2cvMat(Mat& input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
	case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
	case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
	case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
	case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
	case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
	case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
	case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

vector< string> split(const string& s, char seperator) {
	vector< string> output;
	string::size_type prev_pos = 0, pos = 0;

	while ((pos = s.find(seperator, pos)) != string::npos) {
		string substring(s.substr(prev_pos, pos - prev_pos));
		output.push_back(substring);
		prev_pos = ++pos;
	}

	output.push_back(s.substr(prev_pos, pos - prev_pos));
	return output;
}

//set stream param  https://github.com/stereolabs/zed-opencv/blob/master/cpp/src/main.cpp
void setStreamParameter(InitParameters& init_p, string& argument) {
	vector< string> configStream = split(argument, ':');
	String ip(configStream.at(0).c_str());
	if (configStream.size() == 2) {
		init_p.input.setFromStream(ip, atoi(configStream.at(1).c_str()));
	}
	else init_p.input.setFromStream(ip);
}

void slPose2Matrix(Pose& pose, Eigen::Matrix4d& matrix)
{
	Translation translation = pose.getTranslation();
	Rotation rotation = pose.getRotationMatrix();
	Eigen::Vector3d T_SL(translation[0], translation[1], translation[2]);
	Eigen::Map<Eigen::Matrix3f>R_SL(&rotation.transpose(rotation).r[0]);
	matrix = Eigen::Matrix4d::Identity();
	matrix.block<3, 3>(0, 0) = R_SL.cast<double>();
	matrix.block<3, 1>(0, 3) = T_SL;
}


//TODO: loop closure handler calling deintegration
int main(int argc, char **argv)
{

	if (argc > 5) {
		std::cerr << "Usage: ./" << argv[0] << "ip-address configuration-yaml-file [vocabulary-file] [skip-first-seconds]" << std::endl
			<< "Args given: " << argc << std::endl;
		return -1;
	}

    if (argc < 2) {
		std::cerr << "Please input ip address" << std::endl;
		return -1;
	}

	google::InitGoogleLogging(argv[0]);

	okvis::Duration deltaT(0.0);
	if (argc == 5) {
		deltaT = okvis::Duration(atof(argv[4]));
	}

    //initialize ip
    std::string ipParam = string(argv[1]);

	// read configuration file
	std::string configFilename;
	if (argc > 2) configFilename = argv[2];
	else configFilename = util::resolveRootPath("config/d435i_intr.yaml");

	std::string vocabFilename;
	if (argc > 3) vocabFilename = argv[3];
	else vocabFilename = util::resolveRootPath("config/brisk_vocab.bn");

	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);

	//setup display
	if (!MyGUI::Manager::init())
	{
		fprintf(stdout, "Failed to initialize GLFW\n");
		return -1;
	}

	//run until display is closed
	okvis::Time start(0.0);
	int id = 0;

	int frame_counter = 1;
	bool do_integration = true;

	open3d::integration::ScalableTSDFVolume * tsdf_volume = new open3d::integration::ScalableTSDFVolume(0.015, 0.05, open3d::integration::TSDFVolumeColorType::RGB8);

	// TODO: read this from a config file
	auto intr = open3d::camera::PinholeCameraIntrinsic(1280, 720, 521.0787963867188, 521.0787963867188, 658.9640502929688, 357.2626647949219);

	// TODO: read from config file instead of hardcoding
	SegmentedMesh* mesh = new SegmentedMesh(voxel_size, voxel_size * 5, open3d::integration::TSDFVolumeColorType::RGB8, block_size, false);

	cv::namedWindow("image");

    //initialize zed camera
    Camera cam;
    InitParameters init_parameters;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.camera_fps = 30;

	/* FOR STREAMING, UNCOMMENT */
    setStreamParameter(init_parameters, ipParam);

    //check if camera is opened successfully
    auto returned_state = cam.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        std::cerr << "Camera Open " << returned_state << ". Exit program." << std:: endl;
        return -1; //TODO: change to exit param error code
    }

    PositionalTrackingParameters positional_tracking_param;
    positional_tracking_param.enable_area_memory = true;
    // enable Positional Tracking
    returned_state = cam.enablePositionalTracking(positional_tracking_param);
    if (returned_state != ERROR_CODE::SUCCESS) {
        std::cerr << "Enabling positional tracking failed: " << returned_state << std::endl;
        cam.close();
        return -1; //TODO: change to exit param error code
    }

	sl::Mat image;

	FrameAvailableHandler tsdfFrameHandler([&frame_counter, &do_integration, intr, &mesh, &cam](MultiCameraFrame::Ptr frame) {
		if (!do_integration || frame_counter % 3 != 0) {
			return;
		}

		//get color and depth images
		sl::Mat tempImageLeft(1280, 720, MAT_TYPE::U8_C4);
		sl::Mat tempImageDepth(1280, 720, MAT_TYPE::F32_C1);
		
		cv::Mat color_mat = slMat2cvMat(tempImageLeft);
		cam.retrieveImage(tempImageLeft, VIEW::LEFT);
		cv::Mat color_mat3 = cv::Mat();
		cv::cvtColor(color_mat, color_mat3, CV_BGRA2RGB);

		
		cv::Mat depth_mat = slMat2cvMat(tempImageDepth);
		cam.retrieveMeasure(tempImageDepth, MEASURE::DEPTH);

		depth_mat *= 1000;
		depth_mat.convertTo(depth_mat, CV_16UC1);

		auto rgbd_image = generateRGBDImageFromCV(color_mat3, depth_mat);

		//get pose
		Pose zed_pose;
		POSITIONAL_TRACKING_STATE tracking_state;

		tracking_state = cam.getPosition(zed_pose, REFERENCE_FRAME::WORLD);

		if (cam.grab() != ERROR_CODE::SUCCESS) {
			std::cerr << "Cannot get input from zed camera" << std::endl;
		}

		if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
			// Get rotation and translation and displays it
			Eigen::Matrix4d pose;
			slPose2Matrix(zed_pose, pose);

			mesh->Integrate(*rgbd_image, intr, pose.inverse());
		}
		else {
			std::cerr << "Positional tracking state wrong, cannot intergrate frame: " << tracking_state << std::endl;
		}
	});

	MyGUI::MeshWindow mesh_win("Mesh Viewer", 1200, 1200);
	MyGUI::Mesh mesh_obj("mesh", mesh);

	mesh_win.add_object(&mesh_obj);


	std::shared_ptr<open3d::geometry::TriangleMesh> vis_mesh;

	FrameAvailableHandler meshHandler([&frame_counter, &do_integration, &vis_mesh, &mesh_obj, &cam](MultiCameraFrame::Ptr frame) {
		if (!do_integration || frame_counter % 30 != 1) {
			return;
		}

		mesh_obj.update_meshes();
	});

	FrameAvailableHandler viewHandler([&mesh_obj, &mesh_win, &frame_counter, &cam](MultiCameraFrame::Ptr frame) {
		//get pose
		Pose zed_pose;
		POSITIONAL_TRACKING_STATE tracking_state;
		
		tracking_state = cam.getPosition(zed_pose, REFERENCE_FRAME::WORLD);

		if (cam.grab() != ERROR_CODE::SUCCESS) {
			std::cerr << "Cannot get input from zed camera" << std::endl;
		}

		if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
			// Get rotation and translation and displays it
			Eigen::Matrix4d pose;
			slPose2Matrix(zed_pose, pose);

			Eigen::Affine3d transform(pose);
			mesh_obj.set_transform(transform.inverse());
		}
		else {
			std::cerr << "Positional tracking state wrong, cannot intergrate frame: " << tracking_state << std::endl;

		}
	});

	while (MyGUI::Manager::running()) {

		//Update the display
		MyGUI::Manager::update();

        //do reconstruction when receiving data 
        returned_state = cam.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            frame_counter++;

            //call functions to do 3d recon
            tsdfFrameHandler(NULL);
            meshHandler(NULL);
            viewHandler(NULL);

            //display image
            cam.retrieveImage(image, VIEW::LEFT);
            cv::Mat imBGR = slMat2cvMat(image);
            cv::imshow("image", imBGR);

        } else {
			std::cerr << "Error during capture : " << returned_state << std::endl;
            break;
        }

		int k = cv::waitKey(4);
		if (k == ' ') {
			do_integration = !do_integration;
			if (do_integration) {
				std::cout << "----INTEGRATION ENABLED----" << endl;
			}
			else {
				std::cout << "----INTEGRATION DISABLED----" << endl;
			}
		}
		
		if (k == 'q' || k == 'Q' || k == 27) break; // 27 is ESC

	}

	cout << "getting mesh" << endl;

	mesh->WriteMeshes();

	printf("\nTerminate...\n");

	// Clean up
    cam.disablePositionalTracking();
	cam.close();

	printf("\nExiting...\n");
	return 0;
}