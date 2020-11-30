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

std::shared_ptr<open3d::geometry::RGBDImage> generateRGBDImageFromCV(cv::Mat color_mat, cv::Mat depth_mat) {

	int height = 480;
	int width = 640;

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

	auto rgbd_image = open3d::geometry::RGBDImage::CreateFromColorAndDepth(*color_im, *depth_im, 1000.0, 2.3, false);
	return rgbd_image;
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

	//intrinsics need to be set by user (currently does not read d435i_intr.yaml)
	auto intr = open3d::camera::PinholeCameraIntrinsic(640, 480, 612.081, 612.307, 318.254, 237.246);

	FrameAvailableHandler tsdfFrameHandler([&tsdf_volume, &frame_counter, &do_integration, intr](sl::Camera &cam) {
		if (!do_integration || frame_counter % 3 != 0) {
			return;
		}

		//cout << "Integrating frame number: " << frame->frameId_ << endl;
        //get color and depth images
        Mat tempImage;//TODO: change it to sl:mat or sth to indicate its the zed mat
        cam.retrieveImage(tempImage, VIEW::LEFT);
        cv::Mat color_mat = slMat2cvMat(tempImage);

		cam.retrieveMeasure(tempImage, MEASURE::DEPTH);
        cv::Mat depth_mat = slMat2cvMat(tempImage);

		auto rgbd_image = generateRGBDImageFromCV(color_mat, depth_mat);

        //get pose
        Pose zed_pose;
        POSITIONAL_TRACKING_STATE tracking_state;

        tracking_state = zed.getPosition(zed_pose, REFERENCE_FRAME::WORLD);
        if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
            // Get rotation and translation and displays it
            Eigen::Matrix4f pose;
            slPose2Matrix(zed_pose, pose);
            tsdf_volume->Integrate(*rgbd_image, intr, pose.inverse());
        }else{
            std::cerr << "Positional tracking state wrong, cannot intergrate frame." << std::endl;
        }
	});

	MyGUI::MeshWindow mesh_win("Mesh Viewer", 1200, 1200);
	MyGUI::Mesh mesh_obj("mesh");

	mesh_win.add_object(&mesh_obj);


	std::shared_ptr<open3d::geometry::TriangleMesh> vis_mesh;

	FrameAvailableHandler meshHandler([&tsdf_volume, &frame_counter, &do_integration, &vis_mesh, &mesh_obj](sl::Camera &cam) {
		if (!do_integration || frame_counter % 30 != 1) {
			return;
		}
			
			vis_mesh = tsdf_volume->ExtractTriangleMesh();

			cout << "num vertices: " << vis_mesh->vertices_.size() << endl;
			cout << "num triangles: " << vis_mesh->triangles_.size() << endl;

			mesh_obj.update_mesh(vis_mesh->vertices_, vis_mesh->vertex_colors_, vis_mesh->triangles_);
	});

	FrameAvailableHandler viewHandler([&mesh_obj, &tsdf_volume, &mesh_win, &frame_counter](sl::Camera &cam) {
        //get pose
        Pose zed_pose;
        POSITIONAL_TRACKING_STATE tracking_state;

        tracking_state = zed.getPosition(zed_pose, REFERENCE_FRAME::WORLD);
        if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
            // Get rotation and translation and displays it
            Eigen::Matrix4f pose;
            slPose2Matrix(zed_pose, pose);
            
            Eigen::Affine3d transform(pose);
		    mesh_obj.set_transform(transform.inverse());
        }else{
            std::cerr << "Positional tracking state wrong, cannot intergrate frame." << std::endl;
        }
	});

	// thread *app = new thread(application_thread);

	cv::namedWindow("image");

    //initialize zed camera
    Camera cam;
    InitParameters init_parameters;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.camera_fps = 30;

    setStreamParameter(init_parameters, ipParam);

    //check if camera is opened successfully
    auto returned_state = cam.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return -1;//TODO: change to exit param error code
    }

    PositionalTrackingParameters positional_tracking_param;
    positional_tracking_param.enable_area_memory = true;
    // enable Positional Tracking
    returned_state = cam.enablePositionalTracking(positional_tracking_param);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Enabling positionnal tracking failed: ", returned_state);
        cam.close();
        return -1;//TODO: change to exit param error code
    }

    Mat image;//TODO: change it to sl:mat or sth to indicate its the zed mat
	while (MyGUI::Manager::running()) {

		//printf("test\n");
		//Update the display
		MyGUI::Manager::update();

        //do reconstruction when receiving data 
        returned_state = cam.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            frame_counter++;
            //call functions to do 3d recon
            tsdfFrameHandler(cam);
            meshHandler(cam);
            viewHandler(cam);
            //display image
            cam.retrieveImage(image, VIEW::LEFT);
            cv::Mat imBGR = slMat2cvMat(image);

            cv::imshow("image", imBGR);

        } else {
            print("Error during capture : ", returned_state);
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

	//std::shared_ptr<const open3d::geometry::Geometry> mesh = tsdf_volume->ExtractTriangleMesh();

	std::shared_ptr<open3d::geometry::TriangleMesh> write_mesh = tsdf_volume->ExtractTriangleMesh();

	//const std::vector<std::shared_ptr<const open3d::geometry::Geometry>> mesh_vec = { mesh };

	//open3d::visualization::DrawGeometries(mesh_vec);

	open3d::io::WriteTriangleMeshToPLY("mesh.ply", *write_mesh, false, false, true, true, false, false);

	printf("\nTerminate...\n");
	// Clean up
    cam.disablePositionalTracking();
	cam.close();
	printf("\nExiting...\n");
	return 0;
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

//set stream param  https://github.com/stereolabs/zed-opencv/blob/master/cpp/src/main.cpp
void setStreamParameter(InitParameters& init_p, string& argument) {
    vector< string> configStream = split(argument, ':');
    String ip(configStream.at(0).c_str());
    if (configStream.size() == 2) {
        init_p.input.setFromStream(ip, atoi(configStream.at(1).c_str()));
    } else init_p.input.setFromStream(ip);
}

void slPose2Matrix(&Pose pose, &Eigen::Matrix4d matrix)
{
	Translation translation = pose.getTranslation();
	Rotation rotation = pose.getRotationMatrix();
	Eigen::Vector3f T_SL(translation[0], translation[1], translation[2]);
	Eigen::Map<Eigen::Matrix3f>R_SL(&rotation.transpose(rotation).r[0]);
	matrix = Eigen::Matrix4f::Identity();
	matrix.block<3, 3>(0, 0) = R_SL;
	matrix.block<3, 1>(0, 3) = T_SL;
}
