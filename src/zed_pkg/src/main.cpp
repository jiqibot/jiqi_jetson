
// includes
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <costmap_converter/ObstacleMsg.h>
#include <costmap_converter/ObstacleArrayMsg.h>
#include <geometry_msgs/Point32.h>
#include <nav_msgs/Odometry.h>

#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Flag to disable the GUI when using on jetson
#define ENABLE_GUI 0

// ZED includes
#include <sl/Camera.hpp>
#include "utils.hpp"
// other code includes (for GUI)
#if ENABLE_GUI
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"
#endif

// Using std and sl namespaces
using namespace std;
using namespace sl;

bool is_playback = false;
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters& param);

// main code
int main(int argc, char **argv) {

    #ifdef _SL_JETSON_
        const bool isJetson = true;
    #else
        const bool isJetson = false;
    #endif

    bool visualise = false;

    ros::init(argc, argv, "zed2Cam");
    ros::NodeHandle n;
    ROS_INFO("NODE INITIALISED"); //Initialise node


    //Initialise publishers
    ros::Publisher point_cloud_pub = n.advertise<sensor_msgs::PointCloud2>("point_cloud", 1); //Point cloud publisher
    ros::Publisher bbox_pub = n.advertise<visualization_msgs::MarkerArray>("object_bounding_boxes", 1); //Object Bounding Box Publisher, visualisation purposes only
    ros::Publisher obs_pub = n.advertise<costmap_converter::ObstacleArrayMsg>("/test_optim_node/obstacles", 1); //Obstacle publisher for teb local planner
    ros::Publisher stop_obj_pub = n.advertise<std_msgs::Bool>("front_object_close_stop",1); //Bool publisher if a detected object is closer than 1m away, can be used to change the robot max speed
    ros::Publisher slow_obj_pub = n.advertise<std_msgs::Bool>("front_object_close_slow",1); //Bool publisher if a detected object is closer than 3m away, can be used to change the robot max speed
    ros::Publisher left_pub = n.advertise<sensor_msgs::Image>("zed_left_image", 1);
    ros::Publisher right_pub = n.advertise<sensor_msgs::Image>("zed_right_image", 1);
    ros::Publisher stereo_pub = n.advertise<sensor_msgs::Image>("zed_stereo_image", 1);
    ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 50);


    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.camera_fps = 60;
    //init_parameters.coordinate_units = sl::UNIT::METER; 
    init_parameters.depth_maximum_distance = 10.0f * 1000.0f;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // Rviz's coordinate system is right_handed (might have to be changed for Z up)
    init_parameters.sdk_verbose = 1;

    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // enabling position tracking - prereq for body and object tracking 
    auto camera_config = zed.getCameraInformation().camera_configuration;
    PositionalTrackingParameters positional_tracking_parameters;
    zed.enablePositionalTracking(positional_tracking_parameters);

    print("Skeleton Detection: Loading Module...");
    // Define the body detection module parameters
    BodyTrackingParameters body_tracking_parameters;
    body_tracking_parameters.enable_tracking = true;
    body_tracking_parameters.enable_segmentation = false; // set true should give person pixel mask
    body_tracking_parameters.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
    body_tracking_parameters.instance_module_id = 0; // select instance ID
    // enabling body tracking for distance, velocity etc of person
    returned_state = zed.enableBodyTracking(body_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableBodyTracking", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Define the object detection module parameters - using staandard not custom detection
    ObjectDetectionParameters object_detection_parameters;
    object_detection_parameters.enable_tracking = true;
    object_detection_parameters.enable_segmentation = false; 
    object_detection_parameters.detection_model = OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_MEDIUM;
    object_detection_parameters.instance_module_id = 1; // select instance ID
    // enabling object tracking
    print("Object Detection: Loading Module...");
    returned_state = zed.enableObjectDetection(object_detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Detection runtime parameters for objects
    int detection_confidence_od = 20;
    ObjectDetectionRuntimeParameters detection_parameters_rt(detection_confidence_od);

    // Detection runtime parameters for bodies
    // default detection threshold, apply to all object class
    int body_detection_confidence = 20;
    BodyTrackingRuntimeParameters body_tracking_parameters_rt(body_detection_confidence);

    // Detection output
    bool quit = false;
    RuntimeParameters runtime_parameters;
    runtime_parameters.confidence_threshold = 20;
    Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();
    Objects objects;
    Bodies skeletons; // structure containing all detected bodies
    //declare sensor variables
    sl::SensorsData sensors_data;
    sl::SensorsData::IMUData imu_data;

    bool gl_viewer_available = true;
    cout << setprecision(3);
//GUI
#if ENABLE_GUI

    float image_aspect_ratio = camera_config.resolution.width / (1.f * camera_config.resolution.height);
    int requested_low_res_w = min(1280, (int)camera_config.resolution.width);
    sl::Resolution display_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);
    Resolution tracks_resolution(400, display_resolution.height);
    // create a global image to store both image and tracks view
    cv::Mat global_image(display_resolution.height, display_resolution.width + tracks_resolution.width, CV_8UC4, 1);
    // retrieve ref on image part
    auto image_left_ocv = global_image(cv::Rect(0, 0, display_resolution.width, display_resolution.height));
    // retrieve ref on tracks view part
    auto image_track_ocv = global_image(cv::Rect(display_resolution.width, 0, tracks_resolution.width, tracks_resolution.height));
    // init an sl::Mat from the ocv image ref (which is in fact the memory of global_image)
    cv::Mat image_render_left = cv::Mat(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_render_left.data, image_render_left.step);
    sl::float2 img_scale(display_resolution.width / (float) camera_config.resolution.width, display_resolution.height / (float) camera_config.resolution.height);


    // 2D tracks
    TrackingViewer track_view_generator(tracks_resolution, camera_config.fps, init_parameters.depth_maximum_distance, 3);
    track_view_generator.setCameraCalibration(camera_config.calibration_parameters);

    string window_name = "ZED| 2D View and Birds view";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL); // Create Window
    cv::createTrackbar("Confidence OD", window_name, &detection_confidence_od, 100);
    cv::createTrackbar("Confidence Body", window_name, &body_detection_confidence, 100);

    char key = ' ';

    requested_low_res_w = min(720, (int)camera_config.resolution.width);
    Resolution pc_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);

    std::cout << "Res " << display_resolution.width << " " << display_resolution.height << " " << pc_resolution.width << " " << pc_resolution.height << std::endl;

    auto camera_parameters = zed.getCameraInformation(pc_resolution).camera_configuration.calibration_parameters.left_cam;
    Mat point_cloud(pc_resolution, MAT_TYPE::F32_C4, MEM::GPU);
    GLViewer viewer;
    viewer.init(argc, argv, camera_parameters, body_tracking_parameters.enable_tracking);
#endif
    
    //loop containing information to be published and updated in real time
    while (true) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) { //if camera succesfully initialises
        


        //creates boolean message for stopping and slowing and sets them to false at the start of every loop
        bool within1m = false;
        bool within2m = false;
        // outputs number of people in view for testing purpose
        //cout << "No. Persons = " << skeletons.body_list.size() << endl;
        
        /*
        //loops through all detected bodies
        for (int i = 0; i < skeletons.body_list.size(); i++) {
                //retrieves information on detected body
                sl::BodyData body = skeletons.body_list[i];
                zed.retrieveBodies(skeletons, body_tracking_parameters_rt);
                
                unsigned int body_id = body.id; // Get the body id

                //displays the xyz value of person in view - for calibrate/testing
                //cout << "Person x position = " << body.position.x << endl;
                //cout << "Person y position = " << body.position.y << endl;  
                //cout << "Person z position = " << body.position.z << endl;

                //if person is too close update bools to change speed.
                if (body.position.z >-1000 ){ // within 1m stop

                    within1m = true;
                    
                }
                if (body.position.z >-2000 ){ //within 2m slow
            
                    within2m = true;
                }
                
        }*/

        zed.retrieveObjects(objects, detection_parameters_rt);
        // Loop through all detected objects
            for (int i = 0; i < objects.object_list.size(); i++) {
                
                sl::ObjectData object = objects.object_list[i];
                if (object.label == sl::OBJECT_CLASS::PERSON){
                //if person is too close publish bools to change robot speed.
                if (object.position.x >-1000 ){ //object_label ==1 ==person

                    within1m = true;
                }

                if (object.position.x >-2000 ){ //add and if object class = person

                    within2m = true;
                }
            }
            }
        // Publish the appropriate messages based on the flags
        std_msgs::Bool stop_msg;
        stop_msg.data = within1m;
        stop_obj_pub.publish(stop_msg);

        std_msgs::Bool slow_msg;
        slow_msg.data = within2m;
        slow_obj_pub.publish(slow_msg);
       
        sl::Mat point_cloud;
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

        //converting ZED point cloud to ROS sensor_msgs::PointCloud2
        sensor_msgs::PointCloud2 ros_point_cloud;
        ros_point_cloud.header.stamp = ros::Time::now();
        ros_point_cloud.header.frame_id = "zed_camera_frame"; //set desired frame id
        ros_point_cloud.height = point_cloud.getHeight();
        ros_point_cloud.width = point_cloud.getWidth();
        ros_point_cloud.is_dense = false;
        ros_point_cloud.is_bigendian = false;
        ros_point_cloud.point_step = sizeof(float) * 4;
        ros_point_cloud.row_step = ros_point_cloud.point_step * ros_point_cloud.width;
        ros_point_cloud.data.resize(ros_point_cloud.height * ros_point_cloud.row_step);

        //copy data
        memcpy(&ros_point_cloud.data[0], point_cloud.getPtr<sl::float4>(), ros_point_cloud.data.size());
        //publish point cloud
        point_cloud_pub.publish(ros_point_cloud);
       
        // gets required information for odom
        zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
        zed.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT);
        /*
        //Broadcast odom -> base_link TF
        static tf::TransformBroadcaster odom_broadcaster;
        tf::Transform odom_trans;
        odom_trans.setOrigin( tf::Vector3(cam_w_pose.getTranslation().x, cam_w_pose.getTranslation().y, cam_w_pose.getTranslation().z));
        tf::Quaternion q;
        q.setRPY(cam_w_pose.getOrientation().ox, cam_w_pose.getOrientation().oy, cam_w_pose.getOrientation().oz);
        odom_trans.setRotation(q);
        odom_broadcaster.sendTransform(tf::StampedTransform(odom_trans, ros::Time::now(), "odom", "base_link"));
        
        //Calculate position and orientations
        nav_msgs::Odometry odom_msg;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";
        odom_msg.pose.pose.position.x = cam_w_pose.getTranslation().x;
        odom_msg.pose.pose.position.y = cam_w_pose.getTranslation().y;
        odom_msg.pose.pose.position.z = cam_w_pose.getTranslation().z;
        odom_msg.pose.pose.orientation.x = cam_w_pose.getOrientation().ox;
        odom_msg.pose.pose.orientation.y = cam_w_pose.getOrientation().oy;
        odom_msg.pose.pose.orientation.z = cam_w_pose.getOrientation().oz;

        odom_msg.twist.twist.linear.x = 0;
        odom_msg.twist.twist.linear.y = 0;
        odom_msg.twist.twist.linear.z = 0;

        odom_msg.twist.twist.angular.x = sensors_data.imu.angular_velocity.x;
        odom_msg.twist.twist.angular.y = sensors_data.imu.angular_velocity.y;
        odom_msg.twist.twist.angular.z = sensors_data.imu.angular_velocity.z;

        odom_pub.publish(odom_msg);
*/
    }

    //GUI 
    #if ENABLE_GUI
            gl_viewer_available &&
    #endif
            !quit; {

        auto grab_state = zed.grab(runtime_parameters);
        if (grab_state == ERROR_CODE::SUCCESS) {
            detection_parameters_rt.detection_confidence_threshold = detection_confidence_od;
            returned_state = zed.retrieveObjects(objects, detection_parameters_rt, object_detection_parameters.instance_module_id);

            body_tracking_parameters_rt.detection_confidence_threshold = body_detection_confidence;
            returned_state = zed.retrieveBodies(skeletons, body_tracking_parameters_rt, body_tracking_parameters.instance_module_id);
        //GUI             
        #if ENABLE_GUI
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, pc_resolution);
            image_render_left.copyTo(image_left_ocv);
            render_2D(image_left_ocv, img_scale, objects, skeletons, true, body_tracking_parameters.enable_tracking);

            zed.getPosition(cam_w_pose, REFERENCE_FRAME::WORLD);

            viewer.updateData(point_cloud, objects, skeletons, cam_w_pose.pose_data);

            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked);
        #else
            cout << "Detected " << objects.object_list.size() << " Object(s)" << endl;
        #endif


            if (is_playback && zed.getSVOPosition() == zed.getSVONumberOfFrames())
                quit = true;
        }        
    //GUI
    #if ENABLE_GUI
        gl_viewer_available = viewer.isAvailable();
        // as image_left_ocv and image_track_ocv are both ref of global_image, no need to update it
        cv::imshow(window_name, global_image);
        key = cv::waitKey(10);
        if (key == 'i') {
            track_view_generator.zoomIn();
        } else if (key == 'o') {
            track_view_generator.zoomOut();
        } else if (key == 'q') {
            quit = true;
        }
    #endif
    }


    }

//GUI   
#if ENABLE_GUI
    viewer.exit();
    point_cloud.free();
    image_left.free();
#endif
    //Close
    zed.disableObjectDetection();
    zed.close();
    return EXIT_SUCCESS;
}

//Errors function
void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

//ParseArgs function
void parseArgs(int argc, char **argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        is_playback = true;
        cout << "Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "Using Camera in resolution HD720" << endl;
        } else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "Using Camera in resolution SVGA" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "Using Camera in resolution VGA" << endl;
        }
    }
}
