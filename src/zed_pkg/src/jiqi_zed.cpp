// Jiqi Robot ZED Cam 2 program
// Lancaster University School of Engineering
// This Program uses yolov4 object detection to send object data for use 
// in robot navigation, additionally checks for people to 
// send manual stop message whe too close and  publishes 
// pointcloud and IMU data for use in navigation

// includes
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/MagneticField.h>
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


// ZED includes
#include <sl/Camera.hpp>
#include "utils.hpp"
// other code includes (for GUI)

#include "GLViewer.hpp"
#include "TrackingViewer.hpp"


// Using std and sl namespaces
using namespace std;
using namespace sl;

bool is_playback = false;
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters& param);


// Basic structure to compare timestamps of a sensor. Determines if a specific sensor data has been updated or not.
struct TimestampHandler {

    // Compare the new timestamp to the last valid one. If it is higher, save it as new reference.
    inline bool isNew(Timestamp& ts_curr, Timestamp& ts_ref) {
        bool new_ = ts_curr > ts_ref;
        if (new_) ts_ref = ts_curr;
        return new_;
    }
    // Specific function for IMUData.
    inline bool isNew(SensorsData::IMUData& imu_data) {
        return isNew(imu_data.timestamp, ts_imu);
    }
    // Specific function for MagnetometerData.
    inline bool isNew(SensorsData::MagnetometerData& mag_data) {
        return isNew(mag_data.timestamp, ts_mag);
    }

    Timestamp ts_imu = 0, ts_mag = 0; // Initial values
};

constexpr float CONFIDENCE_THRESHOLD = 0.4;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
constexpr int INFERENCE_SIZE = 416;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof (colors) / sizeof (colors[0]);


//convert opencv rectangle to zed bounding box
std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}
// main code
int main(int argc, char **argv) {

    #ifdef _SL_JETSON_
        const bool isJetson = true;
    #else
        const bool isJetson = false;
    #endif

    bool visualise = true;

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
    ros::Publisher imu_pub =n.advertise<sensor_msgs::Imu>("imu/data_raw",10);
    ros::Publisher magnetometer_pub = n.advertise<sensor_msgs::MagneticField>("/imu/mag", 10);


    //read object class names from text file
    std::vector<std::string> class_names;
    {
        std::ifstream class_file("/home/jiqi/jiqi_jetson/src/zed_pkg/coco.names.txt");
        if (!class_file) {
            for (int i = 0; i < NUM_CLASSES; i++)
                class_names.push_back(std::to_string(i));
        } else {
            std::string line;
            while (std::getline(class_file, line))
                class_names.push_back(line);
        }
    }

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.camera_fps = 30;
    init_parameters.coordinate_units = sl::UNIT::METER; 
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP; // Rviz's coordinate system is right_handed (might have to be changed for Z up)
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

    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;

    PositionalTrackingParameters positional_tracking_parameters;
    zed.enablePositionalTracking(positional_tracking_parameters);

    // Define the object detection module parameters - using staandard not custom detection
    ObjectDetectionParameters object_detection_parameters;
    object_detection_parameters.enable_tracking = true;
    object_detection_parameters.enable_segmentation = false; 
    object_detection_parameters.detection_model = OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    // enabling object tracking
    returned_state = zed.enableObjectDetection(object_detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Detection runtime parameters for objects
    int detection_confidence_od = 20;
    ObjectDetectionRuntimeParameters object_detection_parameters_rt(detection_confidence_od);

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
    //
    sl::Mat left_sl, point_cloud;
    sl::Mat left_image, right_image;
    sl::Pose prev_pose;
    //initialise yolo model and weights
    // Weight can be downloaded from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    auto net = cv::dnn::readNetFromDarknet("/home/jiqi/jiqi_jetson/src/zed_pkg/yolov4.cfg", "/home/jiqi/jiqi_jetson/src/zed_pkg/yolov4.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;

    bool gl_viewer_available = true;
    cout << setprecision(3);

    //loop containing information to be published and updated in real time
    while (true) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) { //if camera succesfully initialises
        
        sensor_msgs::Imu imu_msg;
        sensor_msgs::MagneticField magnetometer_msg;

        zed.retrieveImage(left_sl, sl::VIEW::LEFT);//grab left image frame

            // Preparing frame
            cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
            cv::cvtColor(left_cv_rgba, frame, cv::COLOR_BGRA2BGR);

            //perform object detection using YOLO
            cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(INFERENCE_SIZE, INFERENCE_SIZE), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            net.forward(detections, output_names);

            //process detections
            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];
            for (auto& output : detections) {
                const auto num_boxes = output.rows;
                for (int i = 0; i < num_boxes; i++) {
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);

                    for (int c = 0; c < NUM_CLASSES; c++) {
                        auto confidence = *output.ptr<float>(i, 5 + c);
                        if (confidence >= CONFIDENCE_THRESHOLD) {
                            boxes[c].push_back(rect);
                            scores[c].push_back(confidence);
                        }
                    }
                }
            }

            //perform non-maximum suppression to remove redundant objects
            for (int c = 0; c < NUM_CLASSES; c++)
                cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

            //loop through all remaining objects, draw bounding box around object, add class ID and probability to bounding box.
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (int c = 0; c < NUM_CLASSES; c++) {
                for (size_t i = 0; i < indices[c].size(); ++i) {
                    const auto color = colors[c % NUM_COLORS];

                    auto idx = indices[c][i];
                    const auto& rect = boxes[c][idx];
                    auto& rect_score = scores[c][idx];

                    // Fill the detections into the correct format
                    sl::CustomBoxObjectData tmp;
                    tmp.unique_object_id = sl::generate_unique_id();
                    tmp.probability = rect_score;
                    tmp.label = c;
                    tmp.bounding_box_2d = cvt(rect);
                    tmp.is_grounded = (c == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                    // others are tracked in full 3D space
                    objects_in.push_back(tmp);
                    //--

                    cv::rectangle(frame, rect, color, 3);

                    std::ostringstream label_ss;
                    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                    auto label = label_ss.str();

                    int baseline;
                    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);

            //enable below for camera output to display
            //cv::imshow("Objects", frame);
            //cv::waitKey(10);

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, object_detection_parameters_rt);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZBGRA, sl::MEM::CPU, pc_resolution);
            zed.retrieveImage(left_image, sl::VIEW::LEFT);
            zed.retrieveImage(right_image, sl::VIEW::RIGHT);

            //Publishers for all essential data
            ros::Time current_time = ros::Time::now();
            n.getParam("visualise", visualise); //check for visualisation parameter status

            
            //PointCloud publication
            if (visualise=true){
                sensor_msgs::PointCloud2Ptr pointcloudMsg = boost::make_shared<sensor_msgs::PointCloud2>();
                pointcloudMsg->header.stamp = current_time;
                pointcloudMsg->header.frame_id="map";
                int ptsCount = pc_resolution.width * pc_resolution.height;

                if (pointcloudMsg->width != pc_resolution.width || pointcloudMsg->height != pc_resolution.height) {

                    pointcloudMsg->is_bigendian = false;
                    pointcloudMsg->is_dense = false;
                    pointcloudMsg->width = pc_resolution.width;
                    pointcloudMsg->height = pc_resolution.height;
                    sensor_msgs::PointCloud2Modifier modifier(*pointcloudMsg);
                    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32,
                        "z", 1, sensor_msgs::PointField::FLOAT32, "rgb", 1, sensor_msgs::PointField::FLOAT32);
                }

                // Data copy
                sl::Vector4<float>* cpu_cloud = point_cloud.getPtr<sl::float4>();
                float* ptCloudPtr = (float*)(&pointcloudMsg->data[0]);

                // We can do a direct memcpy since data organization is the same
                memcpy(ptCloudPtr, (float*)cpu_cloud, 4 * ptsCount * sizeof(float));

                // Pointcloud publishing
                point_cloud_pub.publish(pointcloudMsg);
            }
            

        //Object Bounding Box publication
        visualization_msgs::MarkerArray bbox_array;
        visualization_msgs::Marker bbox;

        //teb_local_planner obstacle message
        costmap_converter::ObstacleArrayMsg obstacle_msg;
        obstacle_msg.header.stamp = ros::Time::now();
        obstacle_msg.header.frame_id = "map";

        //creates boolean message for stopping and slowing and sets them to false at the start of every loop
        bool within1m = false;
        bool within2m = false;
        
        // Loop through all detected objects
            for (int i = 0; i < objects.object_list.size(); i++) {

                sl::ObjectData object = objects.object_list[i];
                //check object is a person
                for (int c = 0; c < NUM_CLASSES; c++) {
                for (size_t e = 0; e < indices[c].size(); ++e) {
                    auto idx = indices[c][e];

                    //ensures object position is set to a float
                    float objectdistance = object.position.y;

                    
                    sl::CustomBoxObjectData tmp;
                    tmp.unique_object_id = sl::generate_unique_id();
                    
                    tmp.label = c;
                    objects_in.push_back(tmp);
                    
                    //if (tmp.is_grounded == (c == 0)){
                    //    cout << "Person x position = " << object.position.x << endl;
                    //    cout << "Person y position = " << object.position.y << endl;  
                    //    cout << "Person z position = " << object.position.z << endl; 
                    //}
                //if person is too close publish bools to change robot speed.
                    if (objectdistance <1 && tmp.is_grounded == (c == 0)){ //object_label ==1 ==person

                     within1m = true;
                    }

                    if (objectdistance <2 && tmp.is_grounded == (c == 0)){ //add and if object class = person

                       within2m = true;
                    }
                    }
            }
               
            // Custom obstacles for teb_local_planner
                costmap_converter::ObstacleMsg polygon_obstacle;
                polygon_obstacle.id=2; //polygon type obstacle
                polygon_obstacle.polygon.points.resize(4);
                polygon_obstacle.polygon.points[0].x= object.position.x + (object.dimensions[0]/2);
                polygon_obstacle.polygon.points[0].y= object.position.y + (object.dimensions[1]/2);
                polygon_obstacle.polygon.points[1].x= object.position.x + (object.dimensions[0]/2);
                polygon_obstacle.polygon.points[1].y= object.position.y - (object.dimensions[1]/2);
                polygon_obstacle.polygon.points[2].x= object.position.x - (object.dimensions[0]/2);
                polygon_obstacle.polygon.points[2].y= object.position.y - (object.dimensions[1]/2);
                polygon_obstacle.polygon.points[3].x= object.position.x - (object.dimensions[0]/2);
                polygon_obstacle.polygon.points[3].y= object.position.y + (object.dimensions[1]/2);
                obstacle_msg.obstacles.push_back(polygon_obstacle);

            if (visualise=true){ //marker array for rviz
                    bbox.header.frame_id = "map";
                    bbox.type = visualization_msgs::Marker::CUBE;
                    bbox.action = visualization_msgs::Marker::ADD;
                    bbox.color.r = 1.0;
                    bbox.color.a = 0.25;
                    // Update publisher with object positions
                    bbox.id = i;
                    bbox.pose.position.x = object.position.x;
                    bbox.pose.position.y = object.position.y;
                    bbox.pose.position.z = object.position.z;
                    bbox.scale.x = object.dimensions[0];
                    bbox.scale.y = object.dimensions[1];
                    bbox.scale.z = object.dimensions[2];
                    bbox.header.stamp = ros::Time::now();
                    // Publish marker
                    bbox_array.markers.push_back(bbox);
                }

            }
        // Publish the appropriate messages based on the flags
        std_msgs::Bool stop_msg;
        stop_msg.data = within1m;
        stop_obj_pub.publish(stop_msg);

        std_msgs::Bool slow_msg;
        slow_msg.data = within2m;
        slow_obj_pub.publish(slow_msg);
       
        //publish obstacles and bounding boxes
        obs_pub.publish(obstacle_msg);
        bbox_pub.publish(bbox_array);


    // Used to store sensors data.
    SensorsData sensors_data;

    // Used to store sensors timestamps and check if new data is available.
    TimestampHandler ts;

    // Retrieve sensors data 
    auto start_time = std::chrono::high_resolution_clock::now();

        if (zed.getSensorsData(sensors_data, TIME_REFERENCE::CURRENT) == ERROR_CODE::SUCCESS) {

            if (ts.isNew(sensors_data.imu)) {
                //IMU Data being assigned to ros publisher
                //cout << " \t Orientation: {" << sensors_data.imu.pose.getOrientation() << "}\n";
                //cout << " \t Acceleration: {" << sensors_data.imu.linear_acceleration << "} [m/sec^2]\n";
                //cout << " \t Angular Velocitiy: {" << sensors_data.imu.angular_velocity << "} [deg/sec]\n";
                imu_msg.header.stamp =ros::Time::now();
                imu_msg.header.frame_id = "imu_frame";
                imu_msg.orientation.x = sensors_data.imu.pose.getOrientation().ox;
                imu_msg.orientation.y = sensors_data.imu.pose.getOrientation().oy;
                imu_msg.orientation.z = sensors_data.imu.pose.getOrientation().oz;
                imu_msg.angular_velocity.x = sensors_data.imu.angular_velocity.x;
                imu_msg.angular_velocity.y = sensors_data.imu.angular_velocity.y;
                imu_msg.angular_velocity.z = sensors_data.imu.angular_velocity.z;
                imu_msg.linear_acceleration.x = sensors_data.imu.linear_acceleration.x;
                imu_msg.linear_acceleration.y = sensors_data.imu.linear_acceleration.y;
                imu_msg.linear_acceleration.z = sensors_data.imu.linear_acceleration.z;
                //Publish IMU
                imu_pub.publish(imu_msg);

                //magnetometer data being assigned to ros publisher
                //cout << " - Magnetometer\n \t Magnetic Field: {" << sensors_data.magnetometer.magnetic_field_calibrated << "} [uT]\n";
                magnetometer_msg.header.stamp = ros::Time::now();
                magnetometer_msg.header.frame_id = "base_link";
                magnetometer_msg.magnetic_field.x = sensors_data.magnetometer.magnetic_field_calibrated.x;
                magnetometer_msg.magnetic_field.y = sensors_data.magnetometer.magnetic_field_calibrated.y;
                magnetometer_msg.magnetic_field.z = sensors_data.magnetometer.magnetic_field_calibrated.z;
                //publish magnetometer
                magnetometer_pub.publish(magnetometer_msg);
        }
        }

        
    }
    }

    
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
