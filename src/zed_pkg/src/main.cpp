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

#include <sl/Camera.hpp>
#include "utils.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0;
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

//print error messages
void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

//convert opencv rectangle to zed bounding box
std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

int main(int argc, char** argv) {

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

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.coordinate_units = sl::UNIT::METER; 
    init_parameters.camera_resolution = sl::RESOLUTION::HD1080;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP; // RVIZ's coordinate system is right_handed

    //if using video file
    if (argc >= 2) {
        std::string zed_opt = argv[1];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    zed.enablePositionalTracking();
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    // Define the model as custom box object to specify that the inference is done externally
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;


    //Declare all sensor variables
    sl::Mat left_sl, point_cloud;
    sl::Mat left_image, right_image;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose, prev_pose;
    sl::SensorsData sensors_data;
    sl::SensorsData::IMUData imu_data;
    cam_w_pose.pose_data.setIdentity();
    // ---------

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
    while (true) {

        if (zed.grab() == sl::ERROR_CODE::SUCCESS) { //if camera succesfully initialises

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

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
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

            // Loop through all detected objects
            for (int i = 0; i < objects.object_list.size(); i++) {
                
                sl::ObjectData object = objects.object_list[i];

                unsigned int object_id = object.id; // Get the object id

                //if person is too close publish bools to change robot speed.
                if (object.position.x <=1 && object_id == 1){ //object_label ==1 ==person
                    std_msgs::Bool bool_msg;
                    bool_msg.data=true;
                    stop_obj_pub.publish(bool_msg);
                }

                if (object.position.x <=3 && object_id == 1){ //add and if object class = person
                    std_msgs::Bool bool_msg;
                    bool_msg.data=true;
                    slow_obj_pub.publish(bool_msg);
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

                if (visualise=true){ //can publish a markerArray for visualisation purposes in rviz, parameter set in launch file
                    bbox.header.frame_id = "map";
                    //bbox.ns = "object_detection";
                    bbox.type = visualization_msgs::Marker::CUBE;
                    bbox.action = visualization_msgs::Marker::ADD;
                    bbox.color.r = 1.0;
                    bbox.color.a = 0.25;

                    // Update marker message
                    bbox.id = i;
                    bbox.pose.position.x = object.position.x;
                    bbox.pose.position.y = object.position.y;
                    bbox.pose.position.z = object.position.z;
                    bbox.scale.x = object.dimensions[0];
                    bbox.scale.y = object.dimensions[1];
                    bbox.scale.z = object.dimensions[2];
                    bbox.header.stamp = ros::Time::now();

                    // Publish marker message
                    bbox_array.markers.push_back(bbox);
                }
            }
            obs_pub.publish(obstacle_msg);
            bbox_pub.publish(bbox_array);
            

            zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
            zed.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT);

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

        }  
    }
    return 0;
}
