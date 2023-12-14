include(cpp/CMakeLists.txt)
cmake_minimum_required(VERSION 3.0.2)
project(zed_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  geometry_msgs
  costmap_converter
  nav_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES zed_pkg
  LIBRARIES opencv_dnn_zed
  CATKIN_DEPENDS roscpp
  DEPENDS system_lib
  CATKIN_DEPENDS sensor_msgs
  CATKIN_DEPENDS std_msgs
  CATKIN_DEPENDS visualization_msgs
  CATKIN_DEPENDS geometry_msgs
  CATKIN_DEPENDS costmap_converter
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
)

add_executable (zed2Cam src/main.cpp src/GLViewer.cpp)
target_link_libraries(zed2Cam
                        ${SPECIAL_OS_LIBS}
                        ${ZED_LIBS}
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES}
                        ${OpenCV_LIBRARIES}
			${catkin_LIBRARIES}
			#${PROJECT_NAME}
)

add_executable (main src/main.cpp)
target_link_libraries(main 
                        ${SPECIAL_OS_LIBS}
                        ${ZED_LIBS}
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES}
                        ${OpenCV_LIBRARIES}
			${catkin_LIBRARIES}
			#${PROJECT_NAME}
)


