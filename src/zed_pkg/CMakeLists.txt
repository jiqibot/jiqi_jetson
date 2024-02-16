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
  ##new
  OpenCV
  ##CUDA
  ##GLUT
  ##GLEW
  ##Opengl
  ##ZED 3
  ##genmsg ##added by kai
)

##added by kai
## Declare ROS messages and services
##add_message_files(DIRECTORY msg FILES Num.msg)
##add_service_files(DIRECTORY srv FILES AddTwoInts.srv)

## Generate added messages and services
##generate_messages(DEPENDENCIES std_msgs)
##added by kai


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
  ${OpenCV_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  
)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

add_executable (zed2Cam src/main.cpp src/GLViewer.cpp src/TrackingViewer.cpp)
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

##added by kai
add_executable (main src/main.cpp src/GLViewer.cpp src/TrackingViewer.cpp)
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

##added by kai
##add_executable(listener src/listener.cpp)
##target_link_libraries(listener ${catkin_LIBRARIES})
##add_dependencies(listener zed_pkg_generate_messages_cpp)


