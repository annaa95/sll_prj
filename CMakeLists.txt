cmake_minimum_required(VERSION 2.8.3)

project(sll_extern_roscontroller)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  std_srvs
  message_generation
  )

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_service_files(DIRECTORY srv FILES
  deliverNewAction.srv
  supplyFirstCntc.srv
)

generate_messages(DEPENDENCIES
  std_msgs
  )
###################################
## catkin specific configuration ##
###################################

catkin_package(  
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  std_srvs
)

if (CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(tests/roslaunch_test.xml)
endif()

#############
## Install ##
#############

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_install_python(PROGRAMS scripts/sll_runner.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


