cmake_minimum_required (VERSION 2.6)

project (KalmanFilter)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE DEBUG)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++11 -O3 -w")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -w")
SET(BOOST_MIN_VERSION "1.54.0")
SET(PYTHON_MIN_VERSION "2.7.0")

include(FindPkgConfig)
include(GNUInstallDirs)
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/include")
############################# LOAD PYTHON #############################
if(PKG_CONFIG_FOUND)    
    pkg_check_modules(PYTHON python-2.7)
    if(NOT PYTHON_FOUND)
       message(FATAL_ERROR "Python could not be found")
    endif()
    include_directories(${PYTHON_INCLUDE_DIRS})
    link_directories(${PYTHON_LIBRARY_DIRS})    
endif()

############################# LOAD EIGEN #############################
if(PKG_CONFIG_FOUND)
    pkg_check_modules(EIGEN eigen3)
    if(NOT EIGEN_FOUND)
       message(FATAL_ERROR "EIGEN could not be found")
    endif()
    include_directories(${EIGEN_INCLUDE_DIRS}) 
    link_directories(${EIGEN_LIBRARY_DIRS}) 
endif()

############################# LOAD FRAPU_CORE #############################
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FRAPU_CORE frapu_core)
    if(NOT FRAPU_CORE_FOUND)
       message(FATAL_ERROR "FRAPU_CORE could not be found")
    endif()
    include_directories(${FRAPU_CORE_INCLUDE_DIRS})
    link_directories(${FRAPU_CORE_LIBRARY_DIRS}) 
    message("-- FRAPU_CORE LIB DIRS ${FRAPU_CORE_LIBRARY_DIRS}")    
endif()

############################# LOAD ROBOTS #############################
if(PKG_CONFIG_FOUND)    
    pkg_check_modules(ROBOTS robots)
    if(NOT ROBOTS_FOUND)
       message(FATAL_ERROR "ROBOTS could not be found")
    endif()
    include_directories(${ROBOTS_INCLUDE_DIRS})
    link_directories(${ROBOTS_LIBRARY_DIRS})     
endif()

############################# LOAD ROBOT_ENVIRONMENT #############################
if(PKG_CONFIG_FOUND)    
    pkg_check_modules(ROBOT_ENVIRONMENT robot_environment)
    if(NOT ROBOTS_FOUND)
       message(FATAL_ERROR "ROBOT_ENVIRONMENT could not be found")
    endif()
    include_directories(${ROBOT_ENVIRONMENT_INCLUDE_DIRS})
    link_directories(${ROBOT_ENVIRONMENT_LIBRARY_DIRS})     
endif()

############################# LOAD PATH_PLANNER #############################
if(PKG_CONFIG_FOUND)    
    pkg_check_modules(PATH_PLANNER path_planner)
    if(NOT PATH_PLANNER_FOUND)
       message(FATAL_ERROR "PATH_PLANNER could not be found")
    endif()
    include_directories(${PATH_PLANNER_INCLUDE_DIRS})
    link_directories(${PATH_PLANNER_LIBRARY_DIRS})     
endif()

############################# LOAD FCL #############################
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FCL fcl)
    if(NOT FCL_FOUND)
       message(FATAL_ERROR "FCL could not be found")
    endif()
    include_directories(${FCL_INCLUDE_DIRS})
    link_directories(${FCL_LIBRARY_DIRS})     
endif()

set(pkg_conf_file_in "${CMAKE_CURRENT_SOURCE_DIR}/kalman_filter.pc.in")
set(pkg_conf_file_out "${CMAKE_CURRENT_BINARY_DIR}/kalman_filter.pc")
configure_file("${pkg_conf_file_in}" "${pkg_conf_file_out}" @ONLY)

install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kalman_filter
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hxx" PATTERN "*.hpp"
  PATTERN ".DS_Store" EXCLUDE
)
install(FILES "${pkg_conf_file_out}" DESTINATION lib/pkgconfig/ COMPONENT pkgconfig)


