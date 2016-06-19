cmake_minimum_required (VERSION 2.6)

project (KalmanFilter)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE DEBUG)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++11 -O3 -w")
set(CMAKE_CXX_FLAGS_RELEASE "-g -std=c++11 -O3 -w")
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


############################# LOAD BOOST #############################
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS python system thread timer)
if (NOT Boost_FOUND)
    set(Boost_INCLUDE_DIRS "${BOOST_INCLUDE_DIRS}")
    IF ( NOT Boost_INCLUDE_DIRS )
        message(FATAL_ERROR "Please point the environment variable BOOST_INCLUDE_DIRS to the include directory of Boost")
    ENDIF()
endif ()
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

############################# LOAD EIGEN #############################
if(PKG_CONFIG_FOUND)
    pkg_check_modules(EIGEN eigen3)
    if(NOT EIGEN_FOUND)
       message(FATAL_ERROR "EIGEN could not be found")
    endif()
    include_directories(${EIGEN_INCLUDE_DIRS}) 
    message("-- Eigen include dirs: ${EIGEN_INCLUDE_DIRS}")
    message("-- Eigen library dirs: ${EIGEN_LIBRARY_DIRS}")   
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

############################# LOAD FCL #############################
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FCL fcl)
    if(NOT FCL_FOUND)
       message(FATAL_ERROR "FCL could not be found")
    endif()
    include_directories(${FCL_INCLUDE_DIRS})
    link_directories(${FCL_LIBRARY_DIRS}) 
    message("-- FCL LIB DIRS ${FCL_LIBRARY_DIRS}")    
endif()


add_library(KalmanFilter SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/kalman_filter.cpp)
            
add_library(PathEvaluator SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/path_evaluator.cpp)           

target_link_libraries (KalmanFilter                      
                       ${Boost_LIBRARIES}
                       ${PYTHON_LIBRARIES})
                       
target_link_libraries (PathEvaluator                     
                       ${Boost_LIBRARIES}
                       ${PYTHON_LIBRARIES}
                       ${ROBOTS_LIBRARIES}
                       ${ROBOT_ENVIRONMENT_LIBRARIES}
                       ${FCL_LIBRARIES})

set(pkg_conf_file_in "${CMAKE_CURRENT_SOURCE_DIR}/kalman_filter.pc.in")
set(pkg_conf_file_out "${CMAKE_CURRENT_BINARY_DIR}/kalman_filter.pc")
configure_file("${pkg_conf_file_in}" "${pkg_conf_file_out}" @ONLY)

install(TARGETS KalmanFilter DESTINATION lib)
install(TARGETS PathEvaluator DESTINATION lib)
install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hxx" PATTERN "*.hpp"
  PATTERN ".DS_Store" EXCLUDE
)
install(FILES "${pkg_conf_file_out}" DESTINATION lib/pkgconfig/ COMPONENT pkgconfig)


