cmake_minimum_required(VERSION 3.16)
project(sk_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hint cmake at brew's CGAL install (Apple Silicon).
if(NOT CGAL_DIR AND EXISTS "/opt/homebrew/opt/cgal/lib/cmake/CGAL")
    set(CGAL_DIR "/opt/homebrew/opt/cgal/lib/cmake/CGAL")
endif()

find_package(CGAL REQUIRED)

add_executable(sk_solver sk_solver.cpp)
target_link_libraries(sk_solver PRIVATE CGAL::CGAL)

# Release build by default for performance (CGAL is heavy in debug).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
