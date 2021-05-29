# CMakeList.txt: image_stitching 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.17)

# 将源代码添加到此项目的可执行文件。
add_executable(image_stitching "image_stitching.cpp" "serializer.cpp" "cropper.cpp")

find_package(OpenCV REQUIRED)
find_path(LIBEXIF_INCLUDE_PATH libexif REQUIRED)
find_library(LIBEXIF_LIB_PATH libexif REQUIRED)
set_target_properties(image_stitching PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CMAKE_CXX_EXTENSIONS OFF
)

target_include_directories(image_stitching PRIVATE ${OpenCV_INCLUDE_DIRS} ${LIBEXIF_INCLUDE_PATH})
target_link_libraries(image_stitching PRIVATE ${OpenCV_LIBS} ${LIBEXIF_LIB_PATH})


# TODO: 如有需要，请添加测试并安装目标。
