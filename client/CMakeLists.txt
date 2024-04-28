cmake_minimum_required(VERSION 2.8)

project( Vision )

find_package( OpenCV REQUIRED )

# Add the Tesseract include directory
include_directories( /usr/include/tesseract )

# Find the Tesseract library
find_library( Tesseract_LIBRARY NAMES tesseract HINTS /usr/lib /usr/lib64 )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( main main.cpp )

target_link_libraries( main ${OpenCV_LIBS} )

# Link against the Tesseract library
if(Tesseract_LIBRARY)
  target_link_libraries( main ${Tesseract_LIBRARY} )
else()
  message(FATAL_ERROR "Tesseract library not found!")
endif()
