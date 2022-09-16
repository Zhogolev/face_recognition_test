# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.23.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.23.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/konstantin/projects/face_recognition/demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/konstantin/projects/face_recognition/demo

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/main.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/main.cpp.o: main.cpp
CMakeFiles/demo.dir/main.cpp.o: CMakeFiles/demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/konstantin/projects/face_recognition/demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo.dir/main.cpp.o -MF CMakeFiles/demo.dir/main.cpp.o.d -o CMakeFiles/demo.dir/main.cpp.o -c /Users/konstantin/projects/face_recognition/demo/main.cpp

CMakeFiles/demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/konstantin/projects/face_recognition/demo/main.cpp > CMakeFiles/demo.dir/main.cpp.i

CMakeFiles/demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/konstantin/projects/face_recognition/demo/main.cpp -o CMakeFiles/demo.dir/main.cpp.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/main.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/main.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_gapi.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_stitching.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_aruco.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_bgsegm.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_bioinspired.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_ccalib.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_dnn_objdetect.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_dnn_superres.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_dpm.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_face.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_fuzzy.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_hfs.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_img_hash.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_line_descriptor.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_quality.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_reg.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_rgbd.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_saliency.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_stereo.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_structured_light.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_superres.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_surface_matching.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_tracking.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_videostab.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_xfeatures2d.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_xobjdetect.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_xphoto.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_shape.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_highgui.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_datasets.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_plot.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_text.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_dnn.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_ml.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_phase_unwrapping.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_optflow.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_ximgproc.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_video.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_videoio.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_imgcodecs.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_objdetect.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_calib3d.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_features2d.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_flann.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_photo.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_imgproc.4.2.0.dylib
demo: /Users/konstantin/projects/opencv/build_opencv/lib/libopencv_core.4.2.0.dylib
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/konstantin/projects/face_recognition/demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo
.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /Users/konstantin/projects/face_recognition/demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/konstantin/projects/face_recognition/demo /Users/konstantin/projects/face_recognition/demo /Users/konstantin/projects/face_recognition/demo /Users/konstantin/projects/face_recognition/demo /Users/konstantin/projects/face_recognition/demo/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

