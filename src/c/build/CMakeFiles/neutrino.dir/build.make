# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bread/Documents/projects/neutrino/src/c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bread/Documents/projects/neutrino/src/c/build

# Include any dependencies generated for this target.
include CMakeFiles/neutrino.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neutrino.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neutrino.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neutrino.dir/flags.make

CMakeFiles/neutrino.dir/main.c.o: CMakeFiles/neutrino.dir/flags.make
CMakeFiles/neutrino.dir/main.c.o: /home/bread/Documents/projects/neutrino/src/c/main.c
CMakeFiles/neutrino.dir/main.c.o: CMakeFiles/neutrino.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bread/Documents/projects/neutrino/src/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/neutrino.dir/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/neutrino.dir/main.c.o -MF CMakeFiles/neutrino.dir/main.c.o.d -o CMakeFiles/neutrino.dir/main.c.o -c /home/bread/Documents/projects/neutrino/src/c/main.c

CMakeFiles/neutrino.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/neutrino.dir/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/bread/Documents/projects/neutrino/src/c/main.c > CMakeFiles/neutrino.dir/main.c.i

CMakeFiles/neutrino.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/neutrino.dir/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/bread/Documents/projects/neutrino/src/c/main.c -o CMakeFiles/neutrino.dir/main.c.s

# Object files for target neutrino
neutrino_OBJECTS = \
"CMakeFiles/neutrino.dir/main.c.o"

# External object files for target neutrino
neutrino_EXTERNAL_OBJECTS =

neutrino: CMakeFiles/neutrino.dir/main.c.o
neutrino: CMakeFiles/neutrino.dir/build.make
neutrino: CMakeFiles/neutrino.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bread/Documents/projects/neutrino/src/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable neutrino"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neutrino.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neutrino.dir/build: neutrino
.PHONY : CMakeFiles/neutrino.dir/build

CMakeFiles/neutrino.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neutrino.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neutrino.dir/clean

CMakeFiles/neutrino.dir/depend:
	cd /home/bread/Documents/projects/neutrino/src/c/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bread/Documents/projects/neutrino/src/c /home/bread/Documents/projects/neutrino/src/c /home/bread/Documents/projects/neutrino/src/c/build /home/bread/Documents/projects/neutrino/src/c/build /home/bread/Documents/projects/neutrino/src/c/build/CMakeFiles/neutrino.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neutrino.dir/depend

