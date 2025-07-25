cmake_minimum_required(VERSION 3.1)

project(bgslibrary VERSION 3.3.0)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(append_if condition value)
  if (${condition})
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endif()
endfunction()

#if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
#    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
#    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#  check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
#  append_if(SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG "-fvisibility=hidden -fvisibility-inlines-hidden" CMAKE_CXX_FLAGS)
#endif()

if(UNIX)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
  set(CMAKE_MACOSX_RPATH 1)
endif(UNIX)

# Avoid cmake warnings about changes in behavior of some Mac OS X path
# variable we don't care about.
if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

# compilation mode setup
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set(BUILD_SHARED_LIBS TRUE)
  #if(BGS_PYTHON_SUPPORT)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
  #else()
  #  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  #  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
  #endif()
endif(WIN32)

set(bgs_out_dir ".")
# First for the generic no-config case (e.g. with mingw)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${bgs_out_dir})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${bgs_out_dir})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${bgs_out_dir})
# Second, for multi-config builds (e.g. msvc)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${bgs_out_dir})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${bgs_out_dir})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${bgs_out_dir})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

if(UNIX)
  # add some standard warnings
  ADD_DEFINITIONS(-Wno-variadic-macros -Wno-long-long -Wall -Wextra -Winit-self -Woverloaded-virtual -Wsign-promo -Wno-unused-parameter -pedantic -Woverloaded-virtual -Wno-unknown-pragmas)

  # -ansi does not compile with sjn module
  #ADD_DEFINITIONS(-ansi)

  # if you like to have warinings about conversions, e.g. double->int or double->float etc., or float compare
  #ADD_DEFINITIONS(-Wconversion -Wfloat-equal)
endif(UNIX)

# cmake -D BGS_PYTHON_SUPPORT=ON ..
if(NOT DEFINED BGS_PYTHON_SUPPORT)
  set(BGS_PYTHON_SUPPORT OFF)
else()
  # add_definitions(-DBGS_PYTHON_SUPPORT)
endif()
# cmake -D BGS_PYTHON_ONLY=ON ..
if(NOT DEFINED BGS_PYTHON_ONLY)
  set(BGS_PYTHON_ONLY OFF)
else()
  # add_definitions(-DBGS_PYTHON_ONLY)
endif()
# cmake -D BGS_CORE_STATIC=ON ..
if(NOT DEFINED BGS_CORE_STATIC)
  set(BGS_CORE_STATIC OFF)
else()
  # add_definitions(-DBGS_CORE_STATIC)
endif()
message(STATUS "")
message(STATUS "BGS_PYTHON_SUPPORT: ${BGS_PYTHON_SUPPORT}")
message(STATUS "BGS_PYTHON_ONLY:    ${BGS_PYTHON_ONLY}")
message(STATUS "BGS_CORE_STATIC:    ${BGS_CORE_STATIC}")

# cmake -D BGS_PYTHON_SUPPORT=ON -D BGS_PYTHON_VERSION=3 ..
if(NOT DEFINED BGS_PYTHON_VERSION)
  set(BGS_PYTHON_VERSION 3)
endif()
if(BGS_PYTHON_SUPPORT)
  message(STATUS "PYTHON VERSION: ${BGS_PYTHON_VERSION}")
endif()

if(BGS_CORE_STATIC)
  set(OpenCV_STATIC ON)
else()
  set(OpenCV_STATIC OFF)
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "")
  message(STATUS "OpenCV library status:")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}\n")
endif()

# if(${OpenCV_VERSION} VERSION_EQUAL 3 OR ${OpenCV_VERSION} VERSION_GREATER 3)
#   message(FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
# endif()

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message(FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

if(BGS_PYTHON_SUPPORT)
  #if(WIN32)
  #  set(Boost_USE_STATIC_LIBS ON)
  #else()
  #  set(Boost_USE_STATIC_LIBS OFF)
  #endif()

  #set(Boost_USE_MULTITHREADED ON)
  #set(Boost_USE_STATIC_RUNTIME OFF)

  #message(STATUS "SEARCHING FOR BOOST COMPONENT FOR PYTHON ${BGS_PYTHON_VERSION}")
  #if(BGS_PYTHON_VERSION EQUAL 2)
  #  find_package(Boost REQUIRED COMPONENTS python)
  #else()
  #  find_package(Boost REQUIRED COMPONENTS python3)
  #endif()

  # Pybind11's cmake scripts enable link time optimization by default.  However,
  # it makes linking take a really long time and doesn't seem to substantively
  # improve runtime performance.  So we disable LTO here to make building bgslibrary
  # faster.
  set(PYBIND11_LTO_CXX_FLAGS "")

  #set(PYBIND11_PYTHON_VERSION 2.7 3.5 3.6)
  set(PYBIND11_PYTHON_VERSION ${BGS_PYTHON_VERSION})
  #find_package(pybind11 REQUIRED)
  add_subdirectory(modules/pybind11)

  #find_package(PythonInterp ${BGS_PYTHON_VERSION} REQUIRED)
  #find_package(PythonLibs ${BGS_PYTHON_VERSION} REQUIRED)

  #message(STATUS "Boost library status:")
  #message(STATUS "    version: ${Boost_VERSION}")
  #message(STATUS "    libraries: ${Boost_LIBRARIES}")
  #message(STATUS "    include path: ${Boost_INCLUDE_DIRS}")

  message(STATUS "")
  message(STATUS "Python library status:")
  message(STATUS "    executable: ${PYTHON_EXECUTABLE}")
  #message(STATUS "    version: ${PYTHON_VERSION_STRING}")
  #message(STATUS "    libraries: ${PYTHON_LIBRARIES}")
  message(STATUS "    library: ${PYTHON_LIBRARY}")
  message(STATUS "    include path: ${PYTHON_INCLUDE_DIRS}")
  if(NOT NUMPY_INCLUDE_DIR)
    # message(FATAL_ERROR "You must define NUMPY_INCLUDE_DIR by 'cmake -D NUMPY_INCLUDE_DIR=/python/lib/site-packages/numpy/core/include ..'")
    exec_program ("${PYTHON_EXECUTABLE}"
      ARGS "-c \"import numpy; print(numpy.get_include())\""
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
      RETURN_VALUE NUMPY_NOT_FOUND)
  endif()
  message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}\n")
endif()

if(NOT BGS_PYTHON_ONLY)
  file(GLOB main_src bgslibrary/*.cpp bgslibrary/*.c)
  file(GLOB main_inc bgslibrary/*.h bgslibrary/*.hpp)
endif()

file(GLOB_RECURSE utils_src bgslibrary/utils/*.cpp bgslibrary/utils/*.c)
file(GLOB_RECURSE utils_inc bgslibrary/utils/*.h bgslibrary/utils/*.hpp)

file(GLOB_RECURSE tools_src bgslibrary/tools/*.cpp bgslibrary/tools/*.c)
file(GLOB_RECURSE tools_inc bgslibrary/tools/*.h bgslibrary/tools/*.hpp)

file(GLOB_RECURSE bgs_src bgslibrary/algorithms/*.cpp bgslibrary/algorithms/*.c)
file(GLOB_RECURSE bgs_inc bgslibrary/algorithms/*.h bgslibrary/algorithms/*.hpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

if(BGS_PYTHON_SUPPORT)
  file(GLOB_RECURSE bgs_python_src wrapper/python/*.cpp)
  file(GLOB_RECURSE bgs_python_inc wrapper/python/*.h)

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules/pybind11/include)
  #include_directories(${Boost_INCLUDE_DIRS})
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${NUMPY_INCLUDE_DIR})
endif()

# GMG is not available in older OpenCV versions
if(${OpenCV_VERSION} VERSION_LESS 2.4.3)
  file(GLOB gmg bgslibrary/algorithms/GMG.cpp)
  list(REMOVE_ITEM bgs_src ${gmg})
endif()

if(BGS_CORE_STATIC)
  message(STATUS "Bulding bgslibrary_core STATIC")
  add_library(bgslibrary_core STATIC ${bgs_src} ${tools_src} ${utils_src} ${bgs_inc} ${tools_inc} ${utils_inc})
  #set_property(TARGET bgslibrary_core PROPERTY POSITION_INDEPENDENT_CODE ON)
else()
  message(STATUS "Bulding bgslibrary_core SHARED")
  add_library(bgslibrary_core SHARED ${bgs_src} ${tools_src} ${utils_src} ${bgs_inc} ${tools_inc} ${utils_inc})
  target_link_libraries(bgslibrary_core ${OpenCV_LIBS})
  # generates the export header bgslibrary_core_EXPORTS.h automatically
  include(GenerateExportHeader)
  GENERATE_EXPORT_HEADER(bgslibrary_core
      BASE_NAME bgslibrary_core
      EXPORT_MACRO_NAME bgslibrary_core_EXPORTS
      EXPORT_FILE_NAME bgslibrary_core_EXPORTS.h
      STATIC_DEFINE BGSLIBRARY_CORE_EXPORTS_BUILT_AS_STATIC)
  #set_property(TARGET bgslibrary_core PROPERTY PUBLIC_HEADER ${bgs_inc} ${tools_inc} ${utils_inc})
endif()

if(BGS_PYTHON_SUPPORT)
  #add_library(bgs_python SHARED ${bgs_src} ${tools_src})
  #pybind11_add_module(bgs_python ${bgs_src} ${tools_src})
  pybind11_add_module(bgs_python ${bgs_python_src} ${bgs_python_inc})

  target_link_libraries(bgs_python PRIVATE bgslibrary_core ${OpenCV_LIBS} ${PYTHON_LIBRARY} pybind11::module)
  #target_link_libraries(bgs_python ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARY})
  #target_link_libraries(bgs_python ${OpenCV_LIBS} ${PYTHON_LIBRARY} pybind11::module)
  #target_link_libraries(bgs_python PRIVATE ${OpenCV_LIBS} ${PYTHON_LIBRARY} pybind11::embed)

  #set_target_properties(bgs_python PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
  set_target_properties(bgs_python PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

  target_compile_definitions(bgs_python PRIVATE BGS_PYTHON_SUPPORT=1)

  # Set the output library name to bgslibrary because that's what setup.py and distutils expects.
  set_property(TARGET bgs_python PROPERTY OUTPUT_NAME "pybgs")
  #set_property(TARGET bgs_python PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

#if(WIN32)
#  # set_property(TARGET bgslibrary_core PROPERTY SUFFIX ".lib")
#  #if(BGS_PYTHON_SUPPORT)
#  #  set_property(TARGET bgslibrary_core PROPERTY SUFFIX ".pyd")
#  #endif()
#else()
#  set_property(TARGET bgslibrary_core PROPERTY OUTPUT_NAME "bgs")
#endif()

#if(APPLE)
#  if(BGS_PYTHON_SUPPORT)
#    set_property(TARGET bgslibrary_core PROPERTY SUFFIX ".so")
#    set_target_properties(bgslibrary_core PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
#  endif()
#endif()

if(NOT BGS_PYTHON_ONLY)
  add_executable(bgslibrary ${main_src} ${main_inc})
  target_link_libraries(bgslibrary ${OpenCV_LIBS} bgslibrary_core)
  # set_target_properties(bgslibrary PROPERTIES OUTPUT_NAME bgs)
endif()

if(UNIX AND BGS_PYTHON_SUPPORT)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
      from distutils import sysconfig as sc
      print(sc.get_python_lib(prefix='', plat_specific=True))"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  message(STATUS "")
  message(STATUS "The bgslibrary python package will be installed at: ${PYTHON_SITE}\n")

  install(TARGETS bgs_python DESTINATION ${PYTHON_SITE})
  #install(FILES ${mypackage_python_files} DESTINATION ${PYTHON_SITE}/mypackage)
  #install(TARGETS bgs_python DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(NOT BGS_PYTHON_ONLY)
  include(GNUInstallDirs)
  set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/BGSLibrary)
  target_include_directories(bgslibrary_core
    PUBLIC
        $<INSTALL_INTERFACE:include>)


  install(TARGETS bgslibrary_core
    bgslibrary
    EXPORT bgslibrary_core-targets
    RUNTIME DESTINATION bin COMPONENT app
    LIBRARY DESTINATION lib COMPONENT runtime
    ARCHIVE DESTINATION lib COMPONENT runtime
    #PUBLIC_HEADER DESTINATION include/bgslibrary COMPONENT dev
    FRAMEWORK DESTINATION "/Library/Frameworks"
  )
  set_target_properties(bgslibrary_core PROPERTIES EXPORT_NAME BGSLibrary)

  install(DIRECTORY bgslibrary/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bgslibrary)
  install(EXPORT bgslibrary_core-targets
    FILE
      BGSLibraryTargets.cmake
    NAMESPACE
      BGSLibrary::
    DESTINATION
      ${INSTALL_CONFIGDIR}
  )

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/BGSLibraryConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake-modules/BGSLibraryConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/BGSLibraryConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
  )

  install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/cmake-modules/FindOpenCV.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BGSLibraryConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BGSLibraryConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
  )

  if(UNIX)
    # to avoid: error while loading shared libraries: libbgslibrary_core.so
    message(STATUS "You might need to run:")
    message(STATUS "$ LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib")
    message(STATUS "$ export LD_LIBRARY_PATH")
    message(STATUS "after 'make install' to avoid error while loading libbgslibrary_core\n")
  endif()

  if(WIN32)
    message(STATUS "You might need to add ${CMAKE_CURRENT_BINARY_DIR} to your PATH to be able to run your applications.")
    message(STATUS "> set PATH=%PATH%;${CMAKE_CURRENT_BINARY_DIR}\n")
  endif()
endif()
