# d-SEAMS in-situ analysis. The fix source matches the out-of-tree plugin.

set(SEAMS_INCLUDE "" CACHE PATH "Directory that contains seams_insitu.h")
set(SEAMS_LIBRARY "" CACHE FILEPATH "Path to libyodaLib.so")

if(NOT SEAMS_INCLUDE OR NOT SEAMS_LIBRARY)
  message(FATAL_ERROR
    "PKG_DSEAMS needs -DSEAMS_INCLUDE and -DSEAMS_LIBRARY. "
    "Build the plugin and this package against the same LAMMPS version.")
endif()

if(NOT EXISTS "${SEAMS_INCLUDE}/seams_insitu.h")
  message(FATAL_ERROR "seams_insitu.h is not in ${SEAMS_INCLUDE}")
endif()
if(NOT EXISTS "${SEAMS_LIBRARY}")
  message(FATAL_ERROR "libyodaLib is not at ${SEAMS_LIBRARY}")
endif()

target_include_directories(lammps PRIVATE "${SEAMS_INCLUDE}")
target_link_libraries(lammps PRIVATE "${SEAMS_LIBRARY}")
get_filename_component(SEAMS_LIBDIR "${SEAMS_LIBRARY}" DIRECTORY)
set_target_properties(lammps PROPERTIES BUILD_RPATH "${SEAMS_LIBDIR}")
