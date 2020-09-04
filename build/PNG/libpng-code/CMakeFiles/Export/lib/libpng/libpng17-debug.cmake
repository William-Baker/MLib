#----------------------------------------------------------------
# Generated CMake target import file for configuration "DEBUG".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "png" for configuration "DEBUG"
set_property(TARGET png APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(png PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libm.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libpng17d.so.17.0.0"
  IMPORTED_SONAME_DEBUG "libpng17d.so.17"
  )

list(APPEND _IMPORT_CHECK_TARGETS png )
list(APPEND _IMPORT_CHECK_FILES_FOR_png "${_IMPORT_PREFIX}/lib/libpng17d.so.17.0.0" )

# Import target "png_static" for configuration "DEBUG"
set_property(TARGET png_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(png_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libm.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libpng17d.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS png_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_png_static "${_IMPORT_PREFIX}/lib/libpng17d.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
