
set(SRC_FILE_EXTENSIONS h hpp hu c cpp cu cuh cc)
set(EXCLUDE_FILE_EXTENSTIONS pb.h pb.cc)
set(LINT_DIRS include src)

set(EXCLUDE_DIRS jetson_multimedia_api)

cmake_policy(SET CMP0009 NEW)  # suppress cmake warning

# find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
    foreach(dir ${LINT_DIRS})
        file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
        # set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
        foreach(f_var ${FOUND_FILES})
            set(exist FALSE)
            foreach(exclude_dir ${EXCLUDE_DIRS})
                string(FIND ${f_var} "/${exclude_dir}/" output_var REVERSE)
                if(NOT ${output_var} MATCHES -1)
                    set(exist TRUE)
                endif()
            endforeach()

            if(NOT exist)
                list(APPEND LINT_SOURCES ${f_var})
            endif()
        endforeach()
    endforeach()
endforeach()

set(DMNN_SOURCES ${LINT_SOURCES} PARENT_SCOPE)