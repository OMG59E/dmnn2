# ==============================================================================
# dmnn2 Samples Configuration
# ==============================================================================

set(SRC_FILE_EXTENSIONS c cpp cu cc)
set(LINT_DIRS samples)

cmake_policy(SET CMP0009 NEW)

# Find all sample source files
foreach(ext ${SRC_FILE_EXTENSIONS})
    foreach(dir ${LINT_DIRS})
        file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
        set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
    endforeach()
endforeach()

set(SAMPLE_SOURCES ${LINT_SOURCES} PARENT_SCOPE)