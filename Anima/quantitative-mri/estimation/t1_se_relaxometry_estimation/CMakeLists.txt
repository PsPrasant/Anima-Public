if(BUILD_TOOLS AND USE_NLOPT AND NLOPT_FOUND)

project(animaT1SERelaxometryEstimation)

## #############################################################################
## List Sources
## #############################################################################

list_source_files(${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

## #############################################################################
## add executable
## #############################################################################

add_executable(${PROJECT_NAME}
  ${${PROJECT_NAME}_CFILES}
  )


## #############################################################################
## Link
## #############################################################################

target_link_libraries(${PROJECT_NAME}
  ${ITKIO_LIBRARIES}
  AnimaOptimizers
  AnimaRelaxometry
  )

## #############################################################################
## install
## #############################################################################

set_exe_install_rules(${PROJECT_NAME})

endif()
