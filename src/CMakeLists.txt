# Runtime targets will be added milestone by milestone.
add_executable(cuda_smoke_test cuda_smoke_test.cu)

target_compile_features(cuda_smoke_test PRIVATE cxx_std_20)

add_test(NAME cuda_smoke_test COMMAND cuda_smoke_test)
