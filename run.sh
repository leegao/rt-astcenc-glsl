glslangValidator -V -Os --target-env vulkan1.1 astc_compress.comp -o astc_compress.spv
g++ astc.cpp -lvulkan -o astcencrt
./astcencrt