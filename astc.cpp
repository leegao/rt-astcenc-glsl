#include <vulkan/vulkan.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f)                                                     \
    {                                                                          \
        VkResult res = (f);                                                    \
        if (res != VK_SUCCESS)                                                 \
        {                                                                      \
            std::cerr << "Fatal : VkResult is " << res << " in " << __FILE__    \
                      << " at line " << __LINE__ << std::endl;                  \
            assert(res == VK_SUCCESS);                                         \
        }                                                                      \
    }

struct ImageData {
    std::vector<float> tiledData;
    uint32_t totalBlocks;
    int originalWidth;
    int originalHeight;
};

ImageData prepare_input_data(const char* filename) {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image file!");
    }

    std::vector<float> img_data(texWidth * texHeight * 4);
    for (size_t i = 0; i < (size_t)texWidth * texHeight * 4; ++i) {
        img_data[i] = static_cast<float>(pixels[i]) / 255.0f;
    }
    stbi_image_free(pixels);

    int padded_width = (texWidth + 3) / 4 * 4;
    int padded_height = (texHeight + 3) / 4 * 4;
    uint32_t total_blocks = (padded_width / 4) * (padded_height / 4);

    std::vector<float> padded_array(padded_width * padded_height * 4, 0.0f);
    for (int y = 0; y < texHeight; ++y) {
        for (int x = 0; x < texWidth; ++x) {
            for (int c = 0; c < 4; ++c) {
                padded_array[(y * padded_width + x) * 4 + c] = img_data[(y * texWidth + x) * 4 + c];
            }
        }
    }

    std::vector<float> tiled_data(total_blocks * 16 * 4);
    for (uint32_t by = 0; by < (uint32_t)padded_height / 4; ++by) {
        for (uint32_t bx = 0; bx < (uint32_t)padded_width / 4; ++bx) {
            uint32_t block_idx = by * (padded_width / 4) + bx;
            for (uint32_t py = 0; py < 4; ++py) {
                for (uint32_t px = 0; px < 4; ++px) {
                    uint32_t pixel_in_block_idx = py * 4 + px;
                    uint32_t src_x = bx * 4 + px;
                    uint32_t src_y = by * 4 + py;
                    for (int c = 0; c < 4; ++c) {
                        tiled_data[(block_idx * 16 + pixel_in_block_idx) * 4 + c] =
                            padded_array[(src_y * padded_width + src_x) * 4 + c];
                    }
                }
            }
        }
    }
    return {tiled_data, total_blocks, texWidth, texHeight};
}

void save_astc_file(const char* filename, const std::vector<uint32_t>& results, int width, int height) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error writing ASTC file: " << filename << std::endl;
        return;
    }

    char header[16];
    header[0] = 0x13; header[1] = 0xAB; header[2] = 0xA1; header[3] = 0x5C;
    header[4] = 4; header[5] = 4; header[6] = 1;
    header[7] = width & 0xFF; header[8] = (width >> 8) & 0xFF; header[9] = (width >> 16) & 0xFF;
    header[10] = height & 0xFF; header[11] = (height >> 8) & 0xFF; header[12] = (height >> 16) & 0xFF;
    header[13] = 1; header[14] = 0; header[15] = 0;

    file.write(header, 16);
    file.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(uint32_t));
    file.close();
    std::cout << "Successfully saved compressed data to '" << filename << "'" << std::endl;
}

void save_decoded_image(const char* filename, const std::vector<float>& flat_pixel_data, int original_width, int original_height) {
    int padded_width = (original_width + 3) / 4 * 4;
    int padded_height = (original_height + 3) / 4 * 4;
    uint32_t num_blocks_x = padded_width / 4;
    uint32_t num_blocks_y = padded_height / 4;

    std::vector<float> padded_image(padded_width * padded_height * 4);
    for (uint32_t by = 0; by < num_blocks_y; ++by) {
        for (uint32_t bx = 0; bx < num_blocks_x; ++bx) {
            uint32_t block_idx = by * num_blocks_x + bx;
            for (uint32_t py = 0; py < 4; ++py) {
                for (uint32_t px = 0; px < 4; ++px) {
                    uint32_t pixel_in_block_idx = py * 4 + px;
                    uint32_t dst_x = bx * 4 + px;
                    uint32_t dst_y = by * 4 + py;
                    for (int c = 0; c < 4; ++c) {
                        float val = flat_pixel_data[(block_idx * 16 + pixel_in_block_idx) * 4 + c];
                        padded_image[(dst_y * padded_width + dst_x) * 4 + c] = val;
                    }
                }
            }
        }
    }

    std::vector<unsigned char> image_uint8(original_width * original_height * 4);
    for (int y = 0; y < original_height; ++y) {
        for (int x = 0; x < original_width; ++x) {
            for (int c = 0; c < 4; ++c) {
                float val = padded_image[(y * padded_width + x) * 4 + c];
                image_uint8[(y * original_width + x) * 4 + c] = static_cast<unsigned char>(std::min(std::max(val, 0.0f), 1.0f) * 255.0f);
            }
        }
    }

    stbi_write_png(filename, original_width, original_height, 4, image_uint8.data(), original_width * 4);
    std::cout << "Successfully saved decoded image to '" << filename << "'" << std::endl;
}


class ASTCComputeApp {
private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    VkBuffer inputPixelBuffer;
    VkDeviceMemory inputPixelBufferMemory;
    VkBuffer compressedBuffer;
    VkDeviceMemory compressedBufferMemory;
    VkBuffer decodedBuffer;
    VkDeviceMemory decodedBufferMemory;

    VkBuffer readbackCompressedBuffer;
    VkDeviceMemory readbackCompressedMemory;
    VkBuffer readbackDecodedBuffer;
    VkDeviceMemory readbackDecodedMemory;

    VkQueue queue;
    uint32_t queueFamilyIndex;

    VkQueryPool queryPool;
    float timestampPeriod;

    ImageData imageData;
    uint32_t numWorkGroups;
    VkDeviceSize compressedBufferSize, decodedBufferSize;

public:
    void run() {
        imageData = prepare_input_data("screenshot.jpg");
        if (imageData.tiledData.empty()) return;

        numWorkGroups = (imageData.totalBlocks + 3) / 4;
        uint32_t total_blocks_padded = numWorkGroups * 4;
        compressedBufferSize = total_blocks_padded * 16;
        decodedBufferSize = imageData.tiledData.size() * sizeof(float);
        
        createInstance();
        findPhysicalDevice();
        createDevice();
        createQueryPool();
        createCommandPool(); // Command pool must exist before we can use it for copies
        createBuffers();
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSet();
        createComputePipeline();
        createAndRecordCommandBuffer();
        runCommandBuffer();
        processResults();
        cleanup();
    }

    static std::vector<char> readShaderFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    void createInstance() {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "ASTC Compute";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &instance));
    }

    void findPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physicalDevice = devices[0];
        
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "Using device: " << props.deviceName << std::endl;

        // Get the timestamp period for converting timestamps to nanoseconds.
        timestampPeriod = props.limits.timestampPeriod;
        std::cout << "Device timestamp period: " << timestampPeriod << " ns" << std::endl;
    }
    
    uint32_t getComputeQueueFamilyIndex() {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return i;
            }
        }
        throw std::runtime_error("failed to find a compute queue family!");
    }


    void createDevice() {
        queueFamilyIndex = getComputeQueueFamilyIndex();
        
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        float queuePriority = 1.0f;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR subgroupControlFlowFeatures = {};
        subgroupControlFlowFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR;
        subgroupControlFlowFeatures.shaderSubgroupUniformControlFlow = VK_TRUE;
        subgroupControlFlowFeatures.pNext = nullptr; // Chain starts here

        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &subgroupControlFlowFeatures;

        vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);
        if (subgroupControlFlowFeatures.shaderSubgroupUniformControlFlow != VK_TRUE) {
            throw std::runtime_error("Device does not support required subgroup uniform control flow feature!");
        }

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME
        };

        // deviceFeatures2.features.shaderSubgroupUniformControlFlow = VK_TRUE; // This should be set in the features struct itself if available and needed.
        // deviceFeatures2.features.timestampComputeAndGraphics = VK_TRUE; // This is the key change!
        
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pNext = &deviceFeatures2;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();


        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }
    
    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = 0; 
        VK_CHECK_RESULT(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
    }
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }
    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));
        VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));
    }
    
    void createBuffers() {
        VkDeviceSize uniformBufferSize = 3 * sizeof(uint32_t);
        VkDeviceSize inputPixelBufferSize = imageData.tiledData.size() * sizeof(float);
        
        createBuffer(uniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uniformBuffer, uniformBufferMemory);
        createBuffer(inputPixelBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, inputPixelBuffer, inputPixelBufferMemory);
        createBuffer(compressedBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, compressedBuffer, compressedBufferMemory);
        createBuffer(decodedBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, decodedBuffer, decodedBufferMemory);
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        std::cerr << "Uploading uniform data..." << std::endl;
        createBuffer(uniformBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, uniformBufferSize, 0, &data);
        memcpy(data, &imageData.totalBlocks, sizeof(uint32_t));
        (static_cast<uint32_t*>(data))[1] = 1; // Use PCA
        (static_cast<float*>(data))[2] = 0.33f; // Use 2-partition mode if the width of the point cloud is > 33% of the length
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, uniformBuffer, uniformBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        std::cerr << "Uploading pixel data..." << std::endl;
        createBuffer(inputPixelBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        vkMapMemory(device, stagingBufferMemory, 0, inputPixelBufferSize, 0, &data);
        memcpy(data, imageData.tiledData.data(), (size_t)inputPixelBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, inputPixelBuffer, inputPixelBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        createBuffer(compressedBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readbackCompressedBuffer, readbackCompressedMemory);
        createBuffer(decodedBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readbackDecodedBuffer, readbackDecodedMemory);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool; // This is now valid!
        allocInfo.commandBufferCount = 1;
        VkCommandBuffer copyCmdBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &copyCmdBuffer);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(copyCmdBuffer, &beginInfo);

        vkCmdResetQueryPool(copyCmdBuffer, queryPool, 0, 2);

        // Start timestamp
        vkCmdWriteTimestamp(copyCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
        
        VkBufferCopy copyRegion = {};
        copyRegion.size = size;
        vkCmdCopyBuffer(copyCmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkCmdWriteTimestamp(copyCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 1);
        
        vkEndCommandBuffer(copyCmdBuffer);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &copyCmdBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        uint64_t timestamps[2];
        VK_CHECK_RESULT(vkGetQueryPoolResults(
            device,
            queryPool,
            0, // first query
            2, // query count
            sizeof(timestamps),
            &timestamps,
            sizeof(uint64_t), // stride
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

        uint64_t startTime = timestamps[0];
        uint64_t endTime = timestamps[1];
        uint64_t duration_ticks = endTime - startTime;
        double duration_ns = duration_ticks * timestampPeriod;
        double duration_ms = duration_ns * 1e-6;

        std::cout << "----------------------------------------" << std::endl;
        std::cout << "GPU buffer copy time: " << duration_ms << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        vkFreeCommandBuffers(device, commandPool, 1, &copyCmdBuffer);
    }
    
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bindings[4] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = bindings;
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
    }
    
    void createDescriptorPool() {
        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 3;

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = 1;
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
    }

    void createDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        VkDescriptorBufferInfo uniformBufferInfo = {uniformBuffer, 0, 3 * sizeof(uint32_t)};
        VkDescriptorBufferInfo inputPixelBufferInfo = {inputPixelBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo compressedBufferInfo = {compressedBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo decodedBufferInfo = {decodedBuffer, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet descriptorWrites[4] = {};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &inputPixelBufferInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSet;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &compressedBufferInfo;
        
        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSet;
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &decodedBufferInfo;

        vkUpdateDescriptorSets(device, 4, descriptorWrites, 0, nullptr);
    }
    
    void createComputePipeline() {
        auto shaderCode = readShaderFile("astc_compress.spv");
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = shaderCode.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, nullptr, &computeShaderModule));

        VkPipelineShaderStageCreateInfo shaderStageInfo = {};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageInfo.module = computeShaderModule;
        shaderStageInfo.pName = "main";
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = shaderStageInfo;
        pipelineInfo.layout = pipelineLayout;
        VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    }

    void createAndRecordCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        vkCmdResetQueryPool(commandBuffer, queryPool, 0, 5);

        // Start timestamp
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);

        // End timestamp
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
        
        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, 
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
            VK_PIPELINE_STAGE_TRANSFER_BIT, 
            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 2);

        VkBufferCopy copyRegion = {};
        copyRegion.size = compressedBufferSize;
        vkCmdCopyBuffer(commandBuffer, compressedBuffer, readbackCompressedBuffer, 1, &copyRegion);

        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 3);
        
        copyRegion.size = decodedBufferSize;
        vkCmdCopyBuffer(commandBuffer, decodedBuffer, readbackDecodedBuffer, 1, &copyRegion);

        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 4);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
    }

    void runCommandBuffer() {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkFence fence;
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));
        std::cout << "Submitting..." << std::endl;
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        std::cout << "Completed..." << std::endl;

        // The GPU has finished, so we can get the query results.
        uint64_t timestamps[5];
        VK_CHECK_RESULT(vkGetQueryPoolResults(
            device,
            queryPool,
            0, // first query
            5, // query count
            sizeof(timestamps),
            &timestamps,
            sizeof(uint64_t), // stride
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

        uint64_t startTime = timestamps[0];
        uint64_t endTime = timestamps[1];
        uint64_t duration_ticks = endTime - startTime;
        double duration_ns = duration_ticks * timestampPeriod;
        double duration_ms = duration_ns * 1e-6;
        #define TIME(n) ((timestamps[n] - timestamps[n-1]) * timestampPeriod * 1e-6)

        std::cout << "----------------------------------------" << std::endl;
        std::cout << "GPU to dispatch time: " << TIME(1) << " ms" << std::endl;
        std::cout << "GPU to barrier time: " << TIME(2) << " ms" << std::endl;
        std::cout << "GPU to copy astc output buffer time: " << TIME(3) << " ms" << std::endl;
        std::cout << "GPU to copy decoded output buffer time: " << TIME(4) << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        std::cout << "GPU execution finished." << std::endl;

        vkDestroyFence(device, fence, nullptr);
    }
    
    void processResults() {
        void* mappedMemory;

        std::cout << "Reading back compressed data..." << std::endl;
        vkMapMemory(device, readbackCompressedMemory, 0, compressedBufferSize, 0, &mappedMemory);
        std::vector<uint32_t> compressedResult(compressedBufferSize / sizeof(uint32_t));
        memcpy(compressedResult.data(), mappedMemory, compressedBufferSize);
        vkUnmapMemory(device, readbackCompressedMemory);
        compressedResult.resize(imageData.totalBlocks * 4);
        save_astc_file("output.astc", compressedResult, imageData.originalWidth, imageData.originalHeight);
        
        std::cout << "Reading back decoded data..." << std::endl;
        vkMapMemory(device, readbackDecodedMemory, 0, decodedBufferSize, 0, &mappedMemory);
        std::vector<float> decodedResult(decodedBufferSize / sizeof(float));
        memcpy(decodedResult.data(), mappedMemory, decodedBufferSize);
        vkUnmapMemory(device, readbackDecodedMemory);
        save_decoded_image("decoded_screenshot.png", decodedResult, imageData.originalWidth, imageData.originalHeight);
    }

    void createQueryPool() {
        VkQueryPoolCreateInfo queryPoolInfo = {};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2; // One for start, one for end
        VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool));
    }

    void cleanup() {
        vkDestroyBuffer(device, uniformBuffer, nullptr);
        vkFreeMemory(device, uniformBufferMemory, nullptr);
        vkDestroyBuffer(device, inputPixelBuffer, nullptr);
        vkFreeMemory(device, inputPixelBufferMemory, nullptr);
        vkDestroyBuffer(device, compressedBuffer, nullptr);
        vkFreeMemory(device, compressedBufferMemory, nullptr);
        vkDestroyBuffer(device, decodedBuffer, nullptr);
        vkFreeMemory(device, decodedBufferMemory, nullptr);
        vkDestroyBuffer(device, readbackCompressedBuffer, nullptr);
        vkFreeMemory(device, readbackCompressedMemory, nullptr);
        vkDestroyBuffer(device, readbackDecodedBuffer, nullptr);
        vkFreeMemory(device, readbackDecodedMemory, nullptr);

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};

int main() {
    ASTCComputeApp app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}