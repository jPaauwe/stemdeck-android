#include "gpu.hpp"

#include "spirv.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace sdgpu
{
namespace
{
// ─── Vulkan entry points, loaded at run time ───
#define SD_VK_INSTANCE_FNS(X)                                                  \
    X(vkDestroyInstance)                                                       \
    X(vkEnumeratePhysicalDevices)                                              \
    X(vkGetPhysicalDeviceProperties)                                           \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                \
    X(vkGetPhysicalDeviceMemoryProperties)                                     \
    X(vkCreateDevice)                                                          \
    X(vkGetDeviceProcAddr)

#define SD_VK_DEVICE_FNS(X)                                                    \
    X(vkDestroyDevice)                                                         \
    X(vkGetDeviceQueue)                                                        \
    X(vkCreateBuffer)                                                          \
    X(vkDestroyBuffer)                                                         \
    X(vkGetBufferMemoryRequirements)                                           \
    X(vkAllocateMemory)                                                        \
    X(vkFreeMemory)                                                            \
    X(vkBindBufferMemory)                                                      \
    X(vkMapMemory)                                                             \
    X(vkUnmapMemory)                                                           \
    X(vkFlushMappedMemoryRanges)                                               \
    X(vkInvalidateMappedMemoryRanges)                                          \
    X(vkCreateShaderModule)                                                    \
    X(vkDestroyShaderModule)                                                   \
    X(vkCreateDescriptorSetLayout)                                             \
    X(vkDestroyDescriptorSetLayout)                                            \
    X(vkCreatePipelineLayout)                                                  \
    X(vkDestroyPipelineLayout)                                                 \
    X(vkCreateComputePipelines)                                                \
    X(vkDestroyPipeline)                                                       \
    X(vkCreateDescriptorPool)                                                  \
    X(vkDestroyDescriptorPool)                                                 \
    X(vkResetDescriptorPool)                                                   \
    X(vkAllocateDescriptorSets)                                                \
    X(vkUpdateDescriptorSets)                                                  \
    X(vkCreateCommandPool)                                                     \
    X(vkDestroyCommandPool)                                                    \
    X(vkResetCommandPool)                                                      \
    X(vkAllocateCommandBuffers)                                                \
    X(vkBeginCommandBuffer)                                                    \
    X(vkEndCommandBuffer)                                                      \
    X(vkCmdBindPipeline)                                                       \
    X(vkCmdBindDescriptorSets)                                                 \
    X(vkCmdPushConstants)                                                      \
    X(vkCmdDispatch)                                                           \
    X(vkCmdPipelineBarrier)                                                    \
    X(vkCreateFence)                                                           \
    X(vkDestroyFence)                                                          \
    X(vkResetFences)                                                           \
    X(vkWaitForFences)                                                         \
    X(vkQueueSubmit)                                                           \
    X(vkDeviceWaitIdle)

struct GemmParams
{
    uint32_t M, N, K, lda, ldb, ldc, offA, offB, offC, transA, transB;
    float alpha, beta;
};

struct SoftmaxParams
{
    uint32_t rows, cols, ld, off;
};

struct Buffer
{
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    float *map = nullptr;
    size_t bytes = 0;
    bool coherent = true;
};

constexpr uint32_t kMaxSets = 512;
constexpr size_t kPoolKeepBytes = size_t(192) << 20; // free buffers kept for reuse (shared RAM on a phone)

class Device
{
  public:
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
#define SD_DECL(f) PFN_##f f = nullptr;
    SD_VK_INSTANCE_FNS(SD_DECL)
    SD_VK_DEVICE_FNS(SD_DECL)
#undef SD_DECL

    std::mutex lock;          // held by the running Batch
    std::mutex initLock;
    bool tried = false;
    bool ok = false;
    std::string why = "GPU: not started";

    void *lib = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkDeviceSize maxRange = 0;
    VkDeviceSize atom = 64;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout3 = VK_NULL_HANDLE, layout1 = VK_NULL_HANDLE;
    VkPipelineLayout gemmLayout = VK_NULL_HANDLE, softmaxLayout = VK_NULL_HANDLE;
    VkPipeline gemmPipe = VK_NULL_HANDLE, softmaxPipe = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    std::vector<Buffer> freeBufs; // guarded by lock
    std::map<const float *, Buffer> weights;
    std::map<const float *, size_t> weightSizes;

    ~Device() { shutdown(); }

    bool start();
    void shutdown();
    bool create_buffer(size_t bytes, Buffer &out);
    void destroy_buffer(Buffer &b);
    Buffer acquire(size_t bytes, bool &good);
    void release(Buffer &b);

  private:
    bool fail(const std::string &w)
    {
        why = "GPU not used: " + w;
        return false;
    }
    bool make_pipeline(const uint32_t *code, size_t bytes, VkPipelineLayout layout, VkPipeline &out);
};

Device &device()
{
    static Device d;
    return d;
}

std::atomic<bool> g_enabled{true};
std::atomic<bool> g_broken{false};

bool Device::start()
{
    lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib)
        lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib)
        return fail("no Vulkan on this device");
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
    if (!vkGetInstanceProcAddr)
        return fail("Vulkan loader incomplete");
    vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!vkCreateInstance)
        return fail("Vulkan loader incomplete");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "StemDeck";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
        return fail("Vulkan instance could not be created");
#define SD_LOAD_I(f) f = reinterpret_cast<PFN_##f>(vkGetInstanceProcAddr(instance, #f)); if (!f) return fail("missing " #f);
    SD_VK_INSTANCE_FNS(SD_LOAD_I)
#undef SD_LOAD_I

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    if (count)
        vkEnumeratePhysicalDevices(instance, &count, devs.data());
    // A real GPU first; a software one (llvmpipe) only when asked, for tests.
    const bool allowCpu = std::getenv("SD_GPU_ALLOW_CPU") != nullptr;
    int best = -1, bestScore = -1;
    for (uint32_t i = 0; i < count; ++i)
    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        int score = -1;
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 3;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 3;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) score = 2;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && allowCpu) score = 1;
        if (p.limits.maxComputeWorkGroupInvocations < 256 ||
            p.limits.maxComputeWorkGroupSize[0] < 256 ||
            p.limits.maxComputeSharedMemorySize < 16384)
            score = -1;
        if (score > bestScore)
        {
            bestScore = score;
            best = static_cast<int>(i);
        }
    }
    if (best < 0)
        return fail("no suitable GPU");
    phys = devs[best];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    maxRange = props.limits.maxStorageBufferRange;
    atom = std::max<VkDeviceSize>(1, props.limits.nonCoherentAtomSize);
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < qcount && !found; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            family = i;
            found = true;
        }
    if (!found)
        return fail("the GPU has no compute queue");

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS)
        return fail("the GPU could not be opened");
#define SD_LOAD_D(f) f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(dev, #f)); if (!f) return fail("missing " #f);
    SD_VK_DEVICE_FNS(SD_LOAD_D)
#undef SD_LOAD_D
    vkGetDeviceQueue(dev, family, 0, &queue);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    if (vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool) != VK_SUCCESS)
        return fail("command pool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev, &cai, &cmd) != VK_SUCCESS)
        return fail("command buffer");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSets * 3};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kMaxSets;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(dev, &dpi, nullptr, &descPool) != VK_SUCCESS)
        return fail("descriptor pool");

    auto makeLayout = [&](uint32_t n, VkDescriptorSetLayout &out) {
        std::vector<VkDescriptorSetLayoutBinding> b(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            b[i] = {};
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = n;
        li.pBindings = b.data();
        return vkCreateDescriptorSetLayout(dev, &li, nullptr, &out) == VK_SUCCESS;
    };
    if (!makeLayout(3, layout3) || !makeLayout(1, layout1))
        return fail("descriptor layout");

    auto makePipeLayout = [&](VkDescriptorSetLayout sl, uint32_t pcBytes, VkPipelineLayout &out) {
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &sl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        return vkCreatePipelineLayout(dev, &pli, nullptr, &out) == VK_SUCCESS;
    };
    if (!makePipeLayout(layout3, sizeof(GemmParams), gemmLayout) ||
        !makePipeLayout(layout1, sizeof(SoftmaxParams), softmaxLayout))
        return fail("pipeline layout");
    if (!make_pipeline(kSpirv_gemm, sizeof(kSpirv_gemm), gemmLayout, gemmPipe) ||
        !make_pipeline(kSpirv_softmax, sizeof(kSpirv_softmax), softmaxLayout, softmaxPipe))
        return fail("the compute shaders could not be built");

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS)
        return fail("fence");

    const uint32_t v = props.apiVersion;
    why = std::string("GPU: ") + props.deviceName + " (Vulkan " + std::to_string(VK_API_VERSION_MAJOR(v)) +
          "." + std::to_string(VK_API_VERSION_MINOR(v)) + ")";
    return true;
}

bool Device::make_pipeline(const uint32_t *code, size_t bytes, VkPipelineLayout layout, VkPipeline &out)
{
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = bytes;
    smi.pCode = code;
    VkShaderModule mod;
    if (vkCreateShaderModule(dev, &smi, nullptr, &mod) != VK_SUCCESS)
        return false;
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pci.stage.module = mod;
    pci.stage.pName = "main";
    pci.layout = layout;
    const bool good = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &out) == VK_SUCCESS;
    vkDestroyShaderModule(dev, mod, nullptr);
    return good;
}

void Device::shutdown()
{
    if (dev)
    {
        vkDeviceWaitIdle(dev);
        for (auto &b : freeBufs)
            destroy_buffer(b);
        freeBufs.clear();
        for (auto &w : weights)
            destroy_buffer(w.second);
        weights.clear();
        if (fence) vkDestroyFence(dev, fence, nullptr);
        if (gemmPipe) vkDestroyPipeline(dev, gemmPipe, nullptr);
        if (softmaxPipe) vkDestroyPipeline(dev, softmaxPipe, nullptr);
        if (gemmLayout) vkDestroyPipelineLayout(dev, gemmLayout, nullptr);
        if (softmaxLayout) vkDestroyPipelineLayout(dev, softmaxLayout, nullptr);
        if (layout3) vkDestroyDescriptorSetLayout(dev, layout3, nullptr);
        if (layout1) vkDestroyDescriptorSetLayout(dev, layout1, nullptr);
        if (descPool) vkDestroyDescriptorPool(dev, descPool, nullptr);
        if (cmdPool) vkDestroyCommandPool(dev, cmdPool, nullptr);
        vkDestroyDevice(dev, nullptr);
        dev = VK_NULL_HANDLE;
    }
    if (instance && vkDestroyInstance)
    {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    // The library stays loaded: unloading a Vulkan driver at exit is not safe.
}

bool Device::create_buffer(size_t bytes, Buffer &out)
{
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &out.buf) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, out.buf, &req);
    // CPU-visible memory (phones share RAM between CPU and GPU). Cached is
    // best, because results are read back by the CPU.
    const VkMemoryPropertyFlags prefs[] = {
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    int type = -1;
    for (VkMemoryPropertyFlags want : prefs)
    {
        for (uint32_t i = 0; i < memProps.memoryTypeCount && type < 0; ++i)
            if ((req.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & want) == want)
                type = static_cast<int>(i);
        if (type >= 0)
            break;
    }
    if (type < 0)
    {
        vkDestroyBuffer(dev, out.buf, nullptr);
        out.buf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = static_cast<uint32_t>(type);
    if (vkAllocateMemory(dev, &ai, nullptr, &out.mem) != VK_SUCCESS)
    {
        vkDestroyBuffer(dev, out.buf, nullptr);
        out.buf = VK_NULL_HANDLE;
        return false;
    }
    void *p = nullptr;
    if (vkBindBufferMemory(dev, out.buf, out.mem, 0) != VK_SUCCESS ||
        vkMapMemory(dev, out.mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS)
    {
        vkFreeMemory(dev, out.mem, nullptr);
        vkDestroyBuffer(dev, out.buf, nullptr);
        out = Buffer{};
        return false;
    }
    out.map = static_cast<float *>(p);
    out.bytes = bytes;
    out.coherent = (memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return true;
}

void Device::destroy_buffer(Buffer &b)
{
    if (b.mem)
    {
        vkUnmapMemory(dev, b.mem);
        vkFreeMemory(dev, b.mem, nullptr);
    }
    if (b.buf)
        vkDestroyBuffer(dev, b.buf, nullptr);
    b = Buffer{};
}

size_t round_size(size_t bytes)
{
    size_t r = 64 * 1024;
    while (r < bytes && r < (size_t(8) << 20))
        r <<= 1;
    if (r >= bytes)
        return r;
    const size_t mb = size_t(1) << 20;
    return (bytes + 4 * mb - 1) / (4 * mb) * (4 * mb);
}

Buffer Device::acquire(size_t bytes, bool &good)
{
    bytes = std::max<size_t>(bytes, 4);
    if (bytes > maxRange)
    {
        good = false;
        return Buffer{};
    }
    int best = -1;
    for (size_t i = 0; i < freeBufs.size(); ++i)
        if (freeBufs[i].bytes >= bytes && freeBufs[i].bytes <= 2 * bytes + (1 << 20) &&
            (best < 0 || freeBufs[i].bytes < freeBufs[best].bytes))
            best = static_cast<int>(i);
    if (best >= 0)
    {
        Buffer b = freeBufs[best];
        freeBufs.erase(freeBufs.begin() + best);
        return b;
    }
    Buffer b;
    if (!create_buffer(std::min<size_t>(round_size(bytes), maxRange), b) || b.bytes < bytes)
    {
        if (b.buf)
            destroy_buffer(b);
        good = false;
        return Buffer{};
    }
    return b;
}

void Device::release(Buffer &b)
{
    if (!b.buf)
        return;
    freeBufs.push_back(b);
    b = Buffer{};
    size_t total = 0;
    for (auto &f : freeBufs)
        total += f.bytes;
    while (total > kPoolKeepBytes && !freeBufs.empty())
    {
        total -= freeBufs.front().bytes;
        destroy_buffer(freeBufs.front());
        freeBufs.erase(freeBufs.begin());
    }
}

bool self_test();

bool ensure_started()
{
    Device &d = device();
    std::lock_guard<std::mutex> lk(d.initLock);
    if (!d.tried)
    {
        d.tried = true;
        d.ok = d.start();
        if (d.ok && !self_test())
        {
            d.ok = false;
            d.why = "GPU not used: it gave wrong results in the self-test";
        }
    }
    return d.ok;
}
} // namespace

// ─── Batch ───

struct Batch::Impl
{
    Device &d;
    std::unique_lock<std::mutex> lk;
    std::vector<Buffer> bufs;   // by Buf.id
    std::vector<bool> owned;    // returned to the pool at the end
    struct Out
    {
        int id;
        size_t off;
        float *dst;
        size_t n;
    };
    std::vector<Out> outs;
    uint32_t sets = 0;
    bool good = true;
    bool recorded = false;

    Impl(Device &dev, std::unique_lock<std::mutex> l) : d(dev), lk(std::move(l))
    {
        d.vkResetCommandPool(d.dev, d.cmdPool, 0);
        d.vkResetDescriptorPool(d.dev, d.descPool, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        good = d.vkBeginCommandBuffer(d.cmd, &bi) == VK_SUCCESS;
    }

    ~Impl()
    {
        for (size_t i = 0; i < bufs.size(); ++i)
            if (owned[i])
                d.release(bufs[i]);
    }

    Buf add(Buffer b, bool own)
    {
        bufs.push_back(b);
        owned.push_back(own);
        return Buf{static_cast<int>(bufs.size() - 1)};
    }

    void flush(const Buffer &b)
    {
        if (b.coherent)
            return;
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = b.mem;
        r.offset = 0;
        r.size = VK_WHOLE_SIZE;
        d.vkFlushMappedMemoryRanges(d.dev, 1, &r);
    }

    VkDescriptorSet set_for(std::initializer_list<int> ids, VkDescriptorSetLayout layout)
    {
        if (sets >= kMaxSets)
        {
            good = false;
            return VK_NULL_HANDLE;
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = d.descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet s;
        if (d.vkAllocateDescriptorSets(d.dev, &ai, &s) != VK_SUCCESS)
        {
            good = false;
            return VK_NULL_HANDLE;
        }
        ++sets;
        VkDescriptorBufferInfo infos[3];
        VkWriteDescriptorSet writes[3];
        uint32_t n = 0;
        for (int id : ids)
        {
            infos[n] = {bufs[id].buf, 0, VK_WHOLE_SIZE};
            writes[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[n].dstSet = s;
            writes[n].dstBinding = n;
            writes[n].descriptorCount = 1;
            writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[n].pBufferInfo = &infos[n];
            ++n;
        }
        d.vkUpdateDescriptorSets(d.dev, n, writes, 0, nullptr);
        return s;
    }

    void barrier()
    {
        if (!recorded)
            return;
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        d.vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    bool valid(Buf b, size_t off, size_t n) const
    {
        return b.id >= 0 && size_t(b.id) < bufs.size() && (off + n) * sizeof(float) <= bufs[b.id].bytes;
    }
};

Batch::Batch(Impl *impl) : impl_(impl) {}

Batch::~Batch() { delete impl_; }

std::unique_ptr<Batch> Batch::try_begin()
{
    if (!g_enabled.load() || g_broken.load() || !ensure_started())
        return nullptr;
    Device &d = device();
    std::unique_lock<std::mutex> lk(d.lock, std::try_to_lock);
    if (!lk.owns_lock())
        return nullptr;
    return std::unique_ptr<Batch>(new Batch(new Impl(d, std::move(lk))));
}

std::unique_ptr<Batch> Batch::begin()
{
    if (!g_enabled.load() || g_broken.load() || !ensure_started())
        return nullptr;
    Device &d = device();
    std::unique_lock<std::mutex> lk(d.lock);
    return std::unique_ptr<Batch>(new Batch(new Impl(d, std::move(lk))));
}

Buf Batch::input(const float *p, size_t n)
{
    Impl &im = *impl_;
    bool good = true;
    Buffer b = im.d.acquire(n * sizeof(float), good);
    if (!good)
    {
        im.good = false;
        return Buf{};
    }
    std::memcpy(b.map, p, n * sizeof(float));
    im.flush(b);
    return im.add(b, true);
}

Buf Batch::weights(const float *p, size_t n)
{
    Impl &im = *impl_;
    Device &d = im.d;
    auto it = d.weights.find(p);
    if (it != d.weights.end() && d.weightSizes[p] == n)
        return im.add(it->second, false);
    if (it != d.weights.end())
    {
        d.destroy_buffer(it->second);
        d.weights.erase(it);
    }
    Buffer b;
    if (n * sizeof(float) > d.maxRange || !d.create_buffer(n * sizeof(float), b))
    {
        im.good = false;
        return Buf{};
    }
    std::memcpy(b.map, p, n * sizeof(float));
    im.flush(b);
    d.weights[p] = b;
    d.weightSizes[p] = n;
    return im.add(b, false);
}

Buf Batch::scratch(size_t n)
{
    Impl &im = *impl_;
    bool good = true;
    Buffer b = im.d.acquire(n * sizeof(float), good);
    if (!good)
    {
        im.good = false;
        return Buf{};
    }
    return im.add(b, true);
}

void Batch::gemm(bool transA, bool transB, uint32_t M, uint32_t N, uint32_t K, float alpha, Buf A, size_t offA,
                 uint32_t lda, Buf B, size_t offB, uint32_t ldb, float beta, Buf C, size_t offC, uint32_t ldc)
{
    Impl &im = *impl_;
    if (!im.good)
        return;
    if (M == 0 || N == 0)
        return;
    const size_t needA = transA ? size_t(lda) * (M - 1) + K : size_t(lda) * (K ? K - 1 : 0) + M;
    const size_t needB = transB ? size_t(ldb) * (K ? K - 1 : 0) + N : size_t(ldb) * (N - 1) + K;
    const size_t needC = size_t(ldc) * (N - 1) + M;
    if (!im.valid(A, offA, needA) || !im.valid(B, offB, needB) || !im.valid(C, offC, needC) ||
        (M + 63) / 64 > 65535 || (N + 63) / 64 > 65535)
    {
        im.good = false;
        return;
    }
    Device &d = im.d;
    VkDescriptorSet s = im.set_for({A.id, B.id, C.id}, d.layout3);
    if (!im.good)
        return;
    im.barrier();
    GemmParams pc{M, N, K, lda, ldb, ldc, uint32_t(offA), uint32_t(offB), uint32_t(offC),
                  transA ? 1u : 0u, transB ? 1u : 0u, alpha, beta};
    d.vkCmdBindPipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.gemmPipe);
    d.vkCmdBindDescriptorSets(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.gemmLayout, 0, 1, &s, 0, nullptr);
    d.vkCmdPushConstants(d.cmd, d.gemmLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    d.vkCmdDispatch(d.cmd, (M + 63) / 64, (N + 63) / 64, 1);
    im.recorded = true;
}

void Batch::softmax_cols(Buf X, size_t off, uint32_t rows, uint32_t cols, uint32_t ld)
{
    Impl &im = *impl_;
    if (!im.good || rows == 0 || cols == 0)
        return;
    if (!im.valid(X, off, size_t(ld) * (cols - 1) + rows))
    {
        im.good = false;
        return;
    }
    Device &d = im.d;
    VkDescriptorSet s = im.set_for({X.id}, d.layout1);
    if (!im.good)
        return;
    im.barrier();
    SoftmaxParams pc{rows, cols, ld, uint32_t(off)};
    d.vkCmdBindPipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.softmaxPipe);
    d.vkCmdBindDescriptorSets(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.softmaxLayout, 0, 1, &s, 0, nullptr);
    d.vkCmdPushConstants(d.cmd, d.softmaxLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const uint32_t gx = std::min<uint32_t>(cols, 65535u);
    const uint32_t gy = (cols + 65534u) / 65535u;
    d.vkCmdDispatch(d.cmd, gx, gy, 1);
    im.recorded = true;
}

void Batch::output(Buf b, size_t off, float *dst, size_t n)
{
    Impl &im = *impl_;
    if (!im.valid(b, off, n))
    {
        im.good = false;
        return;
    }
    im.outs.push_back({b.id, off, dst, n});
}

bool Batch::run()
{
    Impl &im = *impl_;
    Device &d = im.d;
    if (!im.good)
    {
        d.vkEndCommandBuffer(d.cmd);
        return false;
    }
    // Make the shaders' writes visible to the CPU.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    d.vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
    if (d.vkEndCommandBuffer(d.cmd) != VK_SUCCESS)
        return false;
    d.vkResetFences(d.dev, 1, &d.fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cmd;
    if (d.vkQueueSubmit(d.queue, 1, &si, d.fence) != VK_SUCCESS)
    {
        g_broken.store(true);
        d.why = "GPU not used: it stopped accepting work";
        return false;
    }
    // A stuck GPU should not hang the app: give up after 20 s and use the CPU.
    const VkResult w = d.vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, 20ull * 1000 * 1000 * 1000);
    if (w != VK_SUCCESS)
    {
        g_broken.store(true);
        d.why = "GPU not used: it stopped responding";
        return false;
    }
    for (const auto &o : im.outs)
    {
        const Buffer &b = im.bufs[o.id];
        if (!b.coherent)
        {
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory = b.mem;
            r.offset = 0;
            r.size = VK_WHOLE_SIZE;
            d.vkInvalidateMappedMemoryRanges(d.dev, 1, &r);
        }
        std::memcpy(o.dst, b.map + o.off, o.n * sizeof(float));
    }
    return true;
}

// ─── self-test and settings ───

namespace
{
bool self_test()
{
    Device &d = device();
    std::unique_lock<std::mutex> lk(d.lock);
    // C (37 x 70) = A^T (37 x 53) * B (53 x 70), with offsets, plus beta * C.
    const uint32_t M = 37, N = 70, K = 53, lda = 60, ldb = 55, ldc = 40;
    std::vector<float> A(size_t(lda) * M + 3), B(size_t(ldb) * N + 5), C(size_t(ldc) * N + 7), out(C.size());
    for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin(0.37f * i);
    for (size_t i = 0; i < B.size(); ++i) B[i] = std::cos(0.11f * i);
    for (size_t i = 0; i < C.size(); ++i) C[i] = 0.5f * std::sin(0.05f * i);
    std::vector<float> S(7 * 5);
    for (size_t i = 0; i < S.size(); ++i) S[i] = 0.3f * i - 4.0f;
    std::vector<float> Sout(S.size());
    bool ran;
    {
        Batch b(new Batch::Impl(d, std::move(lk)));
        Buf a = b.input(A.data(), A.size()), bb = b.input(B.data(), B.size()), c = b.input(C.data(), C.size());
        b.gemm(true, false, M, N, K, 1.5f, a, 3, lda, bb, 5, ldb, 0.25f, c, 7, ldc);
        b.output(c, 0, out.data(), out.size());
        Buf s = b.input(S.data(), S.size());
        b.softmax_cols(s, 0, 7, 5, 7);
        b.output(s, 0, Sout.data(), Sout.size());
        ran = b.run();
    }
    if (!ran)
        return false;
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t m = 0; m < M; ++m)
        {
            double acc = 0;
            for (uint32_t k = 0; k < K; ++k)
                acc += double(A[3 + k + size_t(m) * lda]) * B[5 + k + size_t(n) * ldb];
            const double want = 1.5 * acc + 0.25 * C[7 + m + size_t(n) * ldc];
            if (std::fabs(want - out[7 + m + size_t(n) * ldc]) > 1e-3 * (1 + std::fabs(want)))
                return false;
        }
    for (int j = 0; j < 5; ++j)
    {
        double mx = -1e30, sum = 0;
        for (int i = 0; i < 7; ++i) mx = std::max<double>(mx, S[j * 7 + i]);
        for (int i = 0; i < 7; ++i) sum += std::exp(S[j * 7 + i] - mx);
        for (int i = 0; i < 7; ++i)
            if (std::fabs(std::exp(S[j * 7 + i] - mx) / sum - Sout[j * 7 + i]) > 1e-4)
                return false;
    }
    return true;
}
} // namespace

void set_enabled(bool on) { g_enabled.store(on); }

bool enabled() { return g_enabled.load(); }

bool available() { return g_enabled.load() && !g_broken.load() && ensure_started(); }

std::string info()
{
    if (!g_enabled.load())
        return "GPU switched off";
    ensure_started();
    return device().why;
}

void clear_weights()
{
    Device &d = device();
    {
        std::lock_guard<std::mutex> lk(d.initLock);
        if (!d.ok)
            return;
    }
    std::lock_guard<std::mutex> lk(d.lock);
    for (auto &w : d.weights)
        d.destroy_buffer(w.second);
    d.weights.clear();
    d.weightSizes.clear();
}
} // namespace sdgpu
