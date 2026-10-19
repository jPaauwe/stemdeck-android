#include "safetensors.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sdengine
{
namespace
{
// ─── a small JSON reader, enough for a safetensors header ───
struct Json
{
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json *get(const std::string &k) const
    {
        for (const auto &kv : obj)
            if (kv.first == k)
                return &kv.second;
        return nullptr;
    }
};

class Parser
{
  public:
    explicit Parser(const std::string &s) : s_(s) {}

    bool parse(Json &out)
    {
        if (!value(out))
            return false;
        ws();
        return i_ == s_.size();
    }

  private:
    void ws()
    {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t'))
            ++i_;
    }

    bool string(std::string &out)
    {
        if (i_ >= s_.size() || s_[i_] != '"')
            return false;
        ++i_;
        while (i_ < s_.size() && s_[i_] != '"')
        {
            char c = s_[i_++];
            if (c == '\\' && i_ < s_.size())
            {
                const char e = s_[i_++];
                switch (e)
                {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'u':
                    // Not needed for tensor names; keep the escape as is.
                    out += "\\u";
                    continue;
                default: c = e;
                }
            }
            out += c;
        }
        if (i_ >= s_.size())
            return false;
        ++i_;
        return true;
    }

    bool value(Json &v)
    {
        ws();
        if (i_ >= s_.size())
            return false;
        const char c = s_[i_];
        if (c == '{')
        {
            v.kind = Json::Object;
            ++i_;
            ws();
            if (i_ < s_.size() && s_[i_] == '}')
            {
                ++i_;
                return true;
            }
            for (;;)
            {
                ws();
                std::string k;
                if (!string(k))
                    return false;
                ws();
                if (i_ >= s_.size() || s_[i_] != ':')
                    return false;
                ++i_;
                Json item;
                if (!value(item))
                    return false;
                v.obj.emplace_back(std::move(k), std::move(item));
                ws();
                if (i_ < s_.size() && s_[i_] == ',')
                {
                    ++i_;
                    continue;
                }
                if (i_ < s_.size() && s_[i_] == '}')
                {
                    ++i_;
                    return true;
                }
                return false;
            }
        }
        if (c == '[')
        {
            v.kind = Json::Array;
            ++i_;
            ws();
            if (i_ < s_.size() && s_[i_] == ']')
            {
                ++i_;
                return true;
            }
            for (;;)
            {
                Json item;
                if (!value(item))
                    return false;
                v.arr.push_back(std::move(item));
                ws();
                if (i_ < s_.size() && s_[i_] == ',')
                {
                    ++i_;
                    continue;
                }
                if (i_ < s_.size() && s_[i_] == ']')
                {
                    ++i_;
                    return true;
                }
                return false;
            }
        }
        if (c == '"')
        {
            v.kind = Json::String;
            return string(v.str);
        }
        if (s_.compare(i_, 4, "true") == 0) { v.kind = Json::Bool; v.num = 1; i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { v.kind = Json::Bool; i_ += 5; return true; }
        if (s_.compare(i_, 4, "null") == 0) { v.kind = Json::Null; i_ += 4; return true; }
        size_t j = i_;
        while (j < s_.size() && (std::strchr("+-.eE0123456789", s_[j]) != nullptr))
            ++j;
        if (j == i_)
            return false;
        v.kind = Json::Number;
        v.num = std::strtod(s_.substr(i_, j - i_).c_str(), nullptr);
        i_ = j;
        return true;
    }

    const std::string &s_;
    size_t i_ = 0;
};

struct FileCloser
{
    void operator()(std::FILE *f) const
    {
        if (f)
            std::fclose(f);
    }
};
} // namespace

std::string safetensors_to_ggml(const std::string &in, const std::string &out)
{
    std::unique_ptr<std::FILE, FileCloser> f(std::fopen(in.c_str(), "rb"));
    if (!f)
        return "The model file could not be opened.";
    unsigned char lenBytes[8];
    if (std::fread(lenBytes, 1, 8, f.get()) != 8)
        return "The model file is damaged.";
    uint64_t headerLen = 0;
    for (int i = 7; i >= 0; --i)
        headerLen = (headerLen << 8) | lenBytes[i];
    if (headerLen == 0 || headerLen > (64u << 20))
        return "The model file is damaged.";
    std::string header(headerLen, '\0');
    if (std::fread(&header[0], 1, headerLen, f.get()) != headerLen)
        return "The model file is damaged.";
    Json root;
    if (!Parser(header).parse(root) || root.kind != Json::Object)
        return "The model file's header could not be read.";
    const long dataStart = static_cast<long>(8 + headerLen);

    // Six sources (drums, bass, other, vocals, guitar, piano) or four: from
    // the model's init arguments in the metadata.
    bool six = false;
    if (const Json *meta = root.get("__metadata__"))
        for (const auto &kv : meta->obj)
            if ((kv.first == "args" || kv.first == "kwargs") && kv.second.str.find("\"piano\"") != std::string::npos)
                six = true;

    std::unique_ptr<std::FILE, FileCloser> o(std::fopen(out.c_str(), "wb"));
    if (!o)
        return "The converted model could not be written.";
    const uint32_t magic = six ? 0x646d6336u : 0x646d6334u;
    std::fwrite(&magic, 4, 1, o.get());

    std::vector<unsigned char> raw;
    std::vector<Eigen::half> half;
    int tensors = 0;
    for (const auto &kv : root.obj)
    {
        if (kv.first == "__metadata__")
            continue;
        const Json &t = kv.second;
        const Json *dtype = t.get("dtype");
        const Json *shape = t.get("shape");
        const Json *offs = t.get("data_offsets");
        if (!dtype || !shape || !offs || offs->arr.size() != 2)
            return "The model file's header is not as expected.";
        const uint64_t b = static_cast<uint64_t>(offs->arr[0].num), e = static_cast<uint64_t>(offs->arr[1].num);
        uint64_t count = 1;
        std::vector<int32_t> dims; // size-1 dims dropped, as numpy's squeeze
        for (const Json &d : shape->arr)
        {
            const uint64_t n = static_cast<uint64_t>(d.num);
            count *= n;
            if (n != 1)
                dims.push_back(static_cast<int32_t>(n));
        }
        const std::string &dt = dtype->str;
        const size_t esize = (dt == "F32") ? 4 : (dt == "F16" || dt == "BF16") ? 2 : 0;
        if (esize == 0)
            return "The model file has an unsupported number format (" + dt + ").";
        if (e < b || e - b != count * esize)
            return "The model file's header is not as expected.";
        raw.resize(e - b);
        if (std::fseek(f.get(), dataStart + static_cast<long>(b), SEEK_SET) != 0 ||
            (count && std::fread(raw.data(), 1, raw.size(), f.get()) != raw.size()))
            return "The model file is incomplete.";
        half.resize(count);
        if (dt == "F16")
            std::memcpy(half.data(), raw.data(), raw.size());
        else if (dt == "F32")
            for (uint64_t i = 0; i < count; ++i)
            {
                float v;
                std::memcpy(&v, raw.data() + 4 * i, 4);
                half[i] = Eigen::half(v);
            }
        else // BF16: the top half of a float32
            for (uint64_t i = 0; i < count; ++i)
            {
                uint32_t bits = uint32_t(raw[2 * i]) << 16 | uint32_t(raw[2 * i + 1]) << 24;
                float v;
                std::memcpy(&v, &bits, 4);
                half[i] = Eigen::half(v);
            }
        const int32_t nd = static_cast<int32_t>(dims.size());
        const int32_t nameLen = static_cast<int32_t>(kv.first.size());
        std::fwrite(&nd, 4, 1, o.get());
        std::fwrite(&nameLen, 4, 1, o.get());
        for (int32_t d : dims)
            std::fwrite(&d, 4, 1, o.get());
        std::fwrite(kv.first.data(), 1, kv.first.size(), o.get());
        if (count && std::fwrite(half.data(), 2, count, o.get()) != count)
            return "The converted model could not be written. Is the storage full?";
        ++tensors;
    }
    if (tensors == 0)
        return "The model file holds no weights.";
    if (std::fflush(o.get()) != 0)
        return "The converted model could not be written. Is the storage full?";
    return "";
}
} // namespace sdengine
