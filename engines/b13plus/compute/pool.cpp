#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace sdpool
{
namespace
{
struct Job
{
    const std::function<void(int)> *fn;
    int n;
    std::atomic<int> next{0};
    std::atomic<int> done{0};
    int helping = 0; // helper threads inside work(), guarded by the pool lock
    std::mutex m;
    std::condition_variable cv;

    // Takes and runs tasks until none are left. Returns true if it ran any.
    bool work()
    {
        bool any = false;
        for (;;)
        {
            const int i = next.fetch_add(1);
            if (i >= n)
                return any;
            (*fn)(i);
            any = true;
            if (done.fetch_add(1) + 1 == n)
            {
                std::lock_guard<std::mutex> lk(m);
                cv.notify_all();
            }
        }
    }
};

thread_local bool t_inTask = false;

class Pool
{
  public:
    ~Pool() { resize(0); }

    void resize(int helpers)
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            cv_.notify_all();
        }
        for (auto &t : threads_)
            t.join();
        threads_.clear();
        stop_ = false;
        for (int i = 0; i < helpers; ++i)
            threads_.emplace_back([this] { loop(); });
    }

    int helpers() const { return static_cast<int>(threads_.size()); }

    void run(Job &job)
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            jobs_.push_back(&job);
            cv_.notify_all();
        }
        t_inTask = true;
        job.work();
        t_inTask = false;
        {
            std::unique_lock<std::mutex> lk(job.m);
            job.cv.wait(lk, [&] { return job.done.load() == job.n; });
        }
        // No new helper can pick the job once it is off the list; wait for
        // the ones still leaving work() before the job goes out of scope.
        std::unique_lock<std::mutex> lk(m_);
        jobs_.erase(std::find(jobs_.begin(), jobs_.end(), &job));
        cv_.wait(lk, [&] { return job.helping == 0; });
    }

  private:
    void loop()
    {
        t_inTask = true;
        std::unique_lock<std::mutex> lk(m_);
        for (;;)
        {
            cv_.wait(lk, [&] { return stop_ || has_work(); });
            if (stop_)
                return;
            // Help the oldest job that still has tasks.
            Job *job = nullptr;
            for (Job *j : jobs_)
                if (j->next.load() < j->n)
                {
                    job = j;
                    break;
                }
            if (!job)
                continue;
            ++job->helping;
            lk.unlock();
            job->work();
            lk.lock();
            if (--job->helping == 0)
                cv_.notify_all();
        }
    }

    bool has_work() const
    {
        for (const Job *j : jobs_)
            if (j->next.load() < j->n)
                return true;
        return false;
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Job *> jobs_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

std::mutex g_config;
int g_threads = 0; // 0 = not set yet

Pool &pool()
{
    static Pool p;
    return p;
}

int hardware()
{
    const unsigned n = std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 4;
}
} // namespace

void set_threads(int n)
{
    std::lock_guard<std::mutex> lk(g_config);
    const int want = n > 0 ? n : hardware();
    if (want == g_threads)
        return;
    g_threads = want;
    // The callers do maths too, so the pool needs one thread fewer.
    pool().resize(std::max(0, want - 1));
}

int threads()
{
    std::lock_guard<std::mutex> lk(g_config);
    return g_threads > 0 ? g_threads : 1;
}

void parallel_for(int n, const std::function<void(int)> &fn)
{
    if (n <= 0)
        return;
    {
        std::lock_guard<std::mutex> lk(g_config);
        if (g_threads == 0)
        {
            g_threads = hardware();
            pool().resize(std::max(0, g_threads - 1));
        }
    }
    if (n == 1 || t_inTask || pool().helpers() == 0)
    {
        for (int i = 0; i < n; ++i)
            fn(i);
        return;
    }
    Job job;
    job.fn = &fn;
    job.n = n;
    pool().run(job);
}
} // namespace sdpool
