#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class BlockingQueue
{
public:
    BlockingQueue() : _shutdown(false) {}

    ~BlockingQueue() {
        // shutdown();
    }

    void push(T const &_data)
    {
        {
            std::lock_guard<std::mutex> lock(guard);
            queue.push(_data);
        }
        signal.notify_one();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(guard);
        return queue.empty();
    }

    bool tryPop(T &_value)
    {
        std::lock_guard<std::mutex> lock(guard);
        if (queue.empty())
        {
            return false;
        }

        _value = queue.front();
        queue.pop();
        return true;
    }

    bool waitAndPop(T &_value)
    {
        std::unique_lock<std::mutex> lock(guard);
        signal.wait(lock, [this]() { return _shutdown || !queue.empty(); });

        if (!_shutdown)
        {
            _value = queue.front();
            queue.pop();
            return true;
        }

        return false;
    }

    bool tryWaitAndPop(T &_value, int _milli)
    {
        std::unique_lock<std::mutex> lock(guard);
        while (queue.empty())
        {
            signal.wait_for(lock, std::chrono::milliseconds(_milli));
            return false;
        }

        _value = queue.front();
        queue.pop();
        return true;
    }

    void shutdown() {
        _shutdown = true;
        signal.notify_all();
    }

private:
    std::atomic<bool> _shutdown;
    std::queue<T> queue;
    mutable std::mutex guard;
    std::condition_variable signal;
};
