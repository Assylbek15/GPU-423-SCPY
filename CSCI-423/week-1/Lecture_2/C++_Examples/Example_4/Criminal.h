#pragma once
#include <condition_variable>
#include <mutex>

class Police;

class Criminal
{
public:
    void releaseHostage(Police &police);
    void waitUntilHostageReleased();

private:
    std::condition_variable m_cv;
    std::mutex m_mutex;
    bool hostageReleased = false;
};
