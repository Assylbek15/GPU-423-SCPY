#pragma once
#include <condition_variable>
#include <mutex>

class Criminal;

class Police
{
public:
    void giveRansom(Criminal &criminal);
    void waitUntilMoneySent();

private:
    std::condition_variable m_cv;
    std::mutex m_mutex;
    bool moneySent = false;
};
