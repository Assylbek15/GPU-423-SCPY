#pragma once
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

class Starvation
{
    public:
        static constexpr int numberOfThreads = 4;

        void incrementValue(int threadId)
        {
            int value = 0;
            while (value < 4)
            {
                {
                    std::unique_lock dataLock{m_mutex};
                    m_cv.wait(dataLock, [this, threadId] { return currentTurn == threadId; });

                    ++value;
                    std::cout<<"Thread "<<threadId<<" with id "<<std::this_thread::get_id()<<" changed the value to "<<value<<"\n";
                    currentTurn = (currentTurn + 1) % numberOfThreads;
                }
                m_cv.notify_all();
            }
        }

    private:
        std::condition_variable m_cv;
        std::mutex m_mutex;
        int currentTurn = 0;
};
