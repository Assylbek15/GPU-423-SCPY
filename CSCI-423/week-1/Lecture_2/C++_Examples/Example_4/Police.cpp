#include "Police.h"
#include "Criminal.h"
#include <iostream>

void Police::giveRansom(Criminal &criminal)
{
    {
        std::lock_guard dataLock{m_mutex};
        moneySent = true;
        std::cout << "Police: sent money!\n";
    }
    m_cv.notify_one();

    criminal.waitUntilHostageReleased();
}

void Police::waitUntilMoneySent()
{
    std::unique_lock dataLock{m_mutex};
    m_cv.wait(dataLock, [this] { return moneySent; });
}
