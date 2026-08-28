#include "Criminal.h"
#include "Police.h"
#include <iostream>

void Criminal::releaseHostage(Police &police)
{
    police.waitUntilMoneySent();

    {
        std::lock_guard dataLock{m_mutex};
        hostageReleased = true;
        std::cout << "Criminal: hostage is released!\n";
    }
    m_cv.notify_one();
}

void Criminal::waitUntilHostageReleased()
{
    std::unique_lock dataLock{m_mutex};
    m_cv.wait(dataLock, [this] { return hostageReleased; });
}
