#include <sys/syslog.h>
#include <syslog.h>

#include <queue>
#include <thread>
#include <mutex>
#include <exception>

#include "keyboard.hh"
#include "generic.hh"

GenericListener::GenericListener()
{
    listening = false;
}

void GenericListener::init() {}

void GenericListener::invoke_handlers(KeyboardEvent event)
{
    for (Handler handler : handlers)
    {
        try
        {
            if (handler(event))
                // Stop processing this hotkey.
                return;
        }
        catch (std::exception &e)
        {
            syslog(LOG_ERR, "Exception in generic listener: %s", e.what());
        }
    }
}

/*
Starts the listening thread if it wasn't already.
*/
void GenericListener::start_if_necessary()
{
    std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (!listening)
    {
        init();

        listening = true;
        listening_thread = std::thread([this]()
                                       { listen(); });
        processing_thread = std::thread([this]()
                                        { process(); });
    }
}

bool GenericListener::pre_process_event(KeyboardEvent event)
{
    throw std::logic_error("This method should be implemented in the child class.");
}

void GenericListener::listen() {}

/*
Loops over the underlying queue of events and processes them in order.
*/
void GenericListener::process()
{
    while (true)
    {
        KeyboardEvent event;
        if (!queue.waitAndPop(event))
            return;
        if (pre_process_event(event))
            invoke_handlers(event);
    }
}

/*
Adds a function to receive each event captured, starting the capturing
process if necessary.
*/
HandlerIter GenericListener::add_handler(Handler handler)
{
    start_if_necessary();
    return handlers.insert(end(handlers), handler);
}

/* Removes a previously added event handler. */
void GenericListener::remove_handler(HandlerIter handler)
{
    handlers.erase(handler);
}
