#ifndef GENERIC_H_
#define GENERIC_H_

#include <list>
#include <queue>
#include <memory>
#include <functional>

#include "keyboard_event.hh"
#include "../utils/blocking_queue.hpp"

typedef std::function<bool(KeyboardEvent)> Handler;
typedef std::list<Handler>::iterator HandlerIter;

class GenericListener
{

public:
    GenericListener();

    virtual ~GenericListener() = default;

    virtual void init();

    void invoke_handlers(KeyboardEvent event);

    /*
    Starts the listening thread if it wasn't already.
    */
    void start_if_necessary();

    virtual bool pre_process_event(KeyboardEvent event);

    virtual void listen();

    /*
    Loops over the underlying queue of events and processes them in order.
    */
    void process();

    /*
    Adds a function to receive each event captured, starting the capturing
    process if necessary.
    */
    HandlerIter add_handler(Handler handler);

    /* Removes a previously added event handler. */
    void remove_handler(HandlerIter handler);

    std::list<Handler> handlers;
    bool listening;
    BlockingQueue<KeyboardEvent> queue;
    std::thread listening_thread;
    std::thread processing_thread;
};

#endif // GENERIC_H_