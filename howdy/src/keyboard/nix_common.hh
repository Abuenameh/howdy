#ifndef NIX_COMMON_H_
#define NIX_COMMON_H_

#include <linux/uinput.h>

#include <cstdio>
#include <string>
#include <optional>
#include <list>

#include "../utils/blocking_queue.hpp"

struct InputEvent
{
    input_event ie;
    std::string device;
};

class EventDevice
{

public:
    EventDevice(std::string path);

    std::FILE *input_file();

    std::FILE *output_file();

    InputEvent read_event();

    void write_event(unsigned short type, unsigned short code, int value);

    std::string path;
    std::optional<std::FILE *> _input_file;
    std::optional<std::FILE *> _output_file;
};

class AggregatedEventDevice
{
public:
    AggregatedEventDevice(std::list<std::shared_ptr<EventDevice>> devices, std::shared_ptr<EventDevice> output = nullptr);

    InputEvent read_event();

    void write_event(unsigned short type, unsigned short code, int value);

    std::list<std::shared_ptr<EventDevice>> devices;
    std::shared_ptr<EventDevice> output;
    BlockingQueue<InputEvent> event_queue;
};

std::shared_ptr<AggregatedEventDevice> aggregate_devices(std::string type_name);

#endif // NIX_COMMON_H_