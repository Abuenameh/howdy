#include <sys/syslog.h>
#include <syslog.h>
#include <stdio.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <filesystem>
#include <functional>
#include <ranges>

#include "../utils/glob.hpp"

#include "keyboard.hh"
#include "nix_common.hh"

namespace fs = std::filesystem;

using namespace std::literals;

std::FILE *make_uinput()
{
    if (!fs::exists(fs::status("/dev/uinput")))
    {
        syslog(LOG_ERR, "No uinput module found.");
        exit(1);
    }

    // Requires uinput driver, but it's usually available.
    std::FILE *uinput = std::fopen("/dev/uinput", "wb");
    if (!uinput)
    {
        syslog(LOG_ERR, "Could open /dev/uinput for writing.");
        exit(1);
    }
    int uinputfd = fileno(uinput);
    ioctl(uinputfd, UI_SET_EVBIT, EV_KEY);

    for (int i : std::views::iota(0, 256))
        ioctl(uinputfd, UI_SET_KEYBIT, i);

    uinput_user_dev user_dev{};
    std::strncpy(user_dev.name, "Virtual Keyboard", UINPUT_MAX_NAME_SIZE);
    user_dev.id.bustype = BUS_USB;
    user_dev.id.vendor = 1;
    user_dev.id.product = 1;
    user_dev.id.version = 1;
    user_dev.ff_effects_max = 0;
    std::fwrite(&user_dev, sizeof(user_dev), 1, uinput);

    std::fflush(uinput); // Without this you may get Errno 22: Invalid argument.

    ioctl(uinputfd, UI_DEV_CREATE);

    return uinput;
}

EventDevice::EventDevice(std::string path) : path(path), _input_file(std::nullopt), _output_file(std::nullopt) {}

std::FILE *EventDevice::input_file()
{
    if (!_input_file)
    {
        _input_file = std::fopen(path.c_str(), "rb");
        if (!*_input_file)
        {
            syslog(LOG_ERR, "Keyboard device error: %s", std::strerror(errno));
            if (errno == EACCES)
            {
                syslog(LOG_ERR, "Failed to read device '%s'. You must be in the 'input' group to access global events. Use 'sudo usermod -a -G input USERNAME' to add user to the required group.", path.c_str());
                exit(1);
            }
        }
    }
    return *_input_file;
}

std::FILE *EventDevice::output_file()
{
    if (!_output_file)
        _output_file = std::fopen(path.c_str(), "wb");
    return *_output_file;
}

InputEvent EventDevice::read_event()
{
    input_event ie{};
    std::fread(&ie, sizeof(ie), 1, input_file());
    InputEvent event{ie, path};
    return event;
}

void EventDevice::write_event(unsigned short type, unsigned short code, int value)
{
    int time = std::chrono::time_point_cast<std::chrono::microseconds>(now()).time_since_epoch().count();
    int seconds = int(time / 1e6);
    int microseconds = time % 1000000;

    input_event events[2];
    events[0] = input_event{seconds, microseconds, type, code, value};
    // Send a sync event to ensure other programs update.
    events[1] = input_event{seconds, microseconds, EV_SYN, 0, 0};
    std::fwrite(events, sizeof(input_event), 2, output_file());

    std::fflush(output_file());
}

AggregatedEventDevice::AggregatedEventDevice(std::list<std::shared_ptr<EventDevice>> devices, std::shared_ptr<EventDevice> output) : devices(devices), output(output)
{
    if (!output)
        output = devices.front();
    for (std::shared_ptr<EventDevice> device : devices)
    {
        std::thread thread([=, this]()
                           {
                while (true)
                    event_queue.push(device->read_event()); });
        thread.detach();
    }
}

InputEvent AggregatedEventDevice::read_event()
{
    InputEvent event{};
    if (!event_queue.waitAndPop(event))
        event.ie.type = EV_CNT;
    return event;
}

void AggregatedEventDevice::write_event(unsigned short type, unsigned short code, int value)
{
    output->write_event(type, code, value);
}

std::list<std::shared_ptr<EventDevice>> list_devices_from_proc(std::string type_name)
{
    std::list<std::shared_ptr<EventDevice>> devices;
    std::string description;
    try
    {
        std::ifstream f("/proc/bus/input/devices");
        std::stringstream buffer;
        buffer << f.rdbuf();
        description = buffer.str();
    }
    catch (std::exception &e)
    {
        return devices;
    }

    std::regex device_pattern("N: Name=\"([^\"]+?)\"[\\s\\S]+?H: Handlers=([^\\n]+)");
    std::smatch device_match;
    std::regex handler_pattern("event(\\d+)");
    std::smatch handler_match;
    std::string handlers, path;
    while (std::regex_search(description, device_match, device_pattern))
    {
        handlers = device_match[2];
        std::regex_search(handlers, handler_match, handler_pattern);
        path = "/dev/input/event" + handler_match[1].str();
        if (handlers.find(type_name) != std::string::npos)
            devices.push_back(std::make_shared<EventDevice>(path));

        description = device_match.suffix();
    }
    return devices;
}

std::list<std::shared_ptr<EventDevice>> list_devices_from_by_id(std::string name_suffix, bool by_id = true)
{
    std::list<std::shared_ptr<EventDevice>> devices;
    for (auto path : glob::glob("/dev/input/"s + (by_id ? "by-id"s : "by-path"s) + "/*-event-"s + name_suffix))
        devices.push_back(std::make_shared<EventDevice>(path));
    return devices;
}

std::shared_ptr<AggregatedEventDevice> aggregate_devices(std::string type_name)
{
    std::shared_ptr<EventDevice> fake_device;
    // Some systems have multiple keyboards with different range of allowed keys
    // on each one, like a notebook with a "keyboard" device exclusive for the
    // power button. Instead of figuring out which keyboard allows which key to
    // send events, we create a fake device and send all events through there.
    try
    {
        std::FILE *uinput = make_uinput();
        fake_device = std::make_shared<EventDevice>("uinput Fake Device");
        fake_device->_input_file = uinput;
        fake_device->_output_file = uinput;
    }
    catch (std::exception &e)
    {
        syslog(LOG_WARNING, "Failed to create a device file using `uinput` module. Sending of events may be limited or unavailable depending on plugged-in devices.");
        syslog(LOG_WARNING, "%s", e.what());
        fake_device = nullptr;
    }

    // We don't aggregate devices from different sources to avoid
    // duplicates.

    std::list<std::shared_ptr<EventDevice>> devices_from_proc = list_devices_from_proc(type_name);
    if (!devices_from_proc.empty())
        return std::make_shared<AggregatedEventDevice>(devices_from_proc, fake_device);

    // breaks on mouse for virtualbox
    // was getting /dev/input/by-id/usb-VirtualBox_USB_Tablet-event-mouse
    std::list<std::shared_ptr<EventDevice>> devices_from_by_id = list_devices_from_by_id(type_name);
    if (devices_from_by_id.empty())
        devices_from_by_id = list_devices_from_by_id(type_name, false);
    if (!devices_from_by_id.empty())
        return std::make_shared<AggregatedEventDevice>(devices_from_by_id, fake_device);

    // If no keyboards were found we can only use the fake device to send keys.
    if (!fake_device)
    {
        syslog(LOG_ERR, "No keyboards found and was unable to create a fake device");
        exit(1);
    }
    return std::make_shared<AggregatedEventDevice>(std::list<std::shared_ptr<EventDevice>>{fake_device}, fake_device);
}