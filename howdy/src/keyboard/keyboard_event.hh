#ifndef KEYBOARD_EVENT_H_
#define KEYBOARD_EVENT_H_

#include <linux/uinput.h>

#include <list>
#include <vector>
#include <string>
#include <memory>
#include <optional>

#include "../utils.hh"

class KeyboardEvent
{
public:
    KeyboardEvent();
    KeyboardEvent(int event_type, std::optional<int> scan_code, std::optional<std::string> name = std::nullopt, std::optional<time_point> time = std::nullopt, std::optional<std::string> device = std::nullopt, std::optional<std::vector<std::string>> modifiers = std::nullopt, std::optional<bool> is_keypad = std::nullopt);

    std::string to_string();

    bool operator==(const KeyboardEvent &other) const;

    int event_type;
    std::optional<int> scan_code;
    std::optional<std::string> name;
    std::optional<time_point> time;
    std::optional<std::string> device;
    std::optional<std::vector<std::string>> modifiers;
    std::optional<bool> is_keypad;
};

#endif // KEYBOARD_EVENT_H_