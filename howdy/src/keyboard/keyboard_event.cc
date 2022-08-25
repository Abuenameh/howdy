#include "canonical_names.hh"
#include "keyboard_event.hh"

KeyboardEvent::KeyboardEvent()
{
    KeyboardEvent(0, std::nullopt);
}

KeyboardEvent::KeyboardEvent(int event_type, std::optional<int> scan_code, std::optional<std::string> name, std::optional<time_point> time, std::optional<std::string> device, std::optional<std::vector<std::string>> modifiers, std::optional<bool> is_keypad) : event_type(event_type), scan_code(scan_code), name(name), time(time), device(device), modifiers(modifiers), is_keypad(is_keypad)
{
    if (!time)
        time = now();
    if (name)
        name = normalize_name(name);
}

std::string KeyboardEvent::to_string()
{
    return "KeyboardEvent(" + (name ? *name : "Unknown " + std::to_string(*scan_code) + " " + std::to_string(event_type) + ")");
}

bool KeyboardEvent::operator==(const KeyboardEvent &other) const
{
    return (event_type == other.event_type) && (!scan_code || !other.scan_code || scan_code == other.scan_code) && (!name || !other.name || name == other.name);
}
