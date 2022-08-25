#include <sys/syslog.h>
#include <syslog.h>

#include <set>
#include <map>
#include <utility>
#include <list>
#include <tuple>
#include <string>
#include <regex>
#include <chrono>
#include <optional>
#include <functional>

#include "nix_keyboard.hh"
#include "nix_common.hh"
#include "canonical_names.hh"
#include "keyboard_event.hh"
#include "keyboard.hh"

#include "../utils/string.hpp"
#include "../process/process.hpp"

using namespace TinyProcessLib;

template <typename T,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T &&iterable)
{
    struct iterator
    {
        size_t i;
        TIter iter;
        bool operator!=(const iterator &other) const { return iter != other.iter; }
        void operator++()
        {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper
    {
        T iterable;
        auto begin() { return iterator{0, std::begin(iterable)}; }
        auto end() { return iterator{0, std::end(iterable)}; }
    };
    return iterable_wrapper{std::forward<T>(iterable)};
}

/* Formats a dumpkeys format to our standard. */
std::pair<std::string, bool> cleanup_key(std::string name)
{
    ::ltrim(name, "+");
    bool is_keypad = name.starts_with("KP_");
    for (std::string mod : {"Meta_", "Control_", "dead_", "KP_"})
    {
        if (name.starts_with(mod))
            name = name.substr(mod.length());
    }

    // Dumpkeys is weird like that.
    if (name == "Remove")
        name = "Delete";
    else if (name == "Delete")
        name = "Backspace";

    if (name.ends_with("_r"))
        name = "right " + name.substr(0, name.length() - 2);
    if (name.ends_with("_l"))
        name = "left " + name.substr(0, name.length() - 2);

    return std::make_pair(normalize_name(name), is_keypad);
}

std::string cleanup_modifier(std::string modifier)
{
    modifier = normalize_name(modifier);
    if (std::find(begin(all_modifiers), end(all_modifiers), modifier) != end(all_modifiers))
        return modifier;
    if (std::find(begin(all_modifiers), end(all_modifiers), modifier.substr(0, modifier.length() - 1)) != end(all_modifiers))
        return modifier.substr(0, modifier.length() - 1);
    syslog(LOG_ERR, "Unknown modifier %s", modifier.c_str());
    exit(1);
}

/*
Use `dumpkeys --keys-only` to list all scan codes and their names. We
then parse the output and built a table. For each scan code and modifiers we
have a list of names and vice-versa.
*/

std::map<KeyAndModifiers, std::list<std::string>> to_name;
std::map<std::string, std::list<KeyAndModifiers>> from_name;
std::set<int> keypad_scan_codes;

void register_key(KeyAndModifiers key_and_modifiers, std::string name)
{
    std::list<std::string> names = to_name[key_and_modifiers];
    if (std::find(begin(names), end(names), name) == end(names))
        to_name[key_and_modifiers].push_back(name);
    std::list<KeyAndModifiers> keys_and_modifiers = from_name[name];
    if (std::find(begin(keys_and_modifiers), end(keys_and_modifiers), key_and_modifiers) == end(keys_and_modifiers))
        from_name[name].push_back(key_and_modifiers);
}

void build_tables()
{
    if (!to_name.empty() && !from_name.empty())
        return;

    std::map<std::string, int> modifiers_bits{
        {"shift", 1},
        {"alt gr", 2},
        {"ctrl", 4},
        {"alt", 8},
    };

    std::ostringstream oss;
    Process dumpkeys(std::vector<std::string>{"/usr/bin/dumpkeys", "--keys-only"}, "", [&](const char *bytes, size_t n)
                     { oss << std::string{bytes, n}; });
    int ret = dumpkeys.get_exit_status();
    if (ret)
    {
        syslog(LOG_ERR, "Failed to run dumpkeys to get key names. Check if your user is part of the \"tty\" group, and if not, add it with \"sudo usermod -a -G tty USER\".");
        exit(1);
    }
    std::string dump = oss.str();

    std::regex keycode_template("^keycode\\s+(\\d+)\\s+=(.*?)$");
    std::smatch keycode_match;
    std::string str_scan_code, str_names;
    int scan_code;
    std::vector<std::string> str_names_vec;
    std::vector<std::string> modifiers;
    std::string name;
    bool is_keypad;
    std::vector<std::string> dump_lines = split(dump, "\n");
    for (std::string dump_line : dump_lines)
    {
        while (std::regex_search(dump_line, keycode_match, keycode_template))
        {
            str_scan_code = keycode_match[1];
            str_names = keycode_match[2];
            scan_code = std::stoi(str_scan_code);
            trim(str_names);
            str_names_vec = split(str_names, " \t");
            for (auto [i, str_name] : enumerate(str_names_vec))
            {
                modifiers.clear();
                for (auto [modifier, bit] : modifiers_bits)
                {
                    if (i & bit)
                        modifiers.push_back(modifier);
                }
                std::sort(begin(modifiers), end(modifiers));
                std::tie(name, is_keypad) = cleanup_key(str_name);
                register_key(KeyAndModifiers(scan_code, modifiers), name);
                if (is_keypad)
                {
                    keypad_scan_codes.insert(scan_code);
                    register_key(KeyAndModifiers(scan_code, modifiers), "keypad " + name);
                }
            }

            dump_line = keycode_match.suffix();
        }
    }

    // dumpkeys consistently misreports the Windows key, sometimes
    // skipping it completely or reporting as 'alt. 125 = left win,
    // 126 = right win.
    KeyAndModifiers key125(125, std::vector<std::string>());
    if (!to_name.contains(key125) || (to_name[key125] == std::list<std::string>{"alt"}))
    {
        to_name[key125].clear();
        from_name["alt"].remove(key125);
        register_key(key125, "windows");
    }
    KeyAndModifiers key126(126, std::vector<std::string>());
    if (!to_name.contains(key126) || (to_name[key126] == std::list<std::string>{"alt"}))
    {
        to_name[key126].clear();
        from_name["alt"].remove(key126);
        register_key(key126, "windows");
    }

    // The menu key is usually skipped altogether, so we also add it manually.
    KeyAndModifiers key127(127, std::vector<std::string>());
    if (!to_name.contains(key127))
        register_key(key127, "menu");

    oss.clear();
    Process dumpkeys_long(std::vector<std::string>{"/usr/bin/dumpkeys", "--long-info"}, "", [&](const char *bytes, size_t n)
                          { oss << std::string{bytes, n}; });
    dump = oss.str();
    std::regex synonyms_template("^(\\S+)\\s+for (.+)$");
    std::smatch synonyms_match;
    std::string synonym_str, original_str;
    std::string synonym, original;
    dump_lines = split(dump, "\n");
    for (std::string dump_line : dump_lines)
    {
        while (std::regex_search(dump_line, synonyms_match, synonyms_template))
        {
            synonym_str = synonyms_match[1];
            original_str = synonyms_match[2];
            std::tie(synonym, std::ignore) = cleanup_key(synonym_str);
            std::tie(original, std::ignore) = cleanup_key(original_str);
            if (synonym != original)
            {
                auto &original_from = from_name[original];
                auto &synonym_from = from_name[synonym];
                original_from.insert(end(original_from), begin(synonym_from), end(synonym_from));
                synonym_from.insert(end(synonym_from), begin(original_from), end(original_from));
            }

            dump_line = synonyms_match.suffix();
        }
    }
}

std::shared_ptr<AggregatedEventDevice> device = nullptr;

void build_device()
{
    if (device)
        return;
    device = aggregate_devices("kbd");
}

void keyboard_init()
{
    build_device();
    build_tables();
}

std::vector<std::string> pressed_modifiers;

void keyboard_listen(Handler callback)
{
    build_device();
    build_tables();

    while (true)
    {
        auto [ie, device_id] = device->read_event();
        if (ie.type == EV_CNT)
            return;
        if (ie.type != EV_KEY)
            continue;

        auto since_epoch = std::chrono::seconds(ie.time.tv_sec) + std::chrono::microseconds(ie.time.tv_usec);
        time_point time = time_point() + since_epoch;

        int scan_code = ie.code;
        int event_type = ie.value ? KEY_DOWN : KEY_UP; // 0 = UP, 1 = DOWN, 2 = HOLD

        std::sort(begin(pressed_modifiers), end(pressed_modifiers));
        std::list<std::string> names = to_name[KeyAndModifiers{scan_code, pressed_modifiers}];
        if (names.empty())
            names = to_name[KeyAndModifiers(scan_code, std::vector<std::string>())];
        if (names.empty())
            names = std::list<std::string>{"unknown"};
        std::string name = names.front();

        if (std::find(begin(all_modifiers), end(all_modifiers), name) != end(all_modifiers))
        {
            if (event_type == KEY_DOWN)
                pressed_modifiers.push_back(name);
            else
                std::erase(pressed_modifiers, name);
        }

        bool is_keypad = keypad_scan_codes.contains(scan_code);
        callback(KeyboardEvent(event_type, scan_code, name, time, device_id, pressed_modifiers, is_keypad));
    }
}

void write_event(int scan_code, bool is_down)
{
    build_device();
    device->write_event(EV_KEY, scan_code, int(is_down));
}

std::vector<KeyAndModifiers> keyboard_map_name(std::string name)
{
    build_tables();
    std::vector<KeyAndModifiers> entries;
    for (auto entry : from_name[name])
    {
        entries.push_back(entry);
    }

    std::vector<std::string> parts = split(name, " ", 1);
    if ((parts.size() > 1) && ((parts[0] == "left") || (parts[0] == "right")))
    {
        for (auto entry : from_name[parts[1]])
        {
            entries.push_back(entry);
        }
    }
    return entries;
}

void keyboard_press(int scan_code)
{
    write_event(scan_code, true);
}

void keyboard_release(int scan_code)
{
    write_event(scan_code, false);
}
