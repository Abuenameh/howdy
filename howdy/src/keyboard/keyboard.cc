#include <sys/syslog.h>
#include <syslog.h>

#include <map>
#include <set>
#include <string>
#include <variant>
#include <mutex>
#include <algorithm>
#include <functional>
#include <regex>
#include <thread>

#include "keyboard.hh"
#include "canonical_names.hh"
#include "generic.hh"
#include "keyboard_event.hh"
#include "nix_keyboard.hh"
#include "nix_common.hh"

typedef std::vector<std::vector<std::set<int>>> ParsedHotkey;

std::set<int> _modifier_scan_codes;

void press(Hotkey hotkey);

void unhook_all_hotkeys();

/*
Returns a list of scan codes associated with this key (name or scan code).
*/
std::set<int> key_to_scan_codes(Key key, bool error_if_missing = true)
{
    if (std::holds_alternative<int>(key))
        return std::set<int>{std::get<int>(key)};

    std::string normalized = normalize_name(std::get<std::string>(key));
    if (std::find(begin(sided_modifiers), end(sided_modifiers), normalized) != end(sided_modifiers))
    {
        std::set<int> left_scan_codes = key_to_scan_codes("left " + normalized, false);
        std::set<int> right_scan_codes = key_to_scan_codes("right " + normalized, false);
        std::set<int> scan_codes(left_scan_codes);
        scan_codes.insert(begin(right_scan_codes), end(right_scan_codes));
        return scan_codes;
    }

    std::vector<KeyAndModifiers> name_map = keyboard_map_name(normalized);
    std::set<int> scan_codes;
    std::transform(begin(name_map), end(name_map), std::inserter(scan_codes, scan_codes.begin()), [](auto &key)
                   { return key.first; });

    if (scan_codes.empty() && error_if_missing)
    {
        if (std::holds_alternative<int>(key))
            syslog(LOG_ERR, "Key %d is not mapped to any known key.", std::get<int>(key));
        else
            syslog(LOG_ERR, "Key %s is not mapped to any known key.", std::get<std::string>(key).c_str());
        exit(1);
    }
    else
    {
        return scan_codes;
    }
}

std::set<int> key_to_scan_codes(std::vector<Key> key, bool error_if_missing = true)
{
    std::set<int> scan_codes;
    for (auto &i : key)
    {
        std::set<int> result = key_to_scan_codes(i);
        scan_codes.insert(begin(result), end(result));
    }
    return scan_codes;
}

/*
Returns True if `key` is a scan code or name of a modifier key.
*/
bool is_modifier(Key key)
{
    if (std::holds_alternative<std::string>(key))
        return (std::find(begin(all_modifiers), end(all_modifiers), std::get<std::string>(key)) != end(all_modifiers));
    else
    {
        if (_modifier_scan_codes.empty())
        {
            std::set<int> scan_codes;
            for (auto &name : all_modifiers)
            {
                std::set<int> _scan_codes = key_to_scan_codes(name, false);
                scan_codes.insert(begin(_scan_codes), end(_scan_codes));
            }
            _modifier_scan_codes.insert(begin(scan_codes), end((scan_codes)));
        }
        return _modifier_scan_codes.contains(std::get<int>(key));
    }
}

std::mutex _pressed_events_lock;
std::map<int, KeyboardEvent> _pressed_events;
std::map<int, KeyboardEvent> _physically_pressed_keys;
std::map<int, KeyboardEvent> _logically_pressed_keys;

std::map<std::tuple<std::string, int, std::string>, std::tuple<bool, std::optional<bool>, std::string>> transition_table{
    // Current state of the modifier, per `modifier_states`.
    //|
    //|             Type of event that triggered this modifier update.
    //|             |
    //|             |         Type of key that triggered this modifier update.
    //|             |         |
    //|             |         |            Should we send a fake key press?
    //|             |         |            |
    //|             |         |     =>     |       Accept the event?
    //|             |         |            |       |
    //|             |         |            |       |              Next state.
    // v             v         v            v       v              v
    {{"free", KEY_UP, "modifier"}, {false, true, "free"}},
    {{"free", KEY_DOWN, "modifier"}, {false, false, "pending"}},
    {{"pending", KEY_UP, "modifier"}, {true, true, "free"}},
    {{"pending", KEY_DOWN, "modifier"}, {false, true, "allowed"}},
    {{"suppressed", KEY_UP, "modifier"}, {false, false, "free"}},
    {{"suppressed", KEY_DOWN, "modifier"}, {false, false, "suppressed"}},
    {{"allowed", KEY_UP, "modifier"}, {false, true, "free"}},
    {{"allowed", KEY_DOWN, "modifier"}, {false, true, "allowed"}},

    {{"free", KEY_UP, "hotkey"}, {false, std::nullopt, "free"}},
    {{"free", KEY_DOWN, "hotkey"}, {false, std::nullopt, "free"}},
    {{"pending", KEY_UP, "hotkey"}, {false, std::nullopt, "suppressed"}},
    {{"pending", KEY_DOWN, "hotkey"}, {false, std::nullopt, "suppressed"}},
    {{"suppressed", KEY_UP, "hotkey"}, {false, std::nullopt, "suppressed"}},
    {{"suppressed", KEY_DOWN, "hotkey"}, {false, std::nullopt, "suppressed"}},
    {{"allowed", KEY_UP, "hotkey"}, {false, std::nullopt, "allowed"}},
    {{"allowed", KEY_DOWN, "hotkey"}, {false, std::nullopt, "allowed"}},

    {{"free", KEY_UP, "other"}, {false, true, "free"}},
    {{"free", KEY_DOWN, "other"}, {false, true, "free"}},
    {{"pending", KEY_UP, "other"}, {true, true, "allowed"}},
    {{"pending", KEY_DOWN, "other"}, {true, true, "allowed"}},
    // Necessary when hotkeys are removed after beign triggered, such as
    // TestKeyboard.test_add_hotkey_multistep_suppress_modifier.
    {{"suppressed", KEY_UP, "other"}, {false, false, "allowed"}},
    {{"suppressed", KEY_DOWN, "other"}, {true, true, "allowed"}},
    {{"allowed", KEY_UP, "other"}, {false, true, "allowed"}},
    {{"allowed", KEY_DOWN, "other"}, {false, true, "allowed"}},
};

extern std::shared_ptr<AggregatedEventDevice> device;

class KeyboardListener : public GenericListener
{

public:
    virtual ~KeyboardListener()
    {
        if (listening)
        {
            queue.shutdown();
            processing_thread.join();
            device->event_queue.shutdown();
            listening_thread.join();
        }
    }

    void init()
    {
        keyboard_init();

        active_modifiers = std::set<int>();
        blocking_hooks = std::list<Handler>();
        blocking_keys = std::map<int, std::list<Handler>>();
        nonblocking_keys = std::map<int, std::list<Handler>>();
        blocking_hotkeys = std::map<std::set<int>, std::list<Handler>>();
        nonblocking_hotkeys = std::map<std::set<int>, std::list<Handler>>();
        filtered_modifiers = std::map<int, int>();
        is_replaying = false;

        // Supporting hotkey suppression is harder than it looks. See
        // https://github.com/boppreh/keyboard/issues/22
        modifier_states = std::map<int, std::string>(); // "alt" -> "allowed"
    }

    bool pre_process_event(KeyboardEvent event)
    {
        for (auto &key_hook : nonblocking_keys[*event.scan_code])
            key_hook(event);

        std::set<int> hotkey;
        {
            std::lock_guard<std::mutex> lock(_pressed_events_lock);
            std::transform(begin(_pressed_events), end(_pressed_events), std::inserter(hotkey, begin(hotkey)), [](auto &event)
                           { return event.first; });
        }
        for (auto &callback : nonblocking_hotkeys[hotkey])
            callback(event);

        return event.scan_code || (event.name && *event.name != "unknown");
    }

    /*
    This function is called for every OS keyboard event and decides if the
    event should be blocked or not, and passes a copy of the event to
    other, non-blocking, listeners.
    There are two ways to block events: remapped keys, which translate
    events by suppressing and re-emitting; and blocked hotkeys, which
    suppress specific hotkeys.
    */
    bool direct_callback(KeyboardEvent event)
    {
        // Pass through all fake key events, don't even report to other handlers.
        if (is_replaying)
            return true;

        std::vector<bool> hook_results;
        for (auto &hook : blocking_hooks)
            hook_results.push_back(hook(event));
        if (!std::all_of(begin(hook_results), end(hook_results), std::identity()))
            return false;

        int event_type = event.event_type;
        int scan_code = *event.scan_code;

        std::set<int> hotkey;

        // Update tables of currently pressed keys and modifiers.
        {
            std::lock_guard<std::mutex> lock(_pressed_events_lock);
            if (event_type == KEY_DOWN)
            {
                if (is_modifier(scan_code))
                    active_modifiers.insert(scan_code);
                _pressed_events[scan_code] = event;
            }
            std::transform(begin(_pressed_events), end(_pressed_events), std::inserter(hotkey, begin(hotkey)), [](auto &event)
                           { return event.first; });
            if (event_type == KEY_UP)
            {
                active_modifiers.erase(scan_code);
                _pressed_events.erase(scan_code);
            }
        }

        // Mappings based on individual keys instead of hotkeys.
        for (auto &key_hook : blocking_keys[scan_code])
        {
            if (!key_hook(event))
                return false;
        }

        // Default accept.
        bool accept = true;

        if (!blocking_hotkeys.empty())
        {
            std::string origin;
            std::set<int> modifiers_to_update;
            if (filtered_modifiers[scan_code])
            {
                origin = "modifier";
                modifiers_to_update = std::set<int>{scan_code};
            }
            else
            {
                modifiers_to_update = active_modifiers;
                if (is_modifier(scan_code))
                    modifiers_to_update.insert(scan_code);
                std::vector<bool> callback_results;
                for (auto &callback : blocking_hotkeys[hotkey])
                    callback_results.push_back(callback(event));
                if (!callback_results.empty())
                {
                    accept = std::all_of(begin(callback_results), end(callback_results), std::identity());
                    origin = "hotkey";
                }
                else
                {
                    origin = "other";
                }
            }

            for (auto &key : modifiers_to_update)
            {
                std::string modifier_state = modifier_states[key];
                if (modifier_state == "")
                    modifier_state = "free";
                std::tuple<std::string, int, std::string> transition_tuple{modifier_state, event_type, origin};
                auto [should_press, new_accept, new_state] = transition_table[transition_tuple];
                if (should_press)
                    press(key);
                if (new_accept)
                    accept = *new_accept;
                modifier_states[key] = new_state;
            }
        }

        if (accept)
        {
            if (event_type == KEY_DOWN)
                _logically_pressed_keys[scan_code] = event;
            else if (event_type == KEY_UP)
                _logically_pressed_keys.erase(scan_code);
        }

        // Queue for handlers that won't block the event.
        queue.push(event);

        return accept;
    }

    void listen()
    {
        keyboard_listen([this](KeyboardEvent event)
                        { return direct_callback(event); });
    }

    std::set<int> active_modifiers;
    std::list<Handler> blocking_hooks;
    std::map<int, std::list<Handler>> blocking_keys;
    std::map<int, std::list<Handler>> nonblocking_keys;
    std::map<std::set<int>, std::list<Handler>> blocking_hotkeys;
    std::map<std::set<int>, std::list<Handler>> nonblocking_hotkeys;
    std::map<int, int> filtered_modifiers;
    bool is_replaying;
    std::map<int, std::string> modifier_states;
};

KeyboardListener _listener;

/*
Parses a user-provided hotkey into nested tuples representing the
parsed structure, with the bottom values being lists of scan codes.
Also accepts raw scan codes, which are then wrapped in the required
number of nestings.
Example:
    parse_hotkey("alt+shift+a, alt+b, c")
    #    Keys:    ^~^ ^~~~^ ^  ^~^ ^  ^
    #    Steps:   ^~~~~~~~~~^  ^~~~^  ^
    # ((alt_codes, shift_codes, a_codes), (alt_codes, b_codes), (c_codes,))
*/
ParsedHotkey parse_hotkey(Hotkey hotkey_)
{
    if (std::holds_alternative<Key>(hotkey_))
    {
        Key hotkey = std::get<Key>(hotkey_);
        if (std::holds_alternative<int>(hotkey) || (std::holds_alternative<std::string>(hotkey) && (std::get<std::string>(hotkey).length() == 1)))
        {
            std::set<int> scan_codes = key_to_scan_codes(hotkey);
            ParsedHotkey steps{{scan_codes}};
            return steps;
        }
        else
        {
            ParsedHotkey steps;
            std::regex step_regex(",\\s?"), key_regex("\\s?\\+\\s?");
            std::sregex_token_iterator rend;
            std::sregex_token_iterator step_iter(begin(std::get<std::string>(hotkey)), end(std::get<std::string>(hotkey)), step_regex, -1);
            for (; step_iter != rend; ++step_iter)
            {
                std::string step_str = *step_iter;
                std::sregex_token_iterator key_iter(begin(step_str), end(step_str), key_regex, -1);
                std::vector<std::set<int>> step;
                for (; key_iter != rend; ++key_iter)
                {
                    std::string key = *key_iter;
                    step.push_back(key_to_scan_codes(key));
                }
                steps.push_back(step);
            }
            return steps;
        }
    }
    else
    {
        std::vector<std::set<int>> step;
        for (auto &k : std::get<std::vector<Key>>(hotkey_))
        {
            step.push_back(key_to_scan_codes(k));
        }
        return ParsedHotkey{step};
    }
}

/*
Sends OS events that perform the given *hotkey* hotkey.
- `hotkey` can be either a scan code (e.g. 57 for space), single key
(e.g. 'space') or multi-key, multi-step hotkey (e.g. 'alt+F4, enter').
- `do_press` if true then press events are sent. Defaults to True.
- `do_release` if true then release events are sent. Defaults to True.
    send(57)
    send('ctrl+alt+del')
    send('alt+F4, enter')
    send('shift+s')
Note: keys are released in the opposite order they were pressed.
*/
void send(Hotkey hotkey, bool do_press = true, bool do_release = true)
{
    _listener.is_replaying = true;

    ParsedHotkey parsed = parse_hotkey(hotkey);
    for (auto step : parsed)
    {
        if (do_press)
        {
            for (auto &scan_codes : step)
            {
                keyboard_press(*begin(scan_codes));
            }
        }
        if (do_release)
        {
            std::reverse(begin(step), end(step));
            for (auto &scan_codes : step)
            {
                keyboard_release(*begin(scan_codes));
            }
        }
    }

    _listener.is_replaying = false;
}

void press_and_release(Hotkey hotkey)
{
    send(hotkey);
}

/* Presses and holds down a hotkey (see `send`). */
void press(Hotkey hotkey)
{
    send(hotkey, true, false);
}

/* Releases a hotkey (see `send`). */
void release(Hotkey hotkey)
{
    send(hotkey, false, true);
}

/*
Returns True if the key is pressed.
    is_pressed(57) #-> True
    is_pressed('space') #-> True
    is_pressed('ctrl+space') #-> True
*/
bool is_pressed(Hotkey hotkey)
{
    _listener.start_if_necessary();

    if (std::holds_alternative<Key>(hotkey) && std::holds_alternative<int>(std::get<Key>(hotkey)))
    {
        std::lock_guard lock(_pressed_events_lock);
        return _pressed_events.contains(std::get<int>(std::get<Key>(hotkey)));
    }

    ParsedHotkey steps = parse_hotkey(hotkey);
    if (steps.size() > 1)
    {
        syslog(LOG_ERR, "Impossible to check if multi-step hotkeys are pressed (`a+b` is ok, `a, b` isn't).");
        exit(1);
    }

    // Convert _pressed_events into a set
    std::set<int> pressed_scan_codes;
    {
        std::lock_guard lock(_pressed_events_lock);
        std::transform(begin(_pressed_events), end(_pressed_events), std::inserter(pressed_scan_codes, begin(pressed_scan_codes)), [](auto &event)
                       { return event.first; });
    }
    for (auto &scan_codes : steps[0])
    {
        if (std::none_of(begin(scan_codes), end(scan_codes), [&](auto &scan_code)
                         { return pressed_scan_codes.contains(scan_code); }))
            return false;
    }
    return true;
}

namespace std
{
    template <>
    struct hash<HandlerIter>
    {
        std::size_t operator()(HandlerIter const &iter) const noexcept
        {
            return (std::size_t) & (*iter);
        }
    };
}

template <class T>
inline void hash_combine(std::size_t &seed, T const &v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
    template <typename T>
    struct hash<vector<T>>
    {
        typedef vector<T> argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const &in) const
        {
            size_t size = in.size();
            size_t seed = 0;
            for (size_t i = 0; i < size; i++)
                // Combine the hash of the current vector with the hashes of the previous ones
                hash_combine(seed, in[i]);
            return seed;
        }
    };
}

std::unordered_map<std::variant<HandlerIter, Key>, RemoveFunction> _hooks;

/*
Installs a global listener on all available keyboards, invoking `callback`
each time a key is pressed or released.

The event passed to the callback is of type `keyboard.KeyboardEvent`,
with the following attributes:
- `name`: an Unicode representation of the character (e.g. "&") or
description (e.g.  "space"). The name is always lower-case.
- `scan_code`: number representing the physical key, e.g. 55.
- `time`: timestamp of the time the event occurred, with as much precision
as given by the OS.
Returns the given callback for easier development.
*/
HookResult hook(Handler callback, bool suppress = false, RemoveCallback on_remove = RemoveCallback())
{
    HandlerIter callback_iter = _listener.add_handler(callback);
    RemoveFunction remove_{[=]()
                           {
                               _hooks.erase(callback_iter);
                               _listener.remove_handler(callback_iter);
                               on_remove();
                           }};
    _hooks[callback_iter] = remove_;
    return std::make_pair(callback_iter, remove_);
}

/*
Invokes `callback` for every KEY_DOWN event. For details see `hook`.
*/
HookResult on_press(Handler callback, bool suppress = false)
{
    return hook([=](KeyboardEvent e)
                { return ((e.event_type == KEY_UP) || callback(e)); },
                suppress);
}

/*
Invokes `callback` for every KEY_UP event. For details see `hook`.
*/
HookResult on_release(Handler callback, bool suppress = false)
{
    return hook([=](KeyboardEvent e)
                { return ((e.event_type == KEY_DOWN) || callback(e)); },
                suppress);
}

/*
Hooks key up and key down events for a single key. Returns the event handler
created. To remove a hooked key use `unhook_key(key)` or
`unhook_key(handler)`.
Note: this function shares state with hotkeys, so `clear_all_hotkeys`
affects it as well.
*/
HookResult hook_key(Key key, Handler callback, bool suppress = false)
{
    _listener.start_if_necessary();
    std::set<int> scan_codes = key_to_scan_codes(key);
    std::map<int, HandlerIter> callback_iters_map;
    for (auto &scan_code : scan_codes)
    {
        callback_iters_map[scan_code] = _listener.nonblocking_keys[scan_code].insert(end(_listener.nonblocking_keys[scan_code]), callback);
    }

    RemoveFunction remove_{[=]() mutable
                           {
                               _hooks.erase(key);
                               for (auto &scan_code : scan_codes)
                               {
                                   _listener.nonblocking_keys[scan_code].erase(callback_iters_map[scan_code]);
                               }
                           }};
    _hooks[key] = remove_;
    std::list<HandlerIter> callback_iters;
    std::transform(begin(callback_iters_map), end(callback_iters_map), std::back_inserter(callback_iters), [](auto &callback_iter_pair)
                   { return callback_iter_pair.second; });
    return std::make_pair(callback_iters, remove_);
}

/*
Invokes `callback` for KEY_DOWN event related to the given key. For details see `hook`.
*/
HookResult on_press_key(Key key, Handler callback, bool suppress = false)
{
    return hook_key(
        key, [=](KeyboardEvent e)
        { return ((e.event_type == KEY_UP) || callback(e)); },
        suppress);
}

/*
Invokes `callback` for KEY_UP event related to the given key. For details see `hook`.
*/
HookResult on_release_key(Key key, Handler callback, bool suppress = false)
{
    return hook_key(
        key, [=](KeyboardEvent e)
        { return ((e.event_type == KEY_DOWN) || callback(e)); },
        suppress);
}

/*
Removes a previously added hook, either by callback or by the return value
of `hook`.
*/
void unhook(HandlerIter remove)
{
    _hooks[remove]();
}

void unhook_key(Key remove)
{
    _hooks[remove]();
}

/*
Removes all keyboard hooks in use, including hotkeys, abbreviations, word
listeners, `record`ers and `wait`s.
*/
void unhook_all()
{
    _listener.start_if_necessary();
    _listener.blocking_keys.clear();
    _listener.nonblocking_keys.clear();
    _listener.blocking_hooks.clear();
    _listener.handlers.clear();
    unhook_all_hotkeys();
}

void CartesianRecurse(std::vector<std::set<int>> &accum, std::vector<int> stack,
                      std::vector<std::set<int>> sets, int index)
{
    std::set<int> set = sets[index];
    for (int i : set)
    {
        stack.push_back(i);
        if (index == 0)
            accum.push_back(std::set<int>(begin(stack), end(stack)));
        else
            CartesianRecurse(accum, stack, sets, index - 1);
        stack.pop_back();
    }
}

std::vector<std::set<int>> CartesianProduct(std::vector<std::set<int>> sets)
{
    std::vector<std::set<int>> accum;
    std::vector<int> stack;
    if (sets.size() > 0)
        CartesianRecurse(accum, stack, sets, sets.size() - 1);
    return accum;
}

/*
Parses a user-provided hotkey. Differently from `parse_hotkey`,
instead of each step being a list of the different scan codes for each key,
each step is a list of all possible combinations of those scan codes.
*/
ParsedHotkey parse_hotkey_combinations(Hotkey hotkey)
{
    ParsedHotkey parsed = parse_hotkey(hotkey);
    ParsedHotkey parsed_combinations;
    for (auto &step : parsed)
    {
        parsed_combinations.push_back(CartesianProduct(step));
    }
    return parsed_combinations;
}

/*
Hooks a single-step hotkey (e.g. 'shift+a').
*/
HookResult _add_hotkey_step(Handler handler, std::vector<std::set<int>> combinations, bool suppress)
{
    // Register the scan codes of every possible combination of
    // modfiier + main key. Modifiers have to be registered in
    // filtered_modifiers too, so suppression and replaying can work.
    HandlerIter handler_iter;
    for (auto &scan_codes : combinations)
    {
        for (auto &scan_code : scan_codes)
        {
            if (is_modifier(scan_code))
                _listener.filtered_modifiers[scan_code] += 1;
        }
        handler_iter = _listener.nonblocking_hotkeys[scan_codes].insert(begin(_listener.nonblocking_hotkeys[scan_codes]), handler);
    }

    RemoveFunction remove{[=]()
                          {
                              for (auto &scan_codes : combinations)
                              {
                                  for (auto &scan_code : scan_codes)
                                  {
                                      if (is_modifier(scan_code))
                                          _listener.filtered_modifiers[scan_code] -= 1;
                                  }
                                  _listener.nonblocking_hotkeys[scan_codes].erase(handler_iter);
                              }
                          }};
    return std::make_pair(handler_iter, remove);
}

std::unordered_map<std::variant<HandlerIter, Hotkey>, RemoveFunction> _hotkeys;

struct State
{
    int index;
    RemoveFunction remove_catch_misses;
    RemoveFunction remove_last_step;
    std::list<KeyboardEvent> suppressed_events;
    time_point last_update;
};

/*
Invokes a callback every time a hotkey is pressed. The hotkey must
be in the format `ctrl+shift+a, s`. This would trigger when the user holds
ctrl, shift and "a" at once, releases, and then presses "s". To represent
literal commas, pluses, and spaces, use their names ('comma', 'plus',
'space').
- `args` is an optional list of arguments to passed to the callback during
each invocation.
- `suppress` defines if successful triggers should block the keys from being
sent to other programs.
- `timeout` is the amount of seconds allowed to pass between key presses.
- `trigger_on_release` if true, the callback is invoked on key release instead
of key press.
The event handler function is returned. To remove a hotkey call
`remove_hotkey(hotkey)` or `remove_hotkey(handler)`.
before the hotkey state is reset.
Note: hotkeys are activated when the last key is *pressed*, not released.
Note: the callback is executed in a separate thread, asynchronously. For an
example of how to use a callback synchronously, see `wait`.
Examples:
    # Different but equivalent ways to listen for a spacebar key press.
    add_hotkey(' ', print, args=['space was pressed'])
    add_hotkey('space', print, args=['space was pressed'])
    add_hotkey('Space', print, args=['space was pressed'])
    # Here 57 represents the keyboard code for spacebar; so you will be
    # pressing 'spacebar', not '57' to activate the print function.
    add_hotkey(57, print, args=['space was pressed'])
    add_hotkey('ctrl+q', quit)
    add_hotkey('ctrl+alt+enter, space', some_callback)
*/
HookResult
add_hotkey(Hotkey hotkey, std::function<bool()> callback, bool suppress, int timeout, bool trigger_on_release)
{
    _listener.start_if_necessary();

    ParsedHotkey steps = parse_hotkey_combinations(hotkey);

    int event_type = trigger_on_release ? KEY_UP : KEY_DOWN;
    if (steps.size() == 1)
    {
        // Deciding when to allow a KEY_UP event is far harder than I thought,
        // and any mistake will make that key "sticky". Therefore just let all
        // KEY_UP events go through as long as that's not what we are listening
        // for.
        Handler handler{[=](KeyboardEvent e)
                        {
                            return (event_type == KEY_DOWN && e.event_type == KEY_UP && _logically_pressed_keys.contains(*e.scan_code)) || (event_type == e.event_type && callback());
                        }};
        auto [hook_result, remove_step] = _add_hotkey_step(handler, steps[0], suppress);
        HandlerIter callback_iter = std::get<HandlerIter>(hook_result);
        RemoveFunction remove_{[=]()
                               {
                                   remove_step();
                                   _hotkeys.erase(hotkey);
                                   _hotkeys.erase(callback_iter);
                               }};
        // TODO: allow multiple callbacks for each hotkey without overwriting the
        // remover.
        _hotkeys[hotkey] = _hotkeys[callback_iter] = remove_;
        return std::make_pair(callback_iter, remove_);
    }

    std::shared_ptr<State> state = std::make_shared<State>();

    std::shared_ptr<std::function<bool(int)>> set_index = std::make_shared<std::function<bool(int)>>();

    std::shared_ptr<std::vector<std::set<int>>> allowed_keys_by_step = std::make_shared<std::vector<std::set<int>>>();
    std::shared_ptr<std::function<bool(KeyboardEvent, bool)>> catch_misses = std::make_shared<std::function<bool(KeyboardEvent, bool)>>([=](KeyboardEvent event, bool force_fail = false)
                                                                                                                                        {
                                                              if ((
                                                                      event.event_type == event_type && state->index && !(*allowed_keys_by_step)[state->index].contains(*event.scan_code)
                                                                      ) ||
                                                                  (timeout && (now() - state->last_update) >= std::chrono::seconds(timeout)) || force_fail)
                                                              { // Weird formatting to ensure short-circuit.

                                                                  state->remove_last_step();

                                                                  for (auto &event : state->suppressed_events)
                                                                      if (event.event_type == KEY_DOWN)
                                                                          press(*event.scan_code);
                                                                      else
                                                                          release(*event.scan_code);
                                                                  state->suppressed_events.clear();

                                                                  (*set_index)(0);
                                                              }
                                                              return true; });

    HandlerIter callback_iter;
    RemoveFunction remove;
    *set_index = std::function<bool(int)>{[=](int new_index)
                                          {
                                              state->index = new_index;

                                              if (new_index == 0)
                                              {
                                                  // This is done for performance reasons, avoiding a global key hook
                                                  // that is always on.
                                                  state->remove_catch_misses();
                                                  state->remove_catch_misses = RemoveFunction();
                                              }
                                              else if (new_index == 1)
                                              {
                                                  state->remove_catch_misses();
                                                  // Must be `suppress=True` to ensure `send` has priority.
                                                  std::tie(std::ignore, state->remove_catch_misses) = hook(std::bind(*catch_misses, std::placeholders::_1, false), true);
                                              }

                                              if (new_index == steps.size() - 1)
                                              {
                                                  Handler handler{[=](KeyboardEvent event)
                                                                  {
                                                                      if (event.event_type == KEY_UP)
                                                                      {
                                                                          remove();
                                                                          (*set_index)(0);
                                                                      }
                                                                      bool accept = (event.event_type == event_type) && callback();
                                                                      if (accept)
                                                                          return (*catch_misses)(event, true);
                                                                      else
                                                                      {
                                                                          state->suppressed_events = std::list<KeyboardEvent>{event};
                                                                          return false;
                                                                      }
                                                                  }};
                                                  auto [callback_iter, remove] = _add_hotkey_step(handler, steps[state->index], suppress);
                                              }
                                              else
                                              {
                                                  // Fix value of next_index.
                                                  std::function<bool(KeyboardEvent, int)> handler{[=](KeyboardEvent event, int new_index)
                                                                                                  {
                                                                                                      if (event.event_type == KEY_UP)
                                                                                                      {
                                                                                                          remove();
                                                                                                          (*set_index)(new_index);
                                                                                                      }
                                                                                                      state->suppressed_events.push_back(event);
                                                                                                      return false;
                                                                                                  }};
                                                  auto [callback_iter, remove] = _add_hotkey_step(std::bind(handler, std::placeholders::_1, state->index + 1), steps[state->index], suppress);
                                              }
                                              state->remove_last_step = remove;
                                              state->last_update = now();
                                              return false;
                                          }};
    (*set_index)(0);

    for (auto &step : steps)
    {
        std::set<int> allowed_keys;
        for (auto &scan_codes : step)
        {
            allowed_keys.insert(begin(scan_codes), end(scan_codes));
        }
        allowed_keys_by_step->push_back(allowed_keys);
    }

    RemoveFunction remove_{[=]()
                           {
                               state->remove_catch_misses();
                               state->remove_last_step();
                               _hotkeys.erase(hotkey);
                               _hotkeys.erase(callback_iter);
                           }};
    // TODO: allow multiple callbacks for each hotkey without overwriting the
    // remover.
    _hotkeys[hotkey] = _hotkeys[callback_iter] = remove_;
    return std::make_pair(callback_iter, remove_);
}

/*
Removes a previously hooked hotkey. Must be called with the value returned
by `add_hotkey`.
*/
void remove_hotkey(std::variant<HandlerIter, Hotkey> hotkey_or_callback)
{
    _hotkeys[hotkey_or_callback]();
}

/*
Removes all keyboard hotkeys in use, including abbreviations, word listeners,
`record`ers and `wait`s.
*/
void unhook_all_hotkeys()
{
    // Because of "alises" some hooks may have more than one entry, all of which
    // are removed together.
    _listener.blocking_hotkeys.clear();
    _listener.nonblocking_hotkeys.clear();
}
