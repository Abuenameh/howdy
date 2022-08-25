#ifndef KEYBOARD_H_
#define KEYBOARD_H_

#include <chrono>
#include <functional>
#include <variant>
#include <list>
#include <vector>

#include "../utils.hh"
#include "keyboard_event.hh"
#include "generic.hh"

typedef std::variant<int, std::string> Key;

typedef std::variant<Key, std::vector<Key>> Hotkey;

typedef std::function<void()> RemoveCallback;

typedef std::function<void()> RemoveFunction;

typedef std::pair<std::variant<HandlerIter, std::list<HandlerIter>>, RemoveFunction> HookResult;

HookResult add_hotkey(Hotkey hotkey, std::function<bool()> callback, bool suppress = false, int timeout = 1, bool trigger_on_release = false);

#endif // KEYBOARD_H_