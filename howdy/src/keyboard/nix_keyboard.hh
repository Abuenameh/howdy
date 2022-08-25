#ifndef NIX_KEYBOARD_H_
#define NIX_KEYBOARD_H_

#include "generic.hh"

typedef std::pair<int, std::vector<std::string>> KeyAndModifiers;

void keyboard_init();

void keyboard_listen(Handler callback);

std::vector<KeyAndModifiers> keyboard_map_name(std::string name);

void keyboard_press(int scan_code);

void keyboard_release(int scan_code);

#endif // NIX_KEYBOARD_H_