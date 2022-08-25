#include <map>
#include <set>
#include <vector>
#include <string>
#include <regex>

#include "../utils/static_block.hpp"

#include "canonical_names.hh"

std::map<std::string, std::string> canonical_names;
std::vector<std::string> sided_modifiers;
std::vector<std::string> all_modifiers;

static_block
{

    // Defaults to Windows canonical names (platform-specific overrides below)
    canonical_names = std::map<std::string, std::string>{
#include "canonical_names.incl"
    };
    sided_modifiers = std::vector<std::string>{"ctrl", "alt", "shift", "windows"};
    all_modifiers = std::vector<std::string>{"alt", "alt gr", "ctrl", "shift", "windows"};

    std::transform(begin(sided_modifiers), end(sided_modifiers), std::back_inserter(all_modifiers), [](std::string modifier)
                   { return "left " + modifier; });
    std::transform(begin(sided_modifiers), end(sided_modifiers), std::back_inserter(all_modifiers), [](std::string modifier)
                   { return "right " + modifier; });

    canonical_names.insert({
        {"select", "end"},
        {"find", "home"},
        {"next", "page down"},
        {"prior", "page up"},
    });
}

/*
Given a key name (e.g. "LEFT CONTROL"), clean up the string and convert to
the canonical representation (e.g. "left ctrl") if one is known.
*/
std::string normalize_name(std::optional<std::string> name_opt)
{
    if (!name_opt)
        throw std::invalid_argument("Can only normalize non-empty string names.");

    std::string name = *name_opt;
    if (name.length() > 1)
        std::transform(begin(name), end(name), begin(name), [](unsigned char c)
                       { return std::tolower(c); });
    if ((name != "_") && (name.find("_") != std::string::npos))
        name = std::regex_replace(name, std::regex("_"), " ");

    if (canonical_names.contains(name))
        return canonical_names[name];
    else
        return name;
}