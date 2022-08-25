#ifndef CANONICAL_NAMES_H_
#define CANONICAL_NAMES_H_

#include <map>
#include <vector>
#include <string>
#include <optional>

extern std::map<std::string, std::string> canonical_names;
extern std::vector<std::string> sided_modifiers;
extern std::vector<std::string> all_modifiers;

std::string normalize_name(std::optional<std::string> name_opt);

#endif // CANONICAL_NAMES_H_
