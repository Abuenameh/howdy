#ifndef STRING_UTILS_H_
#define STRING_UTILS_H_

#include <vector>
#include <string>

inline std::vector<std::string> split(const std::string &text, const std::string &delims)
{
	std::vector<std::string> tokens;
	std::size_t start = text.find_first_not_of(delims), end = 0;

	while ((end = text.find_first_of(delims, start)) != std::string::npos)
	{
		tokens.push_back(text.substr(start, end - start));
		start = text.find_first_not_of(delims, end);
	}
	if (start != std::string::npos)
		tokens.push_back(text.substr(start));

	return tokens;
}

inline std::vector<std::string> split(const std::string &text, const std::string &delims, int maxsplits)
{
	std::vector<std::string> tokens;
	std::size_t start = text.find_first_not_of(delims), end = 0;
    int numsplits = 0;

	while (((end = text.find_first_of(delims, start)) != std::string::npos) && numsplits < maxsplits)
	{
		tokens.push_back(text.substr(start, end - start));
		start = text.find_first_not_of(delims, end);
        numsplits++;
	}
	if (start != std::string::npos)
		tokens.push_back(text.substr(start));

	return tokens;
}

// trim from start (in place)
inline void ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
									{ return !std::isspace(ch); }));
}

// trim from end (in place)
inline void rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
						 { return !std::isspace(ch); })
				.base(),
			s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s)
{
	ltrim(s);
	rtrim(s);
}

// trim from start with custom characters (in place)
inline void ltrim(std::string &s, std::string chars)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char ch)
									{ return (chars.find(ch) == std::string::npos); }));
}

inline std::string tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), 
                   [](unsigned char c){ return std::tolower(c); }
                  );
    return s;
}


#endif // STRING_UTILS_H_