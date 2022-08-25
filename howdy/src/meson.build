project('howdy-auth', 'cpp', version: '3.0.0', default_options: ['cpp_std=c++2a'])

inih_cpp = dependency('INIReader', fallback: ['inih', 'INIReader_dep'])
dlib = dependency('dlib-1')
opencv = dependency('opencv4')
libevdev = dependency('libevdev')
add_global_arguments(['-Wno-unused', '-Wno-deprecated-enum-enum-conversion', '-Wno-sign-compare', '-Wno-bidi-chars'], language: 'cpp')

# Translations
#subdir('po')

executable(
	'howdy-auth',
	'compare.cc',
	'video_capture.cc',
	'models.cc',
	'snapshot.cc',
	'rubber_stamps.cc',
	'process/process.cpp',
	'process/process_unix.cpp',
	'keyboard/canonical_names.cc',
	'keyboard/generic.cc',
	'keyboard/keyboard_event.cc',
	'keyboard/nix_common.cc',
	'keyboard/nix_keyboard.cc',
	'keyboard/keyboard.cc',
	dependencies: [
		inih_cpp,
		dlib,
		opencv,
	]
)
