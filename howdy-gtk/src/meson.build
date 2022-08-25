project('howdy-gtk', 'cpp', version: '3.0.0', default_options: ['cpp_std=c++2a'])

inih_cpp = dependency('INIReader', fallback: ['inih', 'INIReader_dep'])
dlib = dependency('dlib-1')
opencv = dependency('opencv4')
libevdev = dependency('libevdev')
gtk = dependency('gtkmm-3.0')
add_global_arguments(['-Wno-unused', '-Wno-deprecated-enum-enum-conversion', '-Wno-sign-compare'], language: 'cpp')

executable(
	'howdy-gtk',
	'init.cc',
	'window.cc',
	'onboarding.cc',
	'authsticky.cc',
	'../../howdy/src/process/process.cpp',
	'../../howdy/src/process/process_unix.cpp',
	dependencies: [
		inih_cpp,
		dlib,
		opencv,
		gtk,
	]
)
