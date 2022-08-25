project('howdy', 'cpp', version: '3.0.0', default_options: ['cpp_std=c++2a'])

inih_cpp = dependency('INIReader', fallback: ['inih', 'INIReader_dep'])
dlib = dependency('dlib-1')
opencv = dependency('opencv4')
libevdev = dependency('libevdev')
add_global_arguments(['-Wno-unused', '-Wno-deprecated-enum-enum-conversion', '-Wno-sign-compare'], language: 'cpp')

executable(
	'howdy',
	'cli.cc',
	'add.cc',
	'clear.cc',
	'config.cc',
	'disable.cc',
	'list.cc',
	'remove.cc',
	'set.cc',
	'snap.cc',
	'test.cc',
	'../models.cc',
	'../video_capture.cc',
	'../snapshot.cc',
	dependencies: [
		inih_cpp,
		dlib,
		opencv,
	]
)
