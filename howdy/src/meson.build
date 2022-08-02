project('howdy-auth', 'cpp', version: '3.0.0', default_options: ['cpp_std=c++2a'])

inih_cpp = dependency('INIReader', fallback: ['inih', 'INIReader_dep'])
dlib = dependency('dlib-1')
opencv = dependency('opencv4')
libevdev = dependency('libevdev')
# libdlib = meson.get_compiler('cpp').find_library('dlib')
#libpam = meson.get_compiler('cpp').find_library('pam')
#threads = dependency('threads')
add_global_arguments(['-Wno-unused', '-Wno-deprecated-enum-enum-conversion'], language: 'cpp')

# Translations
#subdir('po')

executable(
	'howdy-auth',
	'compare.cc',
	'video_capture.cc',
#	'enter_device.cc',
	dependencies: [
#		libpam,
		inih_cpp,
		dlib,
		opencv,
#		threads,
#		libevdev,
	]
#	install: false,
#	install_dir: '/lib64/security',
#	name_prefix: ''
)
