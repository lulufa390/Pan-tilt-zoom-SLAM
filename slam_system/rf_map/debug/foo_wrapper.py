from ctypes import cdll
lib = cdll.LoadLibrary('./libfoo.so')

class Foo(object):
    def __init__(self):
        self.obj = lib.Foo_new()

    def bar(self):
        print('type {}'.format(type(self.obj)))
        lib.Foo_bar(self.obj)

f = Foo()
f.bar()