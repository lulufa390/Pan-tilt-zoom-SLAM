#include <iostream>

class Koo {
public:
        void bar()
        {
            printf("world \n");
        }

};
class Foo{
    private:
    Koo k_;
    public:
        void bar(){
            std::cout << "Hello" << std::endl;
            k_.bar();
        }
};

extern "C" {
    Foo* Foo_new(){
    Foo* p_foo = new Foo();
    printf("Before: address %p\n", (void*)p_foo);
    return p_foo; }
    void Foo_bar(Foo* foo){
    printf("After: address %p\n", (void*)foo);
    foo->bar();
     }
}


