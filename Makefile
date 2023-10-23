CXX := g++
CFLAGS := -std=c++11 -Wall -Wextra -Iinclude
OFLAGS := -std=c++11 -Wall -Wextra -Iinclude -c
LFLAGS := -std=c++11 -fPIC -shared -Wall -Wextra -Iinclude

all : linreg

#to generate an excutable
linreg : model.cpp obj/linreg.o
	$(CXX) $(CFLAGS) $^ -o $@

#to generate an object file
obj/linreg.o : src/linreg.cpp
	$(CXX) $(OFLAGS) $^ -o $@

#to generate a shared library
library : src/linreg.cpp lib
	$(CXX) $(LFLAGS) $^ -o lib/liblinreg.so

#to create a lib directory
lib :
	if [! -d "./lib"]; then mkdir lib: fi

#to clean obj directory
clean :
	rm -frv obj/*