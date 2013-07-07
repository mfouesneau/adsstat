cleantemp = rm -rf build; rm -f *.c

.PHONY : clean all build


all: clean build

build:
	python setup.py build_ext --inplace
	$(cleantmp)

clean: 
	$(cleantmp)
	find . -name '*pyc' -exec rm -f {} \;
	rm -f *.c *.so
	rm -rf build

