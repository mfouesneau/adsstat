cleantemp = rm -rf build; rm -f *.c

.PHONY : clean all build


all: clean build_wordcloud

build_wordcloud:
	cd wordcloud && make build

clean: 
	$(cleantmp)
	find . -name '*pyc' -exec rm -f {} \;
	cd wordcloud && make clean

