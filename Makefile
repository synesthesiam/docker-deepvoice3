.PHONY: run-vtck run-ljspeech test

all:
	mkdir -p download
	docker build . -t deepvoice3

run-vtck:
	docker run -it -p 5000:5000 deepvoice3

run-ljspeech:
	docker run -it -p 5000:5000 -e DEEPVOICE3_MODEL=ljspeech deepvoice3

test:
	curl -G --data-urlencode 'INPUT_TEXT=welcome to the world of speech synthesis' --data-urlencode 'VOICE=0' 'localhost:5000/process' --output - | aplay
