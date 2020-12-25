#################################################################################
# GLOBALS                                                                       #
#################################################################################

IMAGE_NAME=python:3.8.6-slim-buster
TAG=edesz/fast-api-demo
NAME=mycontainer
PORT_MAP=8000:80

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf *.egg-info

## Run lint checks manually
lint:
	if [ ! -d .git ]; then git init && git add .; fi;
	tox -e lint
.PHONY: lint

## Run ci build with tox
ci:
	tox -e ci
.PHONY: ci

## Run build with tox
build:
	tox -e build
.PHONY: build

## Run API with tox
api:
	tox -e api
.PHONY: api

## Run api in container
container-api-run:
	docker build -t $(TAG) .
	docker run -d --name $(NAME) -p $(PORT_MAP) $(TAG)
.PHONY: container-api-run

## Show streaming api container logs
container-api-logs:
	docker ps -q | xargs -L 1 docker logs -f
.PHONY: container-api-logs

## Stop api container
container-api-stop:
	docker container stop $(NAME)
.PHONY: container-api-stop

## Remove api container
container-api-delete:
	docker container rm $(NAME)
	docker rmi $(IMAGE_NAME)
	docker rmi $(TAG)
.PHONY: container-api-delete

## Run API tests with tox
api-test:
	tox -e test -- -m "not scrapingtest"
.PHONY: api-test

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
