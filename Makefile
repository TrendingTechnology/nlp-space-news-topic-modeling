#################################################################################
# GLOBALS                                                                       #
#################################################################################

# api/ uses
# - DOCKERFILE_NAME=Dockerfile
# - APP_PORT=8050
# - IMAGE_NAME=tiangolo/uvicorn-gunicorn-fastapi:python3.8
# app/ uses
# - DOCKERFILE_NAME=Dockerfile_app
# - APP_PORT=5006
# - IMAGE_NAME_APP=python:3.8.6-slim-buster
DOCKERFILE_NAME_API=Dockerfile
APP_PORT_API=8050
IMAGE_NAME_API=tiangolo/uvicorn-gunicorn-fastapi:python3.8
DOCKERFILE_NAME_APP=Dockerfile_app
APP_PORT_APP=5006
IMAGE_NAME_APP=python:3.8.6-slim-buster
TAG=edesz/my-containerized-app
NAME=mycontainer
PORT_MAP_API=8000:$(APP_PORT_API)
PORT_MAP_APP=8000:$(APP_PORT_APP)

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

## Build API in container
container-api-build:
	@docker build -t $(TAG) \
		--build-arg AZURE_STORAGE_KEY_ARG=$(AZURE_STORAGE_KEY) \
		--build-arg ENDPOINT_SUFFIX_ARG=$(ENDPOINT_SUFFIX) \
		--build-arg AZURE_STORAGE_ACCOUNT_ARG=$(AZURE_STORAGE_ACCOUNT) \
		--build-arg PORT_ARG=$(APP_PORT_API) \
		-f $(DOCKERFILE_NAME_API) .
.PHONY: container-api-build

## Run API in container
container-api-run:
	@docker run -d -p $(PORT_MAP_API) \
	    -e APP_MODULE="main:app" \
		-e PORT="$(APP_PORT_API)" \
		-e BIND="0.0.0.0:$(APP_PORT_API)" \
		--name $(NAME) $(TAG)
.PHONY: container-api-run

## Build APP in container
container-app-build:
	@docker build -t $(TAG) \
		--build-arg AZURE_STORAGE_KEY_ARG=$(AZURE_STORAGE_KEY) \
		--build-arg ENDPOINT_SUFFIX_ARG=$(ENDPOINT_SUFFIX) \
		--build-arg AZURE_STORAGE_ACCOUNT_ARG=$(AZURE_STORAGE_ACCOUNT) \
		--build-arg PORT_ARG=5006 \
		-f $(DOCKERFILE_NAME_APP) .
.PHONY: container-app-build

## Run APP in container
container-app-run:
	@docker run -d -p $(PORT_MAP_APP) --name $(NAME) $(TAG)
.PHONY: container-app-run

## Show streaming container logs
container-logs:
	@docker ps -q | xargs -L 1 docker logs -f
.PHONY: container-logs

## Stop API container
container-api-stop:
	@docker container stop $(NAME)
.PHONY: container-api-stop

## Remove API container
container-api-delete:
	@docker container rm $(NAME)
	@docker rmi $(IMAGE_NAME_API)
	@docker rmi $(TAG)
.PHONY: container-api-delete

## Stop APP container
container-app-stop:
	@docker container stop $(NAME)
.PHONY: container-app-stop

## Remove APP container
container-app-delete:
	@docker container rm $(NAME)
	@docker rmi $(IMAGE_NAME_APP)
	@docker rmi $(TAG)
.PHONY: container-app-delete

## Remove all containers
container-delete-all:
	@docker rm -f $(docker ps -a -q)
	@docker rmi -f $(docker images -a -q)
.PHONY: container-delete-all

## Run API tests with tox
api-test:
	tox -e test -- -m "not scrapingtest"
.PHONY: api-test

## Run dashboard app with tox
app:
	tox -e app
.PHONY: app

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
