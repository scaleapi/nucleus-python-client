set -eux 
pip install --upgrade pip
pip install poetry
CIRCLE_CI_LOCK=./poetry.circleci.lock
if [[ -f $CIRCLE_CI_LOCK ]]; then
	cp $CIRCLE_CI_LOCK ./poetry.lock
fi
poetry lock -vv
