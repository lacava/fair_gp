#docker run -it quay.io/travisci/travis-ruby /bin/bash

git clone https://github.com/lacava/feat/ -b travis

sh feat/ci/.travis_install.sh
sh feat/ci/.travis_build.sh

