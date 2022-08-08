<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [e3psi](#e3psi)
  - [About](#about)
  - [Development](#development)
    - [First things first](#first-things-first)
    - [Getting Started](#getting-started)
    - [Tests](#tests)
    - [Contributing](#contributing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# e3psi

[![python3](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![CircleCI](https://circleci.com/gh/muhrin/e3psi/tree/master.svg?style=svg)](https://circleci.com/gh/muhrin/e3psi/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/muhrin/e3psi/badge.svg)](https://coveralls.io/github/muhrin/e3psi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/muhrin/e3psi/?ref=repository-badge)
[![dependabot](https://badgen.net/dependabot/muhrin/e3psi/?icon=dependabot)](https://dependabot.com/)
[![Requirements Status](https://requires.io/github/muhrin/e3psi/requirements.svg?branch=master)](https://requires.io/github/muhrin/e3psi/requirements/?branch=master)

## About

Equivariant machine learning library for learning from electronic structures


## Development

### First things first

You can develop on Windows, GNU/Linux or Mac OS X. You need:

- [Python 3.6 and above](https://www.python.org/) and a python [**virtual environment**](https://realpython.com/python-virtual-environments-a-primer/).
- [Git](https://git-scm.com/)

Other considerations:

- If you're running Windows, I highly recommend using [Cmder](https://cmder.net/) as your console emulator. It comes bundled with [Git](https://git-scm.com/), and will be less frustrating than using the default Windows console.
- I also recommend using [VSCode](https://code.visualstudio.com/) as your editor. Of course you're free to use whatever editor you want!

### Getting Started
First, [fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) this repository, then fire up your command prompt and ...

1. Clone the forked repository
2. Navigate to the cloned project directory: `cd e3psi`
3. activate your python virtual environment and `pip install -r requirements.txt`.
4. `pre-commit install`
5. `pre-commit install --hook-type commit-msg`
6. `pre-commit run --all-files`

Now you can start working on the code.

### Tests

Simply run `pytest`. For more detailed output, including test coverage:

```sh
pytest -vv --cov=. --cov-report term-missing
```

### Contributing

If you would like to contribute to the project:

- if you're making code contributions, please try and write some tests to accompany your code, and ensure that the tests pass.
- commit your changes via `cz commit`. Follow the prompts. When you're done, `pre-commit` will be invoked to ensure that your contributions and commits follow defined conventions. See `pre-commit-config.yaml` for more details.
- your commit messages should follow the conventions described [here](https://www.conventionalcommits.org/en/v1.0.0/). Write your commit message in the imperative: "Fix bug" and not "Fixed bug" or "Fixes bug." This convention matches up with commit messages generated by commands like `git merge` and `git revert`.
Once you are done, please create a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


----
